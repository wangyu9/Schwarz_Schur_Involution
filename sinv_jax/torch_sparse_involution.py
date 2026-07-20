"""
JAX port of sinv/torch_sparse_involution.py.

The module name is kept for parity with the original import surface
(``from sinv_jax import torch_sparse_involution as tsi``). Internally
everything uses jax.numpy.

Translation conventions
-----------------------
- In-place mutation `x[..., a, b] = y`    -> `x = x.at[..., a, b].set(y)`
- In-place mutation `x[..., a, b] += y`   -> `x = x.at[..., a, b].add(y)`
- `.clone()`                              -> dropped (JAX arrays are immutable)
- `torch.linalg.inv_ex(X)[0]`             -> `jnp.linalg.inv(X)`
- `tensor.transpose(-2, -3)`              -> `jnp.swapaxes(tensor, -2, -3)`
- `tensor.unfold(-1, size, step)`         -> `_unfold_last_two_windows(...)`
- `torch.cuda.synchronize()`              -> dropped; callers use `block_until_ready`
"""

import math
import time
import gc
import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from . import algorithms as alg

DEFAULT_BOUNDARY_FIRST_ORDER = False


# ----------------------------------------------------------------------
# persistent XLA compilation cache
# ----------------------------------------------------------------------

# Anchor the cache to the repo root (sinvdev/.jax_cache), independent of the
# current working directory. On the first run JAX compiles and writes the
# executable here; later runs with an unchanged computation (same jaxpr,
# shapes/dtypes, JAX version, GPU arch) load it from disk instead of
# recompiling. Override with the SINV_JAX_CACHE env var.
_JAX_CACHE_DIR = os.environ.get(
    "SINV_JAX_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".jax_cache"),
)
_CACHE_CONFIGURED = False
_CACHE_STATS = {"hits": 0, "misses": 0, "saved_sec": 0.0}
_CACHE_SUMMARY_PRINTED = False


def _verbose():
    """Cache logging is off by default; set SINV_JAX_VERBOSE=1 to enable."""
    return bool(int(os.environ.get("SINV_JAX_VERBOSE", "0") or "0"))


def _print_cache_summary():
    """Report persistent-cache hits/misses once, at process exit.

    A JAX ``@jax.jit`` program lowers to several XLA executables, each looked
    up in the persistent cache independently, so a single solve emits multiple
    hit/miss events. We tally them and print one summary (like ``sinv_cuda``'s
    load message) rather than a line per executable. Only hits are announced
    when everything is cached; a nonzero miss count means something recompiled
    (new shape/dtype/ab_wh, changed source, or a JAX/GPU-arch change).
    """
    global _CACHE_SUMMARY_PRINTED
    if _CACHE_SUMMARY_PRINTED:
        return
    if not _verbose():
        return
    h, mi, saved = (_CACHE_STATS["hits"], _CACHE_STATS["misses"],
                    _CACHE_STATS["saved_sec"])
    if h == 0 and mi == 0:
        return  # no compiles happened this process
    _CACHE_SUMMARY_PRINTED = True
    import sys
    if mi == 0:
        print(f"[sinv_jax] JAX compilation cache HIT "
              f"({h} executables loaded from {_JAX_CACHE_DIR}, "
              f"~{saved:.1f}s compile saved)",
              file=sys.stderr, flush=True)
    else:
        print(f"[sinv_jax] JAX compilation cache: {h} hit / {mi} miss "
              f"(dir {_JAX_CACHE_DIR}; {mi} program(s) recompiled — new "
              f"shape/dtype/ab_wh, changed source, or JAX/GPU-arch change)",
              file=sys.stderr, flush=True)


def configure_compilation_cache():
    """Point JAX's persistent compilation cache at ``sinvdev/.jax_cache``.

    Idempotent and safe to call from anywhere; must run before the first
    compile (module import satisfies that). Shared by both ``sinv_jax`` and
    ``sinv_torch_via_jax`` so a cache written by one is loaded by the other.

    Silent by default. Set SINV_JAX_VERBOSE=1 to log the cache location and a
    one-line hit/miss summary at exit (announcing a cache HIT when nothing
    recompiled).
    """
    global _CACHE_CONFIGURED
    if _CACHE_CONFIGURED:
        return
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
    # 0 = cache every program, including fast-to-compile ones (default 1.0s).
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    _CACHE_CONFIGURED = True

    if not _verbose():
        return

    import sys
    print(f"[sinv_jax] using JAX compilation cache at {_JAX_CACHE_DIR}",
          file=sys.stderr, flush=True)

    import atexit
    from jax import monitoring

    def _on_event(event, **_kw):
        if event == "/jax/compilation_cache/cache_hits":
            _CACHE_STATS["hits"] += 1
        elif event == "/jax/compilation_cache/cache_misses":
            _CACHE_STATS["misses"] += 1

    def _on_duration(event, duration, **_kw):
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            _CACHE_STATS["saved_sec"] += duration

    monitoring.register_event_listener(_on_event)
    monitoring.register_event_duration_secs_listener(_on_duration)
    atexit.register(_print_cache_summary)


configure_compilation_cache()


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _mm(A, B):
    return A @ B


def hermitian(XX):
    """Conjugate transpose over the last two axes."""
    return jnp.conj(jnp.swapaxes(XX, -1, -2))


def _is_complex(x):
    return jnp.iscomplexobj(x)


def _unfold_last_two_windows(arr, size, step):
    """torch's `arr.unfold(-1, size, step)` for the specific case of two
    windows that share a midpoint — which is the only thing top_down needs.

    arr shape (..., L) -> output shape (..., 2, size).
    Equivalent to torch's unfold(-1, size, step) when L == 2*step+1 and
    size == step+1.
    """
    return jnp.stack([arr[..., 0:size], arr[..., step:step + size]], axis=-2)


# ----------------------------------------------------------------------
# Boundary-first ordering helpers
# ----------------------------------------------------------------------

def convert_to_boundary_first_ordering(alpha, wh):
    PB, PUB = alg.matrix_bdr_interior_split(wh=wh)
    ind = jnp.asarray(PB + PUB, dtype=jnp.int64)
    return alpha[..., ind, :][..., :, ind]


def convert_to_boundary_first_ordering_rhs(beta, wh):
    PB, PUB = alg.matrix_bdr_interior_split(wh=wh)
    ind = jnp.asarray(PB + PUB, dtype=jnp.int64)
    return beta[..., ind, :]


def convert_to_lexicographic_sweeping_order(alpha, wh):
    B, UB = alg.matrix_bdr_interior_split(wh=wh)
    invBUB = alg.inverse_permutation(B + UB)
    invd = jnp.asarray(invBUB, dtype=jnp.int64)
    return alpha[..., invd, :][..., :, invd]


# ----------------------------------------------------------------------
# Misc tensor helpers
# ----------------------------------------------------------------------

def torch_batch_cartesian_prod(A, B):
    if _is_complex(B) or _is_complex(A):
        print('warning: double check with your use case')
    assert A.shape[-1] == 1
    assert B.shape[-1] == 1
    dims = len(A.shape[:-2])
    assert dims == len(B.shape[:-2])
    a = A.shape[-2]
    b = B.shape[-2]
    BT = jnp.swapaxes(B, -1, -2)
    return jnp.broadcast_to(A, A.shape[:-1] + (b,)), \
           jnp.broadcast_to(BT, BT.shape[:-2] + (a, b))


def torch_batch_outer(A, B):
    assert len(A.shape) == len(B.shape)
    assert A.shape == B.shape
    return _mm(A, hermitian(B))


def inv_default_exact(xx):
    # jnp.linalg.inv (cuBLAS getrf) does not support fp16.  Promote to fp32,
    # invert, then cast back so the rest of the computation stays in fp16.
    if xx.dtype == jnp.float16:
        return jnp.linalg.inv(xx.astype(jnp.float32)).astype(jnp.float16)
    return jnp.linalg.inv(xx)


def fetch_DtN_exact(xx):
    return xx


def fetch_DtN_approx(xx):
    win = 256
    if xx.shape[-1] < (win - 1):
        return xx
    n = xx.shape[-1]
    yy = xx \
        - jnp.tril(xx, k=-win) - jnp.triu(xx, k=win) \
        + jnp.tril(xx, k=-(n - win)) + jnp.triu(xx, k=(n - win))
    return yy


def inv_default_approx(xx):
    print('approx inv called')
    win = 256
    if xx.shape[-1] < (win - 1):
        if xx.dtype == jnp.float16:
            return jnp.linalg.inv(xx.astype(jnp.float32)).astype(jnp.float16)
        return jnp.linalg.inv(xx)
    n = xx.shape[-1]
    yy = xx \
        - jnp.tril(xx, k=-win) - jnp.triu(xx, k=win) \
        + jnp.tril(xx, k=-(n - win)) + jnp.triu(xx, k=(n - win))
    if yy.dtype == jnp.float16:
        return jnp.linalg.inv(yy.astype(jnp.float32)).astype(jnp.float16)
    return jnp.linalg.inv(yy)


inv_default = inv_default_exact
fetch_DtN = fetch_DtN_exact


def unpack_dims(ab_wh):
    ab = ab_wh[0]
    wh = ab_wh[1]
    w = wh[0]
    h = wh[1]
    a = ab[0]
    b = ab[1]
    W = a * (w - 1) + 1
    H = b * (h - 1) + 1
    return a, b, w, h, W, H


def image_to_torch_batch(image):
    """Kept for API parity. Returns a JAX array."""
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    image_batch = image[np.newaxis, ...]
    return jnp.asarray(image_batch)


class Level():
    name = 'Level_'


def cf(arr, same=True):
    if not same:
        arr = jnp.flip(arr, axis=-2)
    return arr


def df(arr, same=[True, False]):
    axes = []
    if not same[0]:
        axes.append(-2)
    if not same[1]:
        axes.append(-1)
    if len(axes) > 0:
        arr = jnp.flip(arr, axis=tuple(axes))
    return arr


def image_flatten(ZZ):
    return jnp.reshape(jnp.swapaxes(ZZ, -2, -3),
                       list(ZZ.shape[:-3]) + [ZZ.shape[-2] * ZZ.shape[-3], -1])


def image_unflatten(Z, dim0, dim1):
    return jnp.swapaxes(jnp.reshape(Z, list(Z.shape[:-2]) + [dim1, dim0, -1]),
                        -2, -3)


def get_boundary(X_uv, size):
    m = size[0]
    n = size[1]
    return jnp.concatenate([
        X_uv[..., 0:m - 1, 0, :],
        X_uv[..., m - 1, 0:n - 1, :],
        jnp.flip(X_uv[..., 1:m, n - 1, :], axis=-2),
        jnp.flip(X_uv[..., 0, 1:n, :], axis=-2),
    ], axis=-2)


def flatten_and_permute(X_uv, pq):
    p, q = pq[0], pq[1]
    assert X_uv.shape[-3] == p
    assert X_uv.shape[-2] == q
    X_B = get_boundary(X_uv, size=[p, q])
    X_UB = jnp.swapaxes(X_uv[..., 1:p - 1, 1:q - 1, :], -3, -2)
    X_UB = jnp.reshape(
        X_UB,
        X_UB.shape[:-3] + (X_UB.shape[-3] * X_UB.shape[-2], X_UB.shape[-1]),
    )
    return jnp.concatenate([X_B, X_UB], axis=-2)


def fillin_bfs_order(BC, Z, p, q):
    Z_uv = image_unflatten(Z, p - 2, q - 2)
    dtype = Z.dtype
    sz = list(Z_uv.shape)
    sz[-2] = sz[-2] + 2
    sz[-3] = sz[-3] + 2
    X_uv = jnp.zeros(sz, dtype=dtype)
    X_uv = X_uv.at[..., 1:-1, 1:-1, :].set(Z_uv)
    X_uv = X_uv.at[..., 0:p - 1, 0, :].set(BC[..., 0:p - 1, :])
    X_uv = X_uv.at[..., -1, 0:q - 1, :].set(BC[..., p - 1:p + q - 2, :])
    X_uv = X_uv.at[..., 1:p, -1, :].set(jnp.flip(BC[..., p + q - 2:2 * p + q - 3, :], axis=-2))
    X_uv = X_uv.at[..., 0, 1:q, :].set(jnp.flip(BC[..., 2 * p + q - 3:, :], axis=-2))

    X = image_flatten(X_uv)
    return X, X_uv, Z_uv


def solution_fillin(sol, BC, Z, p, q):
    sol.X_p = jnp.concatenate([BC, Z], axis=-2)
    sol.X, sol.X_uv, sol.Z_uv = fillin_bfs_order(BC, Z, p, q)


# ----------------------------------------------------------------------
# Schwarz step: scatter boundary elimination
# ----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("p", "q", "transposed_solve"))
def _scatter_compute(DtN, RHS, p, q, transposed_solve):
    """Pure-tensor inner of scatter_boundary_eliminate_pre.

    Returns the updated (DtN, RHS) along with the cached arrays the four
    back_fill closures need.
    """
    # ---- side 1 ----
    ind = 0
    ka = slice(0, 1); kb = slice(p - 1, None); rm = slice(1, p - 1)
    invA1 = inv_default(DtN[:, :, ind, rm, rm])
    Ba1 = DtN[:, :, ind, rm, ka]
    Bb1 = DtN[:, :, ind, rm, kb]
    Ca1 = DtN[:, :, ind, ka, rm]
    Cb1 = DtN[:, :, ind, kb, rm]
    uu1 = RHS[:, :, ind, rm, :]

    DtN = DtN.at[:, :, ind, ka, ka].add(-(Ca1 @ invA1 @ Ba1))
    DtN = DtN.at[:, :, ind, ka, kb].add(-(Ca1 @ invA1 @ Bb1))
    DtN = DtN.at[:, :, ind, kb, ka].add(-(Cb1 @ invA1 @ Ba1))
    DtN = DtN.at[:, :, ind, kb, kb].add(-(Cb1 @ invA1 @ Bb1))
    if not transposed_solve:
        RHS = RHS.at[:, :, ind, ka, :].add(-(Ca1 @ (invA1 @ uu1)))
        RHS = RHS.at[:, :, ind, kb, :].add(-(Cb1 @ (invA1 @ uu1)))
    else:
        RHS = RHS.at[:, :, ind, ka, :].add(-hermitian((hermitian(uu1) @ invA1) @ Ba1))
        RHS = RHS.at[:, :, ind, kb, :].add(-hermitian((hermitian(uu1) @ invA1) @ Bb1))

    # ---- side 2 ----
    ind = -1
    ka2 = slice(0, p + q - 1); kb2 = slice(2 * p + q - 3, None); rm2 = slice(p + q - 1, 2 * p + q - 3)
    invA2 = inv_default(DtN[:, :, ind, rm2, rm2])
    Ba2 = DtN[:, :, ind, rm2, ka2]
    Bb2 = DtN[:, :, ind, rm2, kb2]
    Ca2 = DtN[:, :, ind, ka2, rm2]
    Cb2 = DtN[:, :, ind, kb2, rm2]
    uu2 = RHS[:, :, ind, rm2, :]

    DtN = DtN.at[:, :, ind, ka2, ka2].add(-(Ca2 @ invA2 @ Ba2))
    DtN = DtN.at[:, :, ind, ka2, kb2].add(-(Ca2 @ invA2 @ Bb2))
    DtN = DtN.at[:, :, ind, kb2, ka2].add(-(Cb2 @ invA2 @ Ba2))
    DtN = DtN.at[:, :, ind, kb2, kb2].add(-(Cb2 @ invA2 @ Bb2))
    if not transposed_solve:
        RHS = RHS.at[:, :, ind, ka2, :].add(-(Ca2 @ (invA2 @ uu2)))
        RHS = RHS.at[:, :, ind, kb2, :].add(-(Cb2 @ (invA2 @ uu2)))
    else:
        RHS = RHS.at[:, :, ind, ka2, :].add(-hermitian((hermitian(uu2) @ invA2) @ Ba2))
        RHS = RHS.at[:, :, ind, kb2, :].add(-hermitian((hermitian(uu2) @ invA2) @ Bb2))

    # ---- side 3 ----
    ind3 = -1
    ka3 = slice(0, p); kb3 = slice(p + q - 2, None); rm3 = slice(p, p + q - 2)
    invA3 = inv_default(DtN[:, ind3, :, rm3, rm3])
    Ba3 = DtN[:, ind3, :, rm3, ka3]
    Bb3 = DtN[:, ind3, :, rm3, kb3]
    Ca3 = DtN[:, ind3, :, ka3, rm3]
    Cb3 = DtN[:, ind3, :, kb3, rm3]
    uu3 = RHS[:, ind3, :, rm3, :]

    DtN = DtN.at[:, ind3, :, ka3, ka3].add(-(Ca3 @ invA3 @ Ba3))
    DtN = DtN.at[:, ind3, :, ka3, kb3].add(-(Ca3 @ invA3 @ Bb3))
    DtN = DtN.at[:, ind3, :, kb3, ka3].add(-(Cb3 @ invA3 @ Ba3))
    DtN = DtN.at[:, ind3, :, kb3, kb3].add(-(Cb3 @ invA3 @ Bb3))
    if not transposed_solve:
        RHS = RHS.at[:, ind3, :, ka3, :].add(-(Ca3 @ (invA3 @ uu3)))
        RHS = RHS.at[:, ind3, :, kb3, :].add(-(Cb3 @ (invA3 @ uu3)))
    else:
        RHS = RHS.at[:, ind3, :, ka3, :].add(-hermitian((hermitian(uu3) @ invA3) @ Ba3))
        RHS = RHS.at[:, ind3, :, kb3, :].add(-hermitian((hermitian(uu3) @ invA3) @ Bb3))

    # ---- side 4 ----
    ind4 = 0
    ka4 = slice(0, 1); kb4 = slice(1, 2 * p + q - 2); rm4 = slice(2 * p + q - 2, None)
    invA4 = inv_default(DtN[:, ind4, :, rm4, rm4])
    Ba4 = DtN[:, ind4, :, rm4, ka4]
    Bb4 = DtN[:, ind4, :, rm4, kb4]
    Ca4 = DtN[:, ind4, :, ka4, rm4]
    Cb4 = DtN[:, ind4, :, kb4, rm4]
    uu4 = RHS[:, ind4, :, rm4, :]

    DtN = DtN.at[:, ind4, :, ka4, ka4].add(-(Ca4 @ invA4 @ Ba4))
    DtN = DtN.at[:, ind4, :, ka4, kb4].add(-(Ca4 @ invA4 @ Bb4))
    DtN = DtN.at[:, ind4, :, kb4, ka4].add(-(Cb4 @ invA4 @ Ba4))
    DtN = DtN.at[:, ind4, :, kb4, kb4].add(-(Cb4 @ invA4 @ Bb4))
    if not transposed_solve:
        RHS = RHS.at[:, ind4, :, ka4, :].add(-(Ca4 @ (invA4 @ uu4)))
        RHS = RHS.at[:, ind4, :, kb4, :].add(-(Cb4 @ (invA4 @ uu4)))
    else:
        RHS = RHS.at[:, ind4, :, ka4, :].add(-hermitian((hermitian(uu4) @ invA4) @ Ba4))
        RHS = RHS.at[:, ind4, :, kb4, :].add(-hermitian((hermitian(uu4) @ invA4) @ Bb4))

    return (DtN, RHS,
            invA1, Ba1, Bb1, Ca1, Cb1, uu1,
            invA2, Ba2, Bb2, Ca2, Cb2, uu2,
            invA3, Ba3, Bb3, Ca3, Cb3, uu3,
            invA4, Ba4, Bb4, Ca4, Cb4, uu4)


@partial(jax.jit, static_argnames=("p", "q", "transposed_solve"))
def _scatter_back_fill(nBC, p, q, transposed_solve,
                       invA1, Ba1, Bb1, Ca1, Cb1, uu1,
                       invA2, Ba2, Bb2, Ca2, Cb2, uu2,
                       invA3, Ba3, Bb3, Ca3, Cb3, uu3,
                       invA4, Ba4, Bb4, Ca4, Cb4, uu4):
    """All four back-fill steps fused into one jitted call. The original
    code applied them in reverse order (4, 3, 2, 1)."""
    # ---- side 4 ----
    ind = 0
    ka = slice(0, 1); kb = slice(1, 2 * p + q - 2); rm = slice(2 * p + q - 2, None)
    ya = nBC[:, ind, :, ka, :]; yb = nBC[:, ind, :, kb, :]
    if not transposed_solve:
        new = invA4 @ (uu4 - Ba4 @ ya - Bb4 @ yb)
    else:
        new = hermitian((hermitian(uu4) - hermitian(ya) @ Ca4 - hermitian(yb) @ Cb4) @ invA4)
    nBC = nBC.at[:, ind, :, rm, :].set(new)

    # ---- side 3 ----
    ind = -1
    ka = slice(0, p); kb = slice(p + q - 2, None); rm = slice(p, p + q - 2)
    ya = nBC[:, ind, :, ka, :]; yb = nBC[:, ind, :, kb, :]
    if not transposed_solve:
        new = invA3 @ (uu3 - Ba3 @ ya - Bb3 @ yb)
    else:
        new = hermitian((hermitian(uu3) - hermitian(ya) @ Ca3 - hermitian(yb) @ Cb3) @ invA3)
    nBC = nBC.at[:, ind, :, rm, :].set(new)

    # ---- side 2 ----
    ind = -1
    ka = slice(0, p + q - 1); kb = slice(2 * p + q - 3, None); rm = slice(p + q - 1, 2 * p + q - 3)
    ya = nBC[:, :, ind, ka, :]; yb = nBC[:, :, ind, kb, :]
    if not transposed_solve:
        new = invA2 @ (uu2 - Ba2 @ ya - Bb2 @ yb)
    else:
        new = hermitian((hermitian(uu2) - hermitian(ya) @ Ca2 - hermitian(yb) @ Cb2) @ invA2)
    nBC = nBC.at[:, :, ind, rm, :].set(new)

    # ---- side 1 ----
    ind = 0
    ka = slice(0, 1); kb = slice(p - 1, None); rm = slice(1, p - 1)
    ya = nBC[:, :, ind, ka, :]; yb = nBC[:, :, ind, kb, :]
    if not transposed_solve:
        new = invA1 @ (uu1 - Ba1 @ ya - Bb1 @ yb)
    else:
        new = hermitian((hermitian(uu1) - hermitian(ya) @ Ca1 - hermitian(yb) @ Cb1) @ invA1)
    nBC = nBC.at[:, :, ind, rm, :].set(new)

    return nBC


# ----------------------------------------------------------------------

def scatter_boundary_eliminate_pre(
    dim_pq, DtN_ori, RHS_ori,
    pre_sol=None,
    transposed_solve=False,
    cache=True,
    invf=inv_default,
):
    class Solution:
        name = 'scatter_boundary_eliminate_pre'

    sol = Solution()

    if cache is False:
        pre_sol = None
        assert transposed_solve is False

    p, q = dim_pq[0], dim_pq[1]

    if pre_sol is None:
        (DtN, RHS,
         invA1, Ba1, Bb1, Ca1, Cb1, uu1,
         invA2, Ba2, Bb2, Ca2, Cb2, uu2,
         invA3, Ba3, Bb3, Ca3, Cb3, uu3,
         invA4, Ba4, Bb4, Ca4, Cb4, uu4) = _scatter_compute(
            DtN_ori, RHS_ori, p, q, transposed_solve)
    else:
        # Re-run the same compute path so the JIT cache hits the same
        # compiled program; only the output (DtN, RHS) and uu? matter on
        # back-sub. We reuse the cached invA / Ba / Bb / Ca / Cb arrays.
        (_, RHS,
         _, _, _, _, _, uu1,
         _, _, _, _, _, uu2,
         _, _, _, _, _, uu3,
         _, _, _, _, _, uu4) = _scatter_compute(
            pre_sol.DtN_kept_in, RHS_ori, p, q, transposed_solve)
        DtN = None
        invA1, Ba1, Bb1, Ca1, Cb1 = pre_sol.invA1, pre_sol.Ba1, pre_sol.Bb1, pre_sol.Ca1, pre_sol.Cb1
        invA2, Ba2, Bb2, Ca2, Cb2 = pre_sol.invA2, pre_sol.Ba2, pre_sol.Bb2, pre_sol.Ca2, pre_sol.Cb2
        invA3, Ba3, Bb3, Ca3, Cb3 = pre_sol.invA3, pre_sol.Ba3, pre_sol.Bb3, pre_sol.Ca3, pre_sol.Cb3
        invA4, Ba4, Bb4, Ca4, Cb4 = pre_sol.invA4, pre_sol.Ba4, pre_sol.Bb4, pre_sol.Ca4, pre_sol.Cb4

    sol.DtN_kept_in = DtN_ori
    sol.invA1 = invA1; sol.Ba1 = Ba1; sol.Bb1 = Bb1; sol.Ca1 = Ca1; sol.Cb1 = Cb1
    sol.invA2 = invA2; sol.Ba2 = Ba2; sol.Bb2 = Bb2; sol.Ca2 = Ca2; sol.Cb2 = Cb2
    sol.invA3 = invA3; sol.Ba3 = Ba3; sol.Bb3 = Bb3; sol.Ca3 = Ca3; sol.Cb3 = Cb3
    sol.invA4 = invA4; sol.Ba4 = Ba4; sol.Bb4 = Bb4; sol.Ca4 = Ca4; sol.Cb4 = Cb4

    def nBC_back_fill(nBC, transposed_solve):
        return _scatter_back_fill(nBC, p, q, transposed_solve,
            invA1, Ba1, Bb1, Ca1, Cb1, uu1,
            invA2, Ba2, Bb2, Ca2, Cb2, uu2,
            invA3, Ba3, Bb3, Ca3, Cb3, uu3,
            invA4, Ba4, Bb4, Ca4, Cb4, uu4)

    return DtN, RHS, nBC_back_fill, sol


def scatter_boundary_eliminate_pre_unused(*args, **kwargs):
    raise NotImplementedError(
        "Replaced by the jitted `scatter_boundary_eliminate_pre` above; "
        "kept only as a name placeholder."
    )


# ----------------------------------------------------------------------
# Per-patch Dirichlet (the inner block elimination)
# ----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("irs",))
def _dirichlet_pre_compute(dLA, irs):
    """Pure inner of batch_Dirichlet_solve_pre. Returns:
    (DtN, dLA_rr, dLA_rs, dLA_sr, dLA_ss, inv_dLA_ss, reused_prod)
    """
    dLA_rr = dLA[..., 0:irs, 0:irs]
    dLA_rs = dLA[..., 0:irs, irs:]
    dLA_sr = dLA[..., irs:, 0:irs]
    dLA_ss = dLA[..., irs:, irs:]
    inv_dLA_ss = inv_default(dLA_ss)
    reused_prod = dLA_rs @ inv_dLA_ss
    DtN = dLA_rr - reused_prod @ dLA_sr
    return DtN, dLA_rr, dLA_rs, dLA_sr, dLA_ss, inv_dLA_ss, reused_prod


def batch_Dirichlet_solve_pre(dim_pq, dLA, invf=inv_default):
    p = dim_pq[0]
    q = dim_pq[1]
    irs = 2 * (p + q) - 4

    DtN, dLA_rr, dLA_rs, dLA_sr, dLA_ss, inv_dLA_ss, reused_prod = \
        _dirichlet_pre_compute(dLA, irs)

    def split_RHS(RHS):
        RHS_r = RHS[..., 0:irs, :]
        RHS_s = RHS[..., irs:, :]
        return RHS_s, RHS_r

    @partial(jax.jit, static_argnames=("transposed_solve",))
    def _get_nRHS_jit(RHS, transposed_solve):
        RHS_s = RHS[..., irs:, :]
        RHS_r = RHS[..., 0:irs, :]
        if not transposed_solve:
            return RHS_r - reused_prod @ RHS_s
        return RHS_r - hermitian((hermitian(RHS_s) @ inv_dLA_ss) @ dLA_sr)

    def get_nRHS(RHS, transposed_solve=False):
        return _get_nRHS_jit(RHS, transposed_solve)

    @partial(jax.jit, static_argnames=("transposed_solve",))
    def _solver_Z_jit(Y, RHS, transposed_solve):
        RHS_s = RHS[..., irs:, :]
        if not transposed_solve:
            return inv_dLA_ss @ (RHS_s - dLA_sr @ Y)
        return hermitian((hermitian(RHS_s) - hermitian(Y) @ dLA_rs) @ inv_dLA_ss)

    class Solution:
        name = 'Dirichlet_solve: solution'

    def solver(BC, RHS, compute_energy=True, transposed_solve=False):
        sol = Solution()
        sol.Y = BC
        sol.Z = _solver_Z_jit(sol.Y, RHS, transposed_solve)
        solution_fillin(sol, sol.Y, sol.Z, p=p, q=q)

        if compute_energy:
            def eval_ADirichlet():
                assert transposed_solve is False
                assert _is_complex(sol.Y) is False
                AD = hermitian(sol.Y) @ (dLA_rr @ sol.Y + dLA_rs @ sol.Z)
                assert AD.shape[-2] == 2
                assert AD.shape[-1] == 2
                return AD[..., 0, 0] + AD[..., 1, 1]
            sol.eval_ADirichlet = eval_ADirichlet

        return sol

    return DtN, get_nRHS, solver


# ----------------------------------------------------------------------
# bottom-up / top-down hierarchy on patches
# ----------------------------------------------------------------------

def merge_blocks_split_value(CX, dim=0, coeff=0.5):
    if dim == 0:
        return jnp.concatenate([
            CX[:, 0::2, :, :-1, ...],
            coeff * (CX[:, 0::2, :, [-1], ...] + CX[:, 1::2, :, [0], ...]),
            CX[:, 1::2, :, 1:, ...],
        ], axis=3)
    else:
        assert dim == 1
        return jnp.concatenate([
            CX[:, :, 0::2, :, :-1, ...],
            coeff * (CX[:, :, 0::2, :, [-1], ...] + CX[:, :, 1::2, :, [0], ...]),
            CX[:, :, 1::2, :, 1:, ...],
        ], axis=4)


def bottom_up(XX, value_sum=True):
    size = XX.shape
    a = size[1]
    b = size[2]
    N_iter = math.floor(math.log2(a) + math.log2(b))
    CX = XX
    coeff = 1 if value_sum else 0.5
    for i in range(N_iter):
        if i % 2 == 0:
            CX = merge_blocks_split_value(CX, dim=0, coeff=coeff)
        else:
            CX = merge_blocks_split_value(CX, dim=1, coeff=coeff)
    return CX


def top_down(XX, div_level='Auto', value_divide=True, ab=None):
    if div_level == 'Auto':
        a = ab[0]
        b = ab[1]
        div_level = (math.log2(a) + math.log2(b)) / 2
        assert 12 == 12.0
        div_level = int(div_level)

    CX = XX
    for _ in range(div_level):
        cp = CX.shape[3] - 1
        cq = CX.shape[4] - 1
        assert cp % 2 == 0
        assert cq % 2 == 0
        cp = cp // 2
        cq = cq // 2

        # y axis: permute to put q last, halve the midpoint, split into 2 halves, restore
        CX = jnp.transpose(CX, (0, 5, 1, 3, 2, 4))
        if value_divide:
            CX = CX.at[..., cq].set(CX[..., cq] / 2.0)
        # split last axis (q, length 2*cq+1) into 2 windows of size cq+1 sharing the midpoint
        CX = _unfold_last_two_windows(CX, size=cq + 1, step=cq)
        # collapse the new window axis into the b axis (axis 4 with axis 5)
        CX = CX.reshape(CX.shape[:4] + (CX.shape[4] * CX.shape[5], CX.shape[6]))
        CX = jnp.transpose(CX, (0, 2, 4, 3, 5, 1))

        # x axis
        CX = jnp.transpose(CX, (0, 5, 2, 4, 1, 3))
        if value_divide:
            CX = CX.at[..., cp].set(CX[..., cp] / 2.0)
        CX = _unfold_last_two_windows(CX, size=cp + 1, step=cp)
        CX = CX.reshape(CX.shape[:4] + (CX.shape[4] * CX.shape[5], CX.shape[6]))
        CX = jnp.transpose(CX, (0, 4, 2, 5, 3, 1))
    return CX


def beta_sync_across_patches(Beta, wh):
    a = Beta.shape[1]
    b = Beta.shape[2]
    Chi = image_flatten(
        top_down(
            bottom_up(
                image_unflatten(Beta, wh[0], wh[1]),
                value_sum=True,
            ),
            value_divide=False, ab=(a, b),
        )
    )
    return Chi


chi_from_beta = beta_sync_across_patches


# ----------------------------------------------------------------------
# Schur step: collapse_subdomains
# ----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("p", "q", "transposed_solve"))
def _collapse_compute_dim0(DtN, RHS, p, q, transposed_solve):
    """Pure-tensor inner of collapse_subdomains for dim==0.

    Returns: (mDtN, mRHS, invSS, invSSmSR, RS, fS)
    """
    DtNa = DtN[..., 0::2, :, :, :]
    DtNb = DtN[..., 1::2, :, :, :]

    m1 = p
    m4 = 3 * p + q

    ai = slice(0, p + 1)
    aj = slice(p + 1, p + q)
    ak = slice(p + q, None)
    bu = slice(0, 2 * p + q + 1)
    bv = slice(2 * p + q + 1, None)
    ri = slice(0, m1 + 1)
    ru = slice(m1, m4 + 1)
    rk = slice(m4, None)

    batch_size, m_a, n_a = DtNa.shape[0], DtNa.shape[1], DtNa.shape[2]
    dtype = DtN.dtype
    SS = DtNa[..., aj, aj] + df(DtNb[..., bv, bv], [False, False])

    RR = jnp.zeros((batch_size, m_a, n_a, 2 * (2 * p + q), 2 * (2 * p + q)), dtype=dtype)
    SR = jnp.zeros((batch_size, m_a, n_a, q - 1, 2 * (2 * p + q)), dtype=dtype)
    RS = jnp.zeros((batch_size, m_a, n_a, 2 * (2 * p + q), q - 1), dtype=dtype)

    RR = RR.at[..., ri, ri].add(DtNa[..., ai, ai])
    RR = RR.at[..., rk, rk].add(DtNa[..., ak, ak])
    RR = RR.at[..., ri, rk].add(DtNa[..., ai, ak])
    RR = RR.at[..., rk, ri].add(DtNa[..., ak, ai])
    RS = RS.at[..., ri, :].add(DtNa[..., ai, aj])
    RS = RS.at[..., rk, :].add(DtNa[..., ak, aj])
    SR = SR.at[..., :, ri].add(DtNa[..., aj, ai])
    SR = SR.at[..., :, rk].add(DtNa[..., aj, ak])
    RR = RR.at[..., ru, ru].add(DtNb[..., bu, bu])
    RS = RS.at[..., ru, :].add(df(DtNb[..., bu, bv], [True, False]))
    SR = SR.at[..., :, ru].add(df(DtNb[..., bv, bu], [False, True]))

    RHSa = RHS[..., 0::2, :, :, :]
    RHSb = RHS[..., 1::2, :, :, :]
    c = RHS.shape[-1]
    fR = jnp.zeros((batch_size, m_a, n_a, 2 * (2 * p + q), c), dtype=dtype)
    fS = jnp.zeros((batch_size, m_a, n_a, q - 1, c), dtype=dtype)
    fR = fR.at[..., ri, :].add(RHSa[..., ai, :])
    fR = fR.at[..., rk, :].add(RHSa[..., ak, :])
    fS = fS.at[..., :, :].add(RHSa[..., aj, :])
    fR = fR.at[..., ru, :].add(RHSb[..., bu, :])
    fS = fS.at[..., :, :].add(cf(RHSb[..., bv, :], False))

    invSS = inv_default(SS)
    invSSmSR = invSS @ SR
    mDtN = RR - RS @ invSSmSR

    if not transposed_solve:
        mRHS = fR - RS @ (invSS @ fS)
    else:
        mRHS = fR - hermitian((hermitian(fS) @ invSS) @ SR)

    return mDtN, mRHS, invSS, invSSmSR, RS, fS


@partial(jax.jit, static_argnames=("p", "q", "transposed_solve"))
def _collapse_compute_dim1(DtN, RHS, p, q, transposed_solve):
    """Pure-tensor inner of collapse_subdomains for dim==1."""
    DtNa = DtN[..., :, 0::2, :, :]
    DtNb = DtN[..., :, 1::2, :, :]

    m2 = p + q
    m5 = 2 * p + 3 * q

    ai = slice(0, p + q + 1)
    aj = slice(p + q + 1, 2 * p + q)
    ak = slice(2 * p + q, None)
    bu = slice(0, 1)
    bv = slice(1, p)
    bw = slice(p, None)
    ri = slice(0, m2 + 1)
    rw = slice(m2, m5)
    ru = slice(m5, m5 + 1)
    rk = slice(m5, None)

    batch_size, m_a, n_a = DtNa.shape[0], DtNa.shape[1], DtNa.shape[2]
    dtype = DtN.dtype
    SS = DtNa[..., aj, aj] + df(DtNb[..., bv, bv], [False, False])

    RR = jnp.zeros((batch_size, m_a, n_a, 2 * (p + 2 * q), 2 * (p + 2 * q)), dtype=dtype)
    SR = jnp.zeros((batch_size, m_a, n_a, p - 1, 2 * (p + 2 * q)), dtype=dtype)
    RS = jnp.zeros((batch_size, m_a, n_a, 2 * (p + 2 * q), p - 1), dtype=dtype)

    RR = RR.at[..., ri, ri].add(DtNa[..., ai, ai])
    RR = RR.at[..., rk, rk].add(DtNa[..., ak, ak])
    RR = RR.at[..., ri, rk].add(DtNa[..., ai, ak])
    RR = RR.at[..., rk, ri].add(DtNa[..., ak, ai])
    RS = RS.at[..., ri, :].add(DtNa[..., ai, aj])
    RS = RS.at[..., rk, :].add(DtNa[..., ak, aj])
    SR = SR.at[..., :, ri].add(DtNa[..., aj, ai])
    SR = SR.at[..., :, rk].add(DtNa[..., aj, ak])
    RR = RR.at[..., ru, ru].add(DtNb[..., bu, bu])
    RR = RR.at[..., rw, rw].add(DtNb[..., bw, bw])
    RR = RR.at[..., ru, rw].add(DtNb[..., bu, bw])
    RR = RR.at[..., rw, ru].add(DtNb[..., bw, bu])
    RS = RS.at[..., ru, :].add(df(DtNb[..., bu, bv], [True, False]))
    SR = SR.at[..., :, ru].add(df(DtNb[..., bv, bu], [False, True]))
    RS = RS.at[..., rw, :].add(df(DtNb[..., bw, bv], [True, False]))
    SR = SR.at[..., :, rw].add(df(DtNb[..., bv, bw], [False, True]))

    RHSa = RHS[..., :, 0::2, :, :]
    RHSb = RHS[..., :, 1::2, :, :]
    c = RHS.shape[-1]
    fR = jnp.zeros((batch_size, m_a, n_a, 2 * (p + 2 * q), c), dtype=dtype)
    fS = jnp.zeros((batch_size, m_a, n_a, p - 1, c), dtype=dtype)
    fR = fR.at[..., ri, :].add(RHSa[..., ai, :])
    fR = fR.at[..., rk, :].add(RHSa[..., ak, :])
    fS = fS.at[..., :, :].add(RHSa[..., aj, :])
    fR = fR.at[..., ru, :].add(RHSb[..., bu, :])
    fR = fR.at[..., rw, :].add(RHSb[..., bw, :])
    fS = fS.at[..., :, :].add(cf(RHSb[..., bv, :], False))

    invSS = inv_default(SS)
    invSSmSR = invSS @ SR
    mDtN = RR - RS @ invSSmSR

    if not transposed_solve:
        mRHS = fR - RS @ (invSS @ fS)
    else:
        mRHS = fR - hermitian((hermitian(fS) @ invSS) @ SR)

    return mDtN, mRHS, invSS, invSSmSR, RS, fS


@partial(jax.jit, static_argnames=("p", "q", "transposed_solve"))
def _collapse_back_dim0(BC, fS, invSS, invSSmSR, RS, p, q, transposed_solve):
    """Pure-tensor inner of topdown_solve_Dirichlet for dim==0.

    Output shape: (B, 2*m_a, n_a, 2*(p+q), c).
    """
    batch_size = BC.shape[0]
    m_a = BC.shape[1]   # m_a was m//2; output's m == 2*m_a
    n_a = BC.shape[2]
    cc = BC.shape[-1]
    dtype = BC.dtype

    m1 = p
    m4 = 3 * p + q
    ai = slice(0, p + 1)
    aj = slice(p + 1, p + q)
    ak = slice(p + q, None)
    bu = slice(0, 2 * p + q + 1)
    bv = slice(2 * p + q + 1, None)
    ri = slice(0, m1 + 1)
    ru = slice(m1, m4 + 1)
    rk = slice(m4, None)

    if not transposed_solve:
        uS = invSS @ fS - invSSmSR @ BC
    else:
        uS = hermitian((hermitian(fS) - hermitian(BC) @ RS) @ invSS)

    nBC = jnp.zeros((batch_size, 2 * m_a, n_a, 2 * (p + q), cc), dtype=dtype)
    nBC = nBC.at[..., 0::2, :, ai, :].set(BC[..., ri, :])
    nBC = nBC.at[..., 0::2, :, ak, :].set(BC[..., rk, :])
    nBC = nBC.at[..., 0::2, :, aj, :].set(uS[..., :, :])
    nBC = nBC.at[..., 1::2, :, bu, :].set(BC[..., ru, :])
    nBC = nBC.at[..., 1::2, :, bv, :].set(cf(uS[..., :, :], False))
    return nBC


@partial(jax.jit, static_argnames=("p", "q", "transposed_solve"))
def _collapse_back_dim1(BC, fS, invSS, invSSmSR, RS, p, q, transposed_solve):
    """Pure-tensor inner of topdown_solve_Dirichlet for dim==1."""
    batch_size = BC.shape[0]
    m_a = BC.shape[1]
    n_a = BC.shape[2]   # n_a was n//2; output's n == 2*n_a
    cc = BC.shape[-1]
    dtype = BC.dtype

    m2 = p + q
    m5 = 2 * p + 3 * q
    ai = slice(0, p + q + 1)
    aj = slice(p + q + 1, 2 * p + q)
    ak = slice(2 * p + q, None)
    bu = slice(0, 1)
    bv = slice(1, p)
    bw = slice(p, None)
    ri = slice(0, m2 + 1)
    rw = slice(m2, m5)
    ru = slice(m5, m5 + 1)
    rk = slice(m5, None)

    if not transposed_solve:
        uS = invSS @ fS - invSSmSR @ BC
    else:
        uS = hermitian((hermitian(fS) - hermitian(BC) @ RS) @ invSS)

    nBC = jnp.zeros((batch_size, m_a, 2 * n_a, 2 * (p + q), cc), dtype=dtype)
    nBC = nBC.at[..., :, 0::2, ai, :].set(BC[..., ri, :])
    nBC = nBC.at[..., :, 0::2, ak, :].set(BC[..., rk, :])
    nBC = nBC.at[..., :, 0::2, aj, :].set(uS[..., :, :])
    nBC = nBC.at[..., :, 1::2, bu, :].set(BC[..., ru, :])
    nBC = nBC.at[..., :, 1::2, bw, :].set(BC[..., rw, :])
    nBC = nBC.at[..., :, 1::2, bv, :].set(cf(uS[..., :, :], False))
    return nBC


def collapse_subdomains(
    DtN, RHS, p, q, old_sol,
    dim=0,
    pre_sol=None,
    transposed_solve=False,
    debug_time=False,
    invf=None,
):
    sol = Level()

    if pre_sol is None:
        sol.name = 'Level_num_fact'
    else:
        sol.name = 'Level_back_sub'
        DtN = pre_sol.DtN

    if pre_sol is not None:
        # back-sub path: reuse the cached DtN from the forward call so the
        # jit cache hits the same compiled program (the autograd path keeps
        # `pre_sol.DtN_kept`).
        DtN_for_compute = pre_sol.DtN_kept
    else:
        DtN_for_compute = DtN

    if dim == 0:
        mDtN, mRHS, invSS, invSSmSR, RS, fS = _collapse_compute_dim0(
            DtN_for_compute, RHS, p, q, transposed_solve)
    else:
        mDtN, mRHS, invSS, invSSmSR, RS, fS = _collapse_compute_dim1(
            DtN_for_compute, RHS, p, q, transposed_solve)

    if pre_sol is not None:
        # Cached pre-factor takes precedence (mDtN is unused on back-sub anyway).
        invSS = pre_sol.invSS
        invSSmSR = pre_sol.invSSmSR
        RS = pre_sol.RS
        mDtN = None

    sol.DtN = DtN_for_compute
    sol.DtN_kept = DtN_for_compute
    sol.batch_size = DtN_for_compute.shape[0]
    sol.p = p
    sol.q = q
    sol.m = DtN_for_compute.shape[1]
    sol.n = DtN_for_compute.shape[2]
    sol.invSS = invSS
    sol.invSSmSR = invSSmSR
    sol.RS = RS

    if dim == 0:
        def topdown_solve_Dirichlet(BC, transposed_solve):
            return _collapse_back_dim0(BC, fS, invSS, invSSmSR, RS, p, q, transposed_solve)
    else:
        def topdown_solve_Dirichlet(BC, transposed_solve):
            return _collapse_back_dim1(BC, fS, invSS, invSSmSR, RS, p, q, transposed_solve)
    sol.topdown_solve_Dirichlet = topdown_solve_Dirichlet

    if dim == 0:
        return mDtN, mRHS, 2 * p, q, sol
    else:
        return mDtN, mRHS, p, 2 * q, sol


# ----------------------------------------------------------------------
# midpoint-reflective reduce
# ----------------------------------------------------------------------

def midpoint_reflective_reduce(m, n, ps, qs):
    assert m % 2 == 0
    assert n % 2 == 0
    mph = (m * ps) // 2
    nqh = (n * qs) // 2

    FR = jnp.zeros((4 * (mph + nqh), 2 * (mph + nqh) + 1))

    ua, ub, uc, ud = 0, mph, 2 * mph, 2 * mph + nqh
    ue, uf = 2 * mph + 2 * nqh, 3 * mph + 2 * nqh
    ug, uh = 4 * mph + 2 * nqh, 4 * mph + 3 * nqh
    uab = slice(ua + 1, ub)
    ubc = slice(ub + 1, uc)
    ucd = slice(uc + 1, ud)
    ude = slice(ud + 1, ue)
    uef = slice(ue + 1, uf)
    ufg = slice(uf + 1, ug)
    ugh = slice(ug + 1, uh)
    uha = slice(uh + 1, None)

    va, vb = 0, mph
    vd, vf = mph + nqh, 2 * mph + nqh
    vh = 2 * mph + 2 * nqh
    vab = slice(va + 1, vb)
    vcd = slice(vb + 1, vd)
    vef = slice(vd + 1, vf)
    vgh = slice(vf + 1, vh)

    FR = FR.at[ua, va].set(1).at[uc, va].set(1).at[ue, va].set(1).at[ug, va].set(1)
    FR = FR.at[ub, vb].set(1).at[ud, vd].set(1).at[uf, vf].set(1).at[uh, vh].set(1)

    def slen(s):
        return s.stop - s.start

    FR = FR.at[uab, vab].set(jnp.eye(slen(vab)))
    FR = FR.at[ucd, vcd].set(jnp.eye(slen(vcd)))
    FR = FR.at[uef, vef].set(jnp.eye(slen(vef)))
    FR = FR.at[ugh, vgh].set(jnp.eye(slen(vgh)))

    FR = FR.at[ubc, vab].set(jnp.flip(jnp.eye(slen(vab)), axis=0))
    FR = FR.at[ude, vcd].set(jnp.flip(jnp.eye(slen(vcd)), axis=0))
    FR = FR.at[ufg, vef].set(jnp.flip(jnp.eye(slen(vef)), axis=0))
    FR = FR.at[uha, vgh].set(jnp.flip(jnp.eye(slen(vgh)), axis=0))

    return FR[None, ...]


# ----------------------------------------------------------------------
# top-level solver
# ----------------------------------------------------------------------

def apply_all_schur_steps(
    DtN, RHS, BC, m, n, ps, qs,
    transposed_solve=False,
    debug=False,
    layers_pre=None,
    lastInvDtN_reuse=None,
    invf=inv_default,
):
    layers = [(DtN, RHS, ps, qs, [])]
    N_iter = math.floor(math.log2(m) + math.log2(n))

    for i in range(N_iter):
        dim = 0 if i % 2 == 0 else 1
        if layers_pre is None:
            DtN_new, RHS_new, ps_new, qs_new, sol = collapse_subdomains(
                *layers[i], dim=dim, pre_sol=None, transposed_solve=transposed_solve,
            )
        else:
            DtN_new, RHS_new, ps_new, qs_new, sol = collapse_subdomains(
                *layers[i], dim=dim,
                pre_sol=layers_pre[i + 1][-1], transposed_solve=transposed_solve,
            )
        sol.name += str(i)
        layers.append([DtN_new, RHS_new, ps_new, qs_new, sol])

    lastInvDtN = None

    if BC is None:
        assert False

    if BC == 'Neumann-scatter':
        if layers_pre is None:
            lhs = layers[-1][0]
        else:
            lhs = layers_pre[-1][0]
        rhs = layers[-1][1]
        assert ps == qs
        ss = slice(0, None, ps)
        BC = jnp.zeros(lhs.shape[:-1] + rhs.shape[-1:], dtype=lhs.dtype)
        if layers_pre is None:
            lastInvDtN = invf(lhs[..., ss, ss])
        else:
            lastInvDtN = lastInvDtN_reuse
        if not transposed_solve:
            BC = BC.at[..., ss, :].set(lastInvDtN @ rhs[..., ss, :])
        else:
            BC = BC.at[..., ss, :].set(hermitian(hermitian(rhs[..., ss, :]) @ lastInvDtN))

    elif BC == 'Neumann' or BC == 'Neumann-full':
        if layers_pre is None:
            lhs = layers[-1][0]
        else:
            lhs = layers_pre[-1][0]
        rhs = layers[-1][1]
        if layers_pre is None:
            lastInvDtN = invf(lhs)
        else:
            lastInvDtN = lastInvDtN_reuse
        if not transposed_solve:
            BC = lastInvDtN @ rhs
        else:
            BC = hermitian(hermitian(rhs) @ lastInvDtN)

    elif BC == 'midpoint-reflective':
        assert transposed_solve is False
        F_reduce = midpoint_reflective_reduce(m, n, ps, qs)
        if layers_pre is None:
            lhs = layers[-1][0]
        else:
            lhs = layers_pre[-1][0]
        rhs = layers[-1][1]
        F_reduce = F_reduce.astype(lhs.dtype)
        lhs = jnp.swapaxes(F_reduce, -1, -2) @ lhs @ F_reduce
        rhs = jnp.swapaxes(F_reduce, -1, -2) @ rhs
        if layers_pre is None:
            lastInvDtN = invf(lhs)
        else:
            lastInvDtN = lastInvDtN_reuse
        if not transposed_solve:
            BC = lastInvDtN @ rhs
        else:
            BC = hermitian(hermitian(rhs) @ lastInvDtN)
        BC = F_reduce @ BC

    nBC = BC
    for i in range(N_iter):
        sol = layers[-(i + 1)][-1]
        nBC = sol.topdown_solve_Dirichlet(nBC, transposed_solve=transposed_solve)

    if layers_pre is None:
        size = layers[-1][0].shape
        assert size[1] == 1 or size[2] == 1

    return nBC, layers, lastInvDtN


# ----------------------------------------------------------------------
# Outer Schwarz_Schur_involution entry point
# ----------------------------------------------------------------------

class Helpers():
    name = 'Helpers on how to use solution of Schwarz_Schur_involution'
    flatten = image_flatten
    unflatten = image_unflatten


# ----------------------------------------------------------------------
# Single-jit fast forward solver (no prefact_sol / no transposed_solve)
# ----------------------------------------------------------------------

def _bfs_indices(p, q):
    PB, PUB = alg.matrix_bdr_interior_split(wh=(p, q))
    return jnp.asarray(PB + PUB, dtype=jnp.int32)


@partial(jax.jit, static_argnames=("p", "q", "a", "b", "BC", "transposed_solve", "bfs_order"))
def _solve_pure(Alpha, Beta, p, q, a, b, BC, transposed_solve, bfs_order):
    """End-to-end forward solve, fully jit-compiled.

    Restricted entrypoint: only handles the (prefact_sol is None, BC ∈
    {'Neumann-scatter', 'Neumann-full', 'midpoint-reflective'}) path.
    Returns X_uv with shape (batch_size, W, H, c).
    """
    # ---- bfs reorder ----
    if not bfs_order:
        ind = _bfs_indices(p, q)
        Alpha = Alpha[..., ind, :][..., :, ind]
        Beta = Beta[..., ind, :]

    RHS = Beta
    irs = 2 * (p + q) - 4

    # ---- batch Dirichlet pre ----
    DtN, dLA_rr, dLA_rs, dLA_sr, dLA_ss, inv_dLA_ss, reused_prod = _dirichlet_pre_compute(Alpha, irs)
    # nRHS
    RHS_s = RHS[..., irs:, :]
    RHS_r = RHS[..., 0:irs, :]
    if not transposed_solve:
        nRHS = RHS_r - reused_prod @ RHS_s
    else:
        nRHS = RHS_r - hermitian((hermitian(RHS_s) @ inv_dLA_ss) @ dLA_sr)

    # ---- scatter boundary elimination (only when BC is Neumann-scatter) ----
    if BC == 'Neumann-scatter':
        scatter_out = _scatter_compute(DtN, nRHS, p, q, transposed_solve)
        sDtN, sRHS = scatter_out[0], scatter_out[1]
        scatter_pack = scatter_out[2:]   # the cached arrays for back-fill
    else:
        sDtN, sRHS = DtN, nRHS
        scatter_pack = None

    # ---- Schur recursion ----
    cur_DtN, cur_RHS = sDtN, sRHS
    cur_p, cur_q = p - 1, q - 1
    ps_initial, qs_initial = p - 1, q - 1   # used for top-level BC stride
    N_iter = math.floor(math.log2(a) + math.log2(b))
    layer_packs = []   # list of (dim, p, q, invSS, invSSmSR, RS, fS)

    for i in range(N_iter):
        dim = 0 if i % 2 == 0 else 1
        if dim == 0:
            mDtN, mRHS, invSS, invSSmSR, RS, fS = _collapse_compute_dim0(cur_DtN, cur_RHS, cur_p, cur_q, transposed_solve)
            layer_packs.append((dim, cur_p, cur_q, invSS, invSSmSR, RS, fS))
            cur_p, cur_q = 2 * cur_p, cur_q
        else:
            mDtN, mRHS, invSS, invSSmSR, RS, fS = _collapse_compute_dim1(cur_DtN, cur_RHS, cur_p, cur_q, transposed_solve)
            layer_packs.append((dim, cur_p, cur_q, invSS, invSSmSR, RS, fS))
            cur_p, cur_q = cur_p, 2 * cur_q
        cur_DtN, cur_RHS = mDtN, mRHS

    # ---- top BC solve ----
    if BC == 'Neumann-scatter':
        # The stride uses the *initial* ps == qs == p-1 (not the post-loop value).
        assert ps_initial == qs_initial
        ss = slice(0, None, ps_initial)
        lhs = cur_DtN
        rhs = cur_RHS
        BC_arr = jnp.zeros(lhs.shape[:-1] + rhs.shape[-1:], dtype=lhs.dtype)
        lastInvDtN = inv_default(lhs[..., ss, ss])
        if not transposed_solve:
            BC_arr = BC_arr.at[..., ss, :].set(lastInvDtN @ rhs[..., ss, :])
        else:
            BC_arr = BC_arr.at[..., ss, :].set(hermitian(hermitian(rhs[..., ss, :]) @ lastInvDtN))
    elif BC == 'Neumann-full':
        lastInvDtN = inv_default(cur_DtN)
        if not transposed_solve:
            BC_arr = lastInvDtN @ cur_RHS
        else:
            BC_arr = hermitian(hermitian(cur_RHS) @ lastInvDtN)
    else:
        raise ValueError(f"BC={BC!r} not supported by the fast path")

    # ---- Schur back-sub ----
    nBC = BC_arr
    for pack in reversed(layer_packs):
        dim, p_l, q_l, invSS, invSSmSR, RS, fS = pack
        if dim == 0:
            nBC = _collapse_back_dim0(nBC, fS, invSS, invSSmSR, RS, p_l, q_l, transposed_solve)
        else:
            nBC = _collapse_back_dim1(nBC, fS, invSS, invSSmSR, RS, p_l, q_l, transposed_solve)

    # ---- scatter back-fill ----
    if BC == 'Neumann-scatter':
        nBC = _scatter_back_fill(nBC, p, q, transposed_solve, *scatter_pack)

    # ---- final Dirichlet block solve (Y, Z, fillin) ----
    Y = nBC
    RHS_s = RHS[..., irs:, :]
    if not transposed_solve:
        Z = inv_dLA_ss @ (RHS_s - dLA_sr @ Y)
    else:
        Z = hermitian((hermitian(RHS_s) - hermitian(Y) @ dLA_rs) @ inv_dLA_ss)

    # solution_fillin equivalent (build X_uv from BC + Z)
    Z_uv = jnp.swapaxes(jnp.reshape(Z, list(Z.shape[:-2]) + [q - 2, p - 2, -1]), -2, -3)
    sz = list(Z_uv.shape)
    sz[-2] = sz[-2] + 2
    sz[-3] = sz[-3] + 2
    X_uv = jnp.zeros(sz, dtype=Z.dtype)
    X_uv = X_uv.at[..., 1:-1, 1:-1, :].set(Z_uv)
    X_uv = X_uv.at[..., 0:p - 1, 0, :].set(Y[..., 0:p - 1, :])
    X_uv = X_uv.at[..., -1, 0:q - 1, :].set(Y[..., p - 1:p + q - 2, :])
    X_uv = X_uv.at[..., 1:p, -1, :].set(jnp.flip(Y[..., p + q - 2:2 * p + q - 3, :], axis=-2))
    X_uv = X_uv.at[..., 0, 1:q, :].set(jnp.flip(Y[..., 2 * p + q - 3:, :], axis=-2))

    # bottom_up
    X_full = jnp.reshape(X_uv, (X_uv.shape[0], a, b, p, q, -1))
    cx = X_full
    n_iter = math.floor(math.log2(a) + math.log2(b))
    coeff = 0.5
    for i in range(n_iter):
        if i % 2 == 0:
            cx = jnp.concatenate([
                cx[:, 0::2, :, :-1, ...],
                coeff * (cx[:, 0::2, :, [-1], ...] + cx[:, 1::2, :, [0], ...]),
                cx[:, 1::2, :, 1:, ...],
            ], axis=3)
        else:
            cx = jnp.concatenate([
                cx[:, :, 0::2, :, :-1, ...],
                coeff * (cx[:, :, 0::2, :, [-1], ...] + cx[:, :, 1::2, :, [0], ...]),
                cx[:, :, 1::2, :, 1:, ...],
            ], axis=4)
    return cx[:, 0, 0, ...]   # X_uv shape: (batch, W, H, c)


def Schwarz_Schur_involution_fast(Alpha, Beta, BC, wh, transposed_solve=False,
                                  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    """Single-jit forward path. Returns a thin solution wrapper with
    .X_uv and .X — no per-layer state and no prefact reuse.
    Use this when you don't need autograd through the solver."""
    if BC is None or BC == 'Neumann':
        BC = 'Neumann-scatter'
    p, q = wh[0], wh[1]
    a, b = Alpha.shape[1], Alpha.shape[2]

    X_uv = _solve_pure(Alpha, Beta, p, q, a, b, BC, transposed_solve, bfs_order)

    class Solution:
        name = 'Schwarz_Schur_involution_fast'
    sol = Solution()
    sol.X_uv = X_uv
    sol.X = image_flatten(X_uv)
    sol.ab_wh = ((a, b), (p, q))
    return sol


def Schwarz_Schur_involution(
    Alpha,
    Beta,
    BC,
    wh,
    debug=False,
    prefact_sol=None,
    transposed_solve=False,
    bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER,
):
    # Fast path: when caller doesn't need a stateful sol object (no
    # prefact_sol reuse, no autograd plumbing), dispatch to the single-jit
    # Schwarz_Schur_involution_fast — it ~10x faster on the lemur problem.
    if (prefact_sol is None
        and BC in (None, 'Neumann', 'Neumann-scatter', 'Neumann-full')
        and Beta is not None):
        return Schwarz_Schur_involution_fast(
            Alpha, Beta, BC, wh,
            transposed_solve=transposed_solve, bfs_order=bfs_order,
        )

    if Beta is None:
        assert isinstance(BC, jnp.ndarray)
        Beta = zero_beta_for(Alpha, columns=BC.shape[-1])

    if prefact_sol is None:
        p = wh[0]
        q = wh[1]
        batch_size = Alpha.shape[0]
        a = Alpha.shape[1]
        b = Alpha.shape[2]
        assert len(Alpha.shape) == 5
        assert Alpha.shape[3] == (p * q)
        assert Alpha.shape[4] == (p * q)
        dtype = Alpha.dtype
    else:
        assert Alpha is None
        batch_size = prefact_sol.batch_size
        ab_wh = prefact_sol.ab_wh
        ab = ab_wh[0]
        a = ab[0]
        b = ab[1]
        wh = ab_wh[1]
        p = wh[0]
        q = wh[1]
        dtype = prefact_sol.dtype

    assert len(Beta.shape) == 5
    assert Beta.shape[0] == batch_size
    assert Beta.shape[1] == a
    assert Beta.shape[2] == b
    assert Beta.shape[3] == (p * q)

    if not bfs_order:
        if Alpha is not None:
            Alpha = convert_to_boundary_first_ordering(Alpha, wh=wh)
        Beta = convert_to_boundary_first_ordering_rhs(Beta, wh=wh)

    RHS = Beta

    if BC is None:
        print('warning: use Neumann for BC==None')
        BC = 'Neumann'
    if BC == 'Neumann':
        BC = 'Neumann-scatter'

    if prefact_sol is None:
        DtN, get_nRHS_step1, batch_Dirichlet_solver = batch_Dirichlet_solve_pre(
            dim_pq=(p, q), dLA=Alpha,
        )

        def get_nRHS(RHS, transposed_solve):
            return get_nRHS_step1(RHS, transposed_solve=transposed_solve)

        class Solution:
            name = 'Schwarz_Schur_involution'
    else:
        class Solution:
            name = 'Schwarz_Schur_involution_back_sub'
        DtN = prefact_sol.DtN
        get_nRHS = prefact_sol.get_nRHS
        batch_Dirichlet_solver = prefact_sol.batch_Dirichlet_solver

    sol = Solution()

    nRHS = get_nRHS(RHS, transposed_solve)
    if not (BC == 'Neumann-full' or BC == 'Neumann-scatter' or BC == 'midpoint-reflective'):
        assert nRHS.shape == (DtN.shape[:-1] + BC.shape[-1:])

    if BC == 'Neumann-scatter':
        if prefact_sol is None:
            sDtN, sRHS, nBC_back_fill, sol_scatter = scatter_boundary_eliminate_pre(
                (p, q), DtN, nRHS, transposed_solve=transposed_solve,
            )
        else:
            sDtN = prefact_sol.sDtN
            _, sRHS, nBC_back_fill, sol_scatter = scatter_boundary_eliminate_pre(
                (p, q), DtN, nRHS,
                pre_sol=prefact_sol.sol_scatter, transposed_solve=transposed_solve,
            )
    else:
        sDtN, sRHS = DtN, nRHS
        sol_scatter = None

    if prefact_sol is None:
        sBC, layers, lastInvDtN = apply_all_schur_steps(
            sDtN, sRHS, BC, a, b, p - 1, q - 1,
            debug=debug, transposed_solve=transposed_solve,
        )
    else:
        sBC, layers, lastInvDtN = apply_all_schur_steps(
            sDtN, sRHS, BC, a, b, p - 1, q - 1,
            debug=debug, transposed_solve=transposed_solve,
            layers_pre=prefact_sol.layers, lastInvDtN_reuse=prefact_sol.lastInvDtN,
        )

    if BC == 'Neumann-scatter':
        nBC = nBC_back_fill(sBC, transposed_solve=transposed_solve)
    else:
        nBC = sBC

    sol_blockwise = batch_Dirichlet_solver(
        BC=nBC, RHS=RHS, compute_energy=True, transposed_solve=transposed_solve,
    )

    X_uv = bottom_up(
        jnp.reshape(sol_blockwise.X_uv, (batch_size, a, b, p, q, -1)),
        value_sum=False,
    )
    sol.X_uv = X_uv[:, 0, 0, ...]

    sol.batch_Dirichlet_solver = batch_Dirichlet_solver
    sol.get_nRHS = get_nRHS
    sol.DtN = DtN
    sol.lastInvDtN = lastInvDtN
    sol.ab_wh = ((a, b), (p, q))
    sol.batch_size = batch_size
    sol.dtype = dtype
    sol.layers = layers
    sol.sol_scatter = sol_scatter
    sol.sDtN = sDtN

    sol.BC = BC
    sol.Beta = Beta
    sol.X = image_flatten(sol.X_uv)
    sol.sol_blockwise = sol_blockwise
    sol.helpers = Helpers()

    return sol


# ----------------------------------------------------------------------
# Custom VJP autograd wrapping Schwarz_Schur_involution
# ----------------------------------------------------------------------

@jax.custom_vjp
def sinv2d(Alpha, Beta, BC, ab_wh, options=''):
    sol = Schwarz_Schur_involution(
        Alpha, Beta, BC=BC, wh=ab_wh[1], transposed_solve=False,
    )
    return sol.X_uv


def _sinv2d_fwd(Alpha, Beta, BC, ab_wh, options):
    sol = Schwarz_Schur_involution(
        Alpha, Beta, BC=BC, wh=ab_wh[1], transposed_solve=False,
    )
    X_BWHC = sol.X_uv
    return X_BWHC, (Alpha, Beta, X_BWHC, BC, ab_wh, options)


def _sinv2d_bwd(res, grad):
    Alpha, Beta, X_BWHC, BC, ab_wh, options = res

    beta_grad = beta_from_BWHC(grad, ab_wh=ab_wh)
    sol_backward = Schwarz_Schur_involution(
        Alpha, beta_grad, BC=BC, wh=ab_wh[1], transposed_solve=True,
    )
    gradBeta = sol_backward.X_uv

    gradBeta = top_down(gradBeta[:, None, None, ...], value_divide=False, ab=ab_wh[0])
    gradBeta = image_flatten(gradBeta)

    X_Babwhc = top_down(X_BWHC[:, None, None, ...], value_divide=False, ab=ab_wh[0])
    X_Babnc = image_flatten(X_Babwhc)

    gradAlpha = -torch_batch_outer(gradBeta, X_Babnc)
    assert gradAlpha.shape == Alpha.shape

    return gradAlpha, gradBeta, None, None, None


sinv2d.defvjp(_sinv2d_fwd, _sinv2d_bwd)


# Backward-compat alias to mirror the original autograd.Function name
class Sinv2D_transposed_reuse:
    """Compat shim: in PyTorch this was an autograd.Function with .apply().
    In JAX we use jax.custom_vjp on the function `sinv2d` directly."""
    apply = staticmethod(sinv2d)


# ----------------------------------------------------------------------
# Sparse mat-vec, builders
# ----------------------------------------------------------------------

def Smul2D(Alpha, Chi, wh, Chi_like_output=True):
    Beta = Alpha @ Chi
    if Chi_like_output:
        return beta_sync_across_patches(Beta, wh=wh)
    return Beta


def uniform_lap_patch(p, q):
    import pyamg
    sten = np.array([[0., -1., -0.], [-1., 4., -1.], [-0., -1., 0.]])
    print(f'sten:{sten}')
    A = pyamg.gallery.stencil_grid(sten.transpose(), (q, p))
    d = A.shape[0]
    DA = A.toarray()
    np.fill_diagonal(DA, 0)
    diag = np.sum(DA, axis=1)
    np.fill_diagonal(DA, -diag)
    print(np.linalg.norm(DA @ np.ones(d)))
    return DA


def dense_patchwise_laplacian_rand_walk(images):
    assert False  # WIP


# ---- beta / chi shaping ----

def _beta_topdown(beta_abuv, ab_wh, value_divide=True):
    ab = ab_wh[0]
    a = ab[0]
    b = ab[1]
    div_level = (math.log2(a) + math.log2(b)) / 2
    assert 12 == 12.0
    assert div_level == int(div_level)
    div_level = int(div_level)
    RHS_bdr = top_down(beta_abuv, div_level=div_level, value_divide=value_divide)
    return image_flatten(RHS_bdr)


def _core_beta_or_chi_from_BWHC(BWHC, ab_wh, value_divide=True, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    RHS_abuv = BWHC[:, None, None, ...]
    beta = _beta_topdown(RHS_abuv, ab_wh=ab_wh, value_divide=value_divide)
    if bfs_order:
        beta = convert_to_boundary_first_ordering_rhs(beta, wh=ab_wh[1])
    return beta


def beta_from_BWHC(BWHC, ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    return _core_beta_or_chi_from_BWHC(BWHC, ab_wh=ab_wh, value_divide=True, bfs_order=bfs_order)


def chi_from_BWHC(BWHC, ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    return _core_beta_or_chi_from_BWHC(BWHC, ab_wh=ab_wh, value_divide=False, bfs_order=bfs_order)


def diag_alpha_from_beta(beta):
    assert beta.shape[-1] == 1
    n = beta.shape[-2]
    alpha = jnp.zeros(beta.shape[:-2] + (n, n), dtype=beta.dtype)
    idx = jnp.arange(n)
    alpha = alpha.at[..., idx, idx].set(beta[..., 0])
    return alpha


def beta_from_alpha_diag(alpha):
    assert alpha.shape[-1] == alpha.shape[-2]
    n = alpha.shape[-2]
    idx = jnp.arange(n)
    beta = jnp.zeros(alpha.shape[:-2] + (n, 1), dtype=alpha.dtype)
    beta = beta.at[..., 0].set(alpha[..., idx, idx])
    return beta


def alpha_from_phi(phi, ab_wh, in_place=False):
    # `in_place` is a no-op in JAX (arrays are immutable); always returns a new array.
    alpha = phi
    return patch_laplacian_adjust_boundary(alpha, ab_wh)


def beta_from_numpy(numpyRHS, ab_wh, dtype=None, value_divide=True, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    if dtype is None:
        dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32
    a, b, w, h, W, H = unpack_dims(ab_wh)
    assert len(numpyRHS.shape) == 2
    RHS_f = jnp.asarray(numpyRHS, dtype=dtype)[None, ...]
    BWHC = image_unflatten(RHS_f, W, H)
    assert value_divide is True
    return beta_from_BWHC(BWHC, ab_wh, bfs_order=bfs_order)


def meshgrid_coordinates(WH, dtype):
    grid_x, grid_y = jnp.meshgrid(
        jnp.arange(WH[0], dtype=dtype),
        jnp.arange(WH[1], dtype=dtype),
        indexing='ij',
    )
    return jnp.stack((grid_x, grid_y), axis=-1)


def mesh_hier(ab_wh, flattened=True):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    grid_x, grid_y = jnp.meshgrid(
        jnp.arange(W, dtype=jnp.int64),
        jnp.arange(H, dtype=jnp.int64),
        indexing='ij',
    )
    FX = grid_x[None, None, None, ..., None]
    FY = grid_y[None, None, None, ..., None]
    FXY = jnp.concatenate([FX, FY], axis=-1)
    FXY = top_down(FXY, ab=(a, b), value_divide=False)
    FXY = FXY[0]
    if flattened:
        FXY = jnp.reshape(jnp.swapaxes(FXY, -2, -3),
                          FXY.shape[:-3] + (-1,) + FXY.shape[-1:])
    return FXY


def patch_adjacency_mask(wh, nebr):
    XY = mesh_hier(((1, 1), wh))[0, 0] * 1.0
    # cdist via broadcasting
    diff = XY[:, None, :] - XY[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    if nebr == '5pt':
        return d2 <= 1.01
    assert nebr == '9pt'
    return d2 <= 2.01


def _torch_bottom_up_sparse_indices(ab_wh):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    FXY = mesh_hier(ab_wh=ab_wh, flattened=True)
    FX = FXY[..., 0:1]
    FY = FXY[..., 1:2]
    FD = FX + FY * W
    IX, IY = torch_batch_cartesian_prod(FD, FD)
    return IX, IY


def _torch_bottom_up_sparse_VIJ(DA, wh, numpy_out=False, adj='all'):
    w = wh[0]
    h = wh[1]
    a = DA.shape[-4]
    b = DA.shape[-3]
    IV = DA
    IX, IY = _torch_bottom_up_sparse_indices(ab_wh=((a, b), (w, h)))

    if adj == '9pt' or adj == '5pt':
        adj_mask = patch_adjacency_mask(wh=(w, h), nebr='9pt')
        ii, jj = jnp.where(adj_mask)
        ii2, jj2 = jnp.where(jnp.logical_not(adj_mask))
        removed = IV[..., ii2, jj2]
        assert 0.0 == jnp.abs(removed).sum()
        IX = IX[..., ii, jj]
        IY = IY[..., ii, jj]
        IV = IV[..., ii, jj]
    else:
        assert adj == 'all'

    if numpy_out:
        return np.asarray(IV), np.asarray(IX), np.asarray(IY)
    return IV, IX, IY


def alpha_to_scipy(Alpha, ab_wh, shape=None, boundary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER, adj='all'):
    assert Alpha.shape[0] == 1
    DA = Alpha[0]
    import scipy.sparse
    a, b, w, h, W, H = unpack_dims(ab_wh)
    DA_abxx = jnp.reshape(DA, (a, b) + DA.shape[-2:])
    if boundary_first_order:
        DA_abxx = convert_to_lexicographic_sweeping_order(DA_abxx, [w, h])
    IV, IX, IY = _torch_bottom_up_sparse_VIJ(DA_abxx, wh=(w, h), numpy_out=True, adj=adj)
    N = W * H
    A_scipy = scipy.sparse.csr_matrix((IV.flatten(), (IX.flatten(), IY.flatten())), shape=(N, N))
    return [A_scipy]


def torch_bottom_up_sparse_scipy(DA, ab_wh, shape=None, boundary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER, adj='all'):
    return alpha_to_scipy(
        Alpha=DA[None, ...], ab_wh=ab_wh, shape=shape,
        boundary_first_order=boundary_first_order, adj=adj,
    )[0]


def flattenWH(X):
    return jnp.reshape(jnp.swapaxes(X, -2, -3),
                       X.shape[:-3] + (-1,) + X.shape[-1:])


def patch_laplacian_adjust_boundary(alpha, ab_wh):
    """Modify alpha (functionally — returns a new array) for patches at the
    whole-image border."""
    a, b, w, h, W, H = unpack_dims(ab_wh)

    # Row-major flat indices of the four edges of a w×h patch.
    bottom = jnp.arange(w, dtype=jnp.int64)
    top    = jnp.arange((h - 1) * w, h * w, dtype=jnp.int64)
    left   = jnp.arange(h, dtype=jnp.int64) * w
    right  = jnp.arange(h, dtype=jnp.int64) * w + (w - 1)

    rr, cc = jnp.meshgrid(left, left, indexing='ij')
    alpha = alpha.at[..., 1:, :, rr, cc].multiply(0.5)
    rr, cc = jnp.meshgrid(right, right, indexing='ij')
    alpha = alpha.at[..., :-1, :, rr, cc].multiply(0.5)
    rr, cc = jnp.meshgrid(bottom, bottom, indexing='ij')
    alpha = alpha.at[..., :, 1:, rr, cc].multiply(0.5)
    rr, cc = jnp.meshgrid(top, top, indexing='ij')
    alpha = alpha.at[..., :, :-1, rr, cc].multiply(0.5)
    return alpha


def batch_dense_laplacian_whwh(X):
    assert False  # WIP


def single_patch_uniform_laplacian(wh):
    from . import torch_mesh_processing as tmgp
    w, h = wh[0], wh[1]
    batch_size = 1
    n = w * h
    f = (w - 1) * (h - 1) * 4

    V, F, _ = tmgp.grid_to_double_covered_mesh(w, h)
    II, JJ, _, _ = tmgp.lap_entries(V, F, multicol=False)

    au = jnp.concatenate([
        jnp.ones((batch_size, f, 1)),
        jnp.zeros((batch_size, f, 1)),
        jnp.ones((batch_size, f, 1)),
    ], axis=-1)

    gn_p = tmgp.grad_normalized(V, F)
    print(f'gn_p: {gn_p.shape, jnp.stack([II, JJ]).shape, F.shape}')
    print('au', au.shape)

    values = tmgp.assemble_lap_values(au, gn_p)

    # Densify the COO entries: dLA[i,j] = sum over k where II[k]==i and JJ[k]==j
    dLA = jnp.zeros((batch_size, n, n), dtype=values.dtype)
    dLA = dLA.at[0, II, JJ].add(values)
    return dLA[0]


def batch_diag(XX):
    assert XX.shape[-1] == 1
    n = XX.shape[-2]
    RR = jnp.zeros(list(XX.shape[:-2]) + [n, n], dtype=XX.dtype)
    idx = jnp.arange(n)
    RR = RR.at[..., idx, idx].set(XX[..., 0])
    return RR


def alpha_from_patchwise_features(CX):
    _, a, b, w, h, c = list(CX.shape)
    assert c == 1
    import mesh_processing as mgp
    mesh = mgp.MeshGrid(w, h)
    Gx = jnp.asarray(mesh.Gx.todense())[None, None, None]
    Gy = jnp.asarray(mesh.Gy.todense())[None, None, None]
    F = jnp.asarray(mesh.F, dtype=jnp.int64)

    C = flattenWH(CX)
    CF = (C[..., F[:, 0], :] + C[..., F[:, 1], :] + C[..., F[:, 2], :]) / 6.0
    CF = batch_diag(CF)

    LA = jnp.swapaxes(Gx, -1, -2) @ CF @ Gx + jnp.swapaxes(Gy, -1, -2) @ CF @ Gy
    return LA


def single_patch_mass_diagonal(wh):
    w, h = wh[0], wh[1]
    M = jnp.eye(w * h)
    m = 2 * jnp.ones((w, h))
    m = m.at[[0, -1], :].multiply(0.5)
    m = m.at[:, [0, -1]].multiply(0.5)
    ind = jnp.arange(w * h)
    M = M.at[ind, ind].set(jnp.swapaxes(m, -1, -2).flatten())
    return M


def alpha_uniform_laplacian(ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    alpha_shape = (1, a, b, w * h, w * h)
    alpha = jnp.zeros(alpha_shape)
    alpha = alpha.at[..., :, :].set(single_patch_uniform_laplacian([w, h]))
    if bfs_order:
        alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
    return alpha


def alpha_uniform_mass_diagonal(ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    alpha_shape = (1, a, b, w * h, w * h)
    alpha = jnp.zeros(alpha_shape)
    alpha = alpha.at[..., :, :].set(single_patch_mass_diagonal([w, h]))
    if bfs_order:
        alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
    return alpha


def alpha_coeff_laplacian(image_hwc, ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    image_whc = jnp.swapaxes(image_hwc[None, None, None], 3, 4)
    a, b, w, h, W, H = unpack_dims(ab_wh)
    CX = top_down(image_whc, div_level='Auto', value_divide=False, ab=ab_wh[0])
    alpha = alpha_from_patchwise_features(CX)
    alpha_shape = [1, a, b, w * h, w * h]
    assert list(alpha.shape) == alpha_shape
    adj_mask = patch_adjacency_mask([w, h], nebr='9pt').astype(alpha.dtype)
    alpha = alpha * adj_mask
    if bfs_order:
        alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
    return alpha


def batch_dense_affinity_image(X, wh, params=None):
    w, h = wh[0], wh[1]
    shape = X.shape
    assert shape[-2] == (w * h)

    # cdist equivalent
    diff = X[..., :, None, :] - X[..., None, :, :]
    D = jnp.sqrt(jnp.sum(diff * diff, axis=-1))

    sigma = 0.0
    if hasattr(params, 'sigma'):
        sigma = params.sigma
    sigma = jnp.asarray(sigma)

    if hasattr(params, 'temp_kernel'):
        Weights = jnp.maximum(jnp.exp(-1.0 * D * D), jnp.asarray(0.01))
    else:
        Weights = jnp.maximum(jnp.exp(-900.0 * D * D), sigma)
    return Weights


def alpha_image_laplacian(
    image, ab_wh,
    diag_zero_sum=True,
    adjust_border_patches=False,
    bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER,
    params=None,
):
    if image.ndim == 3:
        image = image[None, ...]
    else:
        assert image.ndim == 4

    image_whc = jnp.swapaxes(image[:, None, None, ...], 3, 4)
    print(image_whc.shape)
    a, b, w, h, W, H = unpack_dims(ab_wh)

    CX = top_down(image_whc, div_level='Auto', value_divide=False, ab=ab_wh[0])
    FCX = flattenWH(CX)

    alpha = batch_dense_affinity_image(FCX, ab_wh[1], params=params)
    alpha_shape = [1, a, b, w * h, w * h]
    assert list(alpha.shape) == alpha_shape

    adj_mask = patch_adjacency_mask([w, h], nebr='5pt' if hasattr(params, '5pt') else '9pt').astype(alpha.dtype)

    if hasattr(params, 'random_sparse_patch'):
        alpha = jax.random.uniform(jax.random.PRNGKey(0), alpha_shape) - 0.5

    alpha = alpha * adj_mask

    if hasattr(params, 'random_dense_patch'):
        alpha = jax.random.uniform(jax.random.PRNGKey(1), alpha_shape) - 0.5

    if hasattr(params, 'eye'):
        alpha = jnp.zeros(alpha_shape)
        ind = jnp.arange(w * h)
        alpha = alpha.at[..., ind, ind].set(1)
        diag_zero_sum = False
        assert adjust_border_patches is True

    if adjust_border_patches:
        alpha = patch_laplacian_adjust_boundary(alpha, ab_wh)

    if diag_zero_sum:
        alpha = -alpha
        ind = jnp.arange(w * h)
        alpha = alpha.at[..., ind, ind].set(0)
        alpha = alpha.at[..., ind, ind].set(-alpha.sum(axis=-1))

    if bfs_order:
        alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
    return alpha


def alpha_image_matting(image_hwc, ab_wh):
    lam = 0.0001 * 2
    alpha = alpha_image_laplacian(image_hwc, ab_wh, diag_zero_sum=True, adjust_border_patches=True)
    alpha = alpha + alpha_uniform_laplacian(ab_wh) * lam / 2
    alpha = alpha + alpha_uniform_mass_diagonal(ab_wh) * 0.000002 / 2
    return alpha


def alpha_image_matting_normalized(image_hwc, ab_wh):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    lam = 0.0001 * 2
    alpha = alpha_image_laplacian(image_hwc, ab_wh, diag_zero_sum=True, adjust_border_patches=True)
    alpha = alpha + alpha_uniform_laplacian(ab_wh) * lam / 2
    alpha = alpha + alpha_uniform_mass_diagonal(ab_wh) * 0.000002 / 2
    return alpha


def scipy_sparse_top_down(
    A, ab_wh=None,
    init_shape='not used given ab_wh',
    div_level='not used given ab_wh',
    value_divide=True,
    input_x_first=True,
    permute_bounary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER,
    wh_if_permute_bounary_first_order=None,
    beta_shaped=True,
):
    if ab_wh is not None:
        a, b, w, h, W, H = unpack_dims(ab_wh)
        div_level = int((math.log2(a) + math.log2(b)) / 2)
        init_shape = (W, H)

    if wh_if_permute_bounary_first_order is not None:
        permute_bounary_first_order = True
        w, h = wh_if_permute_bounary_first_order

    assert A.shape[0] == (init_shape[0] * init_shape[1])
    p = init_shape[0]
    q = init_shape[1]
    Cs = [[A]]

    for level in range(div_level):
        m = len(Cs)
        n = len(Cs[0])
        Ns = [[None for j in range(n * 2)] for i in range(m)]
        assert q % 2 == 1
        cp = p
        cq = (q - 1) // 2
        assert Cs[0][0].shape[0] == p * q
        for i in range(m):
            for j in range(n):
                indices = alg.indices_matrix_2d(p, q, 1, 1, 0, 0, x_first=True)
                shared = indices[:, cq].flatten()
                part1 = indices[:, 0:cq + 1].transpose().flatten()
                part2 = indices[:, cq:None].transpose().flatten()
                S = Cs[i][j].copy()
                if value_divide:
                    S[np.ix_(shared, shared)] /= 2
                Ns[i][j * 2] = S[:, part1][part1, :]
                Ns[i][j * 2 + 1] = S[:, part2][part2, :]
        Cs = Ns
        p = cp
        q = cq + 1

        m = len(Cs)
        n = len(Cs[0])
        Ns = [[None for j in range(n)] for i in range(m * 2)]
        assert p % 2 == 1
        cp = (p - 1) // 2
        cq = q
        for i in range(m):
            for j in range(n):
                indices = alg.indices_matrix_2d(p, q, 1, 1, 0, 0, x_first=True)
                shared = indices[cp, :].flatten()
                part1 = indices[0:cp + 1, :].transpose().flatten()
                part2 = indices[cp:None, :].transpose().flatten()
                S = Cs[i][j].copy()
                if value_divide:
                    S[np.ix_(shared, shared)] /= 2
                Ns[i * 2][j] = S[:, part1][part1, :]
                Ns[i * 2 + 1][j] = S[:, part2][part2, :]
        Cs = Ns
        p = cp + 1
        q = cq

    dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32
    DA = jnp.stack([
        jnp.stack([
            jnp.asarray(Cs[i][j].todense(), dtype=dtype) for j in range(len(Cs[i]))
        ], axis=0) for i in range(len(Cs))
    ])

    if permute_bounary_first_order:
        DA = convert_to_boundary_first_ordering(DA, wh=[w, h])

    if not beta_shaped:
        DA = DA.reshape((-1,) + DA.shape[-2:])
    return Cs, DA


def alpha_from_scipy_sparse(A_scipy, ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
    _, alpha = scipy_sparse_top_down(
        A_scipy, ab_wh=ab_wh, beta_shaped=True,
        permute_bounary_first_order=bfs_order,
    )
    return alpha[None, ...]


# ---- empty alpha/beta/chi builders ----

def simple_WHC(ab_wh, dtype=None, fun=jnp.ones):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    if dtype is None:
        dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32
    if isinstance(fun, str):
        assert fun == 'xy_coordinates'
        return meshgrid_coordinates(WH=[W, H], dtype=dtype)
    return fun([W, H, 1], dtype=dtype)


def BWHC_of_shape(ab_wh, batch_size=1, columns=1, fun=jnp.zeros, dtype=None):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    if dtype is None:
        dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32
    return fun([batch_size, W, H, columns], dtype=dtype)


def beta_of_shape(ab_wh, batch_size=1, columns=1, fun=jnp.zeros, dtype=None):
    BWHC = BWHC_of_shape(ab_wh=ab_wh, batch_size=batch_size, columns=columns, fun=fun, dtype=dtype)
    print('beta_of_shape:', BWHC.shape, ab_wh)
    return beta_from_BWHC(BWHC, ab_wh)


def chi_of_shape(ab_wh, batch_size=1, columns=1, fun=jnp.zeros, dtype=None):
    a, b, w, h, W, H = unpack_dims(ab_wh)
    if dtype is None:
        dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32
    return fun([batch_size, a, b, w * h, columns], dtype=dtype)


def zero_beta_for(Alpha, columns=1):
    return jnp.zeros(Alpha.shape[:-1] + (columns,), dtype=Alpha.dtype)


def default_chi_for(Alpha, columns=1, fun=jnp.zeros):
    return fun(Alpha.shape[:-1] + (columns,), dtype=Alpha.dtype)
