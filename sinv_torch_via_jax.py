"""PyTorch-facing wrapper around ``sinv_jax.Schwarz_Schur_involution_fast``.

For the demo problem (((128,128),(5,5)), fp32 RGB) the JAX single-jit forward
runs at ~24 ms, vs. ~46 ms for ``sinv_cuda`` and ~83 ms for the pure-torch
reference. If you don't need autograd, the cheapest way to use it from
PyTorch is to hand the tensors to JAX zero-copy via DLPack, run the solver,
and DLPack the result back.

This module exposes:
  - ``SinvViaJax``: a ``torch.autograd.Function`` whose ``forward`` does the
    DLPack round-trip described above. ``backward`` is left as a stub for
    you to fill in.
  - ``sinv2d_via_jax``: convenience wrapper, same surface as
    ``sinv_cuda.sinv2d``.

Notes
-----
1. **Preallocation.** Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` *before*
   the first JAX import. We do that here at module load. If something else
   in your process imported jax first with the default settings, JAX has
   already grabbed ~75% of VRAM and torch's caching allocator will starve.
   In that case re-launch with the env var set externally.

2. **VMM allocator (large GPUs).** On GPUs with ≥48 GB VRAM (A6000, A100,
   H100, …) JAX's BFC allocator uses CUDA Virtual Memory Management (VMM),
   which places allocations in a special address space.
   ``cublasDgetrfBatched`` — used internally for batched LU factorization —
   does **not** support VMM addresses and raises
   ``CUDA_ERROR_INVALID_ADDRESS_SPACE``.  Setting
   ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` forces JAX to use plain
   ``cudaMalloc``/``cudaFree``, which always returns standard global-memory
   addresses and works around the crash.

   We deliberately do **not** set this by default.  The platform allocator
   disables JAX's BFC memory *pool*, so every intermediate buffer in the
   deeply-unrolled solve pays a fresh ``cudaMalloc``/``cudaFree`` — a large
   throughput regression (observed ~10× slower in multi-GPU training) on the
   vast majority of GPUs that never hit the VMM threshold.  Default behavior
   therefore keeps the fast BFC pool.  If you actually run on a ≥48 GB GPU and
   hit ``CUDA_ERROR_INVALID_ADDRESS_SPACE``, opt in explicitly *before*
   launching Python:

       export XLA_PYTHON_CLIENT_ALLOCATOR=platform

   (``setdefault`` below honors that external value.)

3. **Float64.** Requires ``jax_enable_x64=True`` set before any jax op. We
   set it lazily on the first fp64 call; that may be too late if other
   code already touched jax. Set it explicitly at process start if you
   plan to mix.

4. **Streams.** DLPack does *not* transfer CUDA stream semantics. We call
   ``torch.cuda.synchronize()`` on the way in (so torch's writes are
   visible) and ``block_until_ready()`` on the way out (so JAX's writes
   are visible to torch). This is conservative; tighter pipelines can
   replace these with stream-wait events.

5. **Shape.** Inputs match the rest of the repo:
       Alpha : (B, a, b, w*h, w*h)
       Beta  : (B, a, b, w*h, c)
   Output is X_uv with shape (B, W, H, c) where W = a*(w-1)+1,
   H = b*(h-1)+1.
"""
import os

# Must precede the first jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# NOTE: we intentionally do NOT force XLA_PYTHON_CLIENT_ALLOCATOR=platform.
# That setting disables JAX's BFC memory pool and makes every intermediate
# buffer pay a cudaMalloc/cudaFree round-trip, which was a ~10x throughput
# regression in multi-GPU training.  The fast default BFC pool is kept here.
# Large-VRAM (>=48 GB) GPUs that hit the VMM/cublasDgetrfBatched crash
# (CUDA_ERROR_INVALID_ADDRESS_SPACE) can opt in by exporting
# XLA_PYTHON_CLIENT_ALLOCATOR=platform before launching Python (see header).

import torch
from torch.utils import dlpack as tdlp


_FP64_ENABLED = False


def _ensure_jax(dtype):
    """Lazy-import jax (so torch-only callers don't pay the JAX init cost
    if they never invoke this module). Configures the persistent compilation
    cache on first use, and enables x64 on first fp64 call."""
    global _FP64_ENABLED
    import jax  # noqa: F401
    # Persistent XLA cache is configured in sinv_jax (repo-root .jax_cache),
    # shared by both entry points. Importing the module runs it at load, but
    # call it here too so this path never depends on import order.
    import sinv_jax.torch_sparse_involution as sj
    sj.configure_compilation_cache()
    if dtype is torch.float64 and not _FP64_ENABLED:
        jax.config.update("jax_enable_x64", True)
        _FP64_ENABLED = True


def _torch_to_jax(t):
    """Zero-copy torch -> jax via DLPack. Tensor must be CUDA, contiguous.

    We explicitly set the current CUDA stream to the default stream for this
    tensor's device before calling __dlpack__. JAX's from_dlpack passes the
    stream back to PyTorch's __dlpack__(stream=...) to insert a CUDA event
    that gates JAX's read. If the calling thread's "current stream" happens
    to be a CPU stream (a DataLoader worker thread that hasn't touched CUDA
    yet, or a thread that reset the device), __dlpack__ raises:
        RuntimeError: Event device type CUDA does not match blocking
        stream's device type CPU
    Pinning to the device's default stream avoids this regardless of which
    thread we're on.

    Returns (jax_array, contiguous_torch_tensor). The caller MUST keep the
    second element alive until JAX has finished using the array (i.e. until
    block_until_ready() returns), otherwise PyTorch's caching allocator may
    reclaim the buffer while JAX's async kernel is still reading it.
    """
    from jax import dlpack as jdlp
    t_cont = t.detach().contiguous()
    with torch.cuda.device(t_cont.device):
        return jdlp.from_dlpack(t_cont), t_cont


def _jax_to_torch(j, device=None):
    """Copy jax -> torch via DLPack + clone.

    We do NOT use zero-copy here. ``torch.from_dlpack(j)`` produces a PyTorch
    tensor that shares JAX's buffer via a DLPack capsule.  With the platform
    allocator (``XLA_PYTHON_CLIENT_ALLOCATOR=platform``) JAX calls
    ``cudaFree`` immediately when no JAX-side reference remains.  If the
    Python GC drops the last JAX reference before PyTorch is done reading the
    tensor (e.g. between ``forward`` returning and ``backward`` unpacking
    ``ctx.saved_tensors``), CUDA marks that address as invalid.  The error is
    sticky and only surfaces at the *next* kernel launch — which appears to be
    an unrelated JAX kernel, masking the real cause.

    Cloning immediately copies the data into PyTorch-managed memory so JAX
    can free its buffer at any time without affecting the torch tensor.
    If ``device`` is given the clone lands on that device (guards against
    DLPack returning a CPU tensor when the JAX default device is unexpected).
    """
    t = torch.from_dlpack(j).clone()
    if device is not None:
        t = t.to(device)
    return t


def _solve_jax(Alpha, Beta, ab_wh, transposed_solve=False):
    """Pure-tensor forward: torch in, torch out. No autograd."""
    _ensure_jax(Alpha.dtype)
    import sinv_jax.torch_sparse_involution as sj

    if Alpha.device.type != "cuda":
        raise RuntimeError(
            "sinv_torch_via_jax requires CUDA tensors; got Alpha on "
            f"{Alpha.device}"
        )
    if Beta.device != Alpha.device:
        raise RuntimeError(
            "Alpha and Beta must be on the same CUDA device "
            f"(got {Alpha.device} vs {Beta.device})"
        )
    if Alpha.dtype != Beta.dtype:
        raise RuntimeError(
            f"Alpha/Beta dtype mismatch: {Alpha.dtype} vs {Beta.dtype}"
        )

    # Make sure pending torch writes are visible before JAX reads.
    # Pass the device index explicitly: torch.cuda.synchronize() with no
    # argument uses the *current thread's* current device, which may be
    # unset (CPU) on DataLoader worker threads and would raise the same
    # "Event device type CUDA does not match blocking stream's device type
    # CPU" error we're guarding against.
    torch.cuda.synchronize(Alpha.device)

    alpha_j, _alpha_cont = _torch_to_jax(Alpha)
    beta_j,  _beta_cont  = _torch_to_jax(Beta)

    sol = sj.Schwarz_Schur_involution_fast(
        alpha_j, beta_j,
        BC="Neumann", wh=ab_wh[1],
        transposed_solve=transposed_solve,
    )
    # Force JAX to finish before we hand the buffer to torch.
    # _alpha_cont / _beta_cont are kept alive here so PyTorch's caching
    # allocator cannot reclaim their storage while JAX kernels are running.
    sol.X_uv.block_until_ready()
    del _alpha_cont, _beta_cont

    return _jax_to_torch(sol.X_uv, device=Alpha.device)


class SinvViaJax(torch.autograd.Function):
    """Torch autograd wrapper. ``forward`` is the JAX single-jit fast path
    via DLPack. ``backward`` mirrors ``sinv_jax._sinv2d_bwd``: a transposed
    re-solve plus an outer-product to assemble ``gradAlpha``.

    Inputs:
      Alpha : (B, a, b, w*h, w*h)  CUDA, fp32 or fp64
      Beta  : (B, a, b, w*h, c)
      ab_wh : ((a, b), (w, h))     non-tensor, passed through ctx
    Output:
      X_uv  : (B, W, H, c)  with W = a*(w-1)+1, H = b*(h-1)+1
    """

    @staticmethod
    def forward(ctx, Alpha, Beta, ab_wh):
        ctx.ab_wh = ab_wh
        X_uv = _solve_jax(Alpha, Beta, ab_wh, transposed_solve=False)
        # Save inputs for the user's backward.
        ctx.save_for_backward(Alpha, X_uv)
        return X_uv

    @staticmethod
    def backward(ctx, grad_X_uv):
        """Mirrors ``sinv_jax._sinv2d_bwd``: re-solve with the transposed
        flag, then assemble gradAlpha = -outer(gradBeta, X) over the patch
        grid. Reuses helpers from the torch reference.

        For x = A \\ b:
            gradBeta  = A^{-T} @ grad_X        (re-solve with transposed_solve=True)
            gradAlpha = -outer(gradBeta, x)    (per-patch outer product)

        The transposed solve goes through the fast fused JAX path
        (``_solve_jax`` with ``transposed_solve=True``) — the same single-jit
        kernel as forward.  Backward runs on every training step, so the slow
        PyTorch reference solver is never used here: it would dominate step
        time and was the main multi-GPU training regression.

        (Historically a PyTorch fallback was used to dodge a
        ``CUDA_ERROR_ILLEGAL_ADDRESS`` that only appears with
        ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` on large-VRAM GPUs.  That
        allocator is no longer forced — see the module header — so the fast
        JAX path is safe in the default BFC-allocator configuration.)
        """
        from sinv.torch_sparse_involution import (
            beta_from_BWHC, top_down, image_flatten, torch_batch_outer,
        )

        Alpha, X_uv = ctx.saved_tensors
        ab_wh = ctx.ab_wh
        dev = Alpha.device

        # grad_X_uv: (B, W, H, c) -> beta-shaped (B, a, b, w*h, c).
        beta_grad = beta_from_BWHC(grad_X_uv.contiguous(), ab_wh=ab_wh)

        # Transposed solve: gradBeta_uv = A^{-T} @ beta_grad, in (B, W, H, c).
        # Fast fused JAX path — never the slow PyTorch reference solver.
        gradBeta_uv = _solve_jax(Alpha, beta_grad, ab_wh, transposed_solve=True)

        # (B, W, H, c) -> (B, a, b, w*h, c).
        gradBeta = top_down(
            gradBeta_uv[:, None, None, ...], value_divide=False, ab=ab_wh[0])
        gradBeta = image_flatten(gradBeta)

        # Same reshape on the saved forward solution.
        # X_uv comes from _jax_to_torch().clone() in forward; force to the
        # same device as gradBeta in case JAX produced a non-CUDA clone.
        X_Babwhc = top_down(
            X_uv.to(dev)[:, None, None, ...], value_divide=False, ab=ab_wh[0])
        X_Babnc = image_flatten(X_Babwhc)

        gradAlpha = -torch_batch_outer(gradBeta, X_Babnc)
        assert gradAlpha.shape == Alpha.shape, \
            f"gradAlpha shape {gradAlpha.shape} != Alpha shape {Alpha.shape}"

        # ab_wh is a non-tensor input, gets None.
        return gradAlpha, gradBeta, None


def sinv2d_via_jax(Alpha, Beta, ab_wh):
    """Convenience wrapper matching ``sinv_cuda.sinv2d``'s surface.

    Args:
      Alpha:  (B, a, b, w*h, w*h)  CUDA tensor.
      Beta:   (B, a, b, w*h, c)
      ab_wh:  ((a, b), (w, h))

    Returns:
      X_uv:   (B, W, H, c) where W = a*(w-1)+1, H = b*(h-1)+1.
    """
    return SinvViaJax.apply(Alpha, Beta, ab_wh)


def precompile_solver(ab_wh, batch_size=1, num_rhs=3, dtype=torch.float32,
                      device="cuda"):
    """Warm up (compile) the JAX kernel for a given problem shape.

    Runs one throwaway ``sinv2d_via_jax`` solve on dummy inputs so the
    ``@jax.jit`` program is compiled and cached ahead of time. Later calls at
    the *same* ``ab_wh`` / ``batch_size`` / ``num_rhs`` / ``dtype`` hit the
    compiled cache and skip the (multi-second) compile.

    Args:
      ab_wh:      ((a, b), (w, h)).
      batch_size: batch dimension B of the real inputs.
      num_rhs:    number of RHS columns (channels) c.
      dtype:      torch.float32 or torch.float64 — a separate compiled program.
      device:     CUDA device for the dummy tensors (JAX path is CUDA-only).
    """
    from sinv import torch_sparse_involution as tsi

    # alpha_uniform_mass_diagonal returns (1, a, b, w*h, w*h); tile to B.
    alpha_dummy = tsi.alpha_uniform_mass_diagonal(ab_wh).to(
        device=device, dtype=dtype)
    alpha_dummy = alpha_dummy.expand(
        batch_size, *alpha_dummy.shape[1:]).contiguous()

    beta_dummy = torch.zeros(
        alpha_dummy.shape[:-1] + torch.Size([num_rhs]),
        dtype=dtype, device=device)

    _ = sinv2d_via_jax(alpha_dummy, beta_dummy, ab_wh)
    torch.cuda.synchronize(alpha_dummy.device)
