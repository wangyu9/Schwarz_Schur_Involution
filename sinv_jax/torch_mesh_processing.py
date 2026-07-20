"""
JAX port of sinv/torch_mesh_processing.py — mesh utilities for the
per-patch Laplacian. Function names match the original; tensor ops use
jnp instead of torch.

Note: The module name is kept as `torch_mesh_processing` so the demo /
import surface mirrors the original. Internally everything is JAX.
"""

import jax
import jax.numpy as jnp


def cross2d(a, b):
    assert a.shape[-1] == 2
    assert b.shape == a.shape
    return a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]


def doublearea(V, F):
    assert F.shape[-1] == 3
    assert len(F.shape) == 2
    assert V.shape[-1] == 2
    r = V[F[:, 0], :] - V[F[:, 2], :]
    s = V[F[:, 1], :] - V[F[:, 2], :]
    dblA = cross2d(r, s)
    return dblA[..., 0]


def inner_prod(x, y):
    return jnp.sum(x * y, axis=-1, keepdims=True)


def grad_core_2d(V0, V1, V2):
    v20 = V2 - V0
    v10 = V1 - V0
    dot12 = inner_prod(v20, v10)
    c21 = cross2d(v20, v10)
    ns12 = c21 * c21
    r = (v10 * inner_prod(v20, v20) + v20 * inner_prod(v10, v10)) / ns12 \
        - (v10 + v20) * dot12 / ns12
    return r


def grad_area_vectorized(V, F):
    V0 = V[F[:, 0], :]
    V1 = V[F[:, 1], :]
    V2 = V[F[:, 2], :]
    g0 = grad_core_2d(V0, V1, V2)
    g1 = grad_core_2d(V1, V2, V0)
    g2 = grad_core_2d(V2, V0, V1)
    return jnp.stack([g0, g1, g2], axis=0), doublearea(V, F) / 2


def grad_normalized(V, F):
    g_all, area = grad_area_vectorized(V, F)
    area = area[None, :, None]
    assert area.min() > 0
    return g_all * jnp.sqrt(area)


def lap_entries(V, F, multicol=True):
    g = grad_normalized(V, F)
    v01 = inner_prod(g[0], g[1])[:, 0]
    v12 = inner_prod(g[1], g[2])[:, 0]
    v20 = inner_prod(g[2], g[0])[:, 0]
    v00 = -(v01 + v20)
    v11 = -(v12 + v01)
    v22 = -(v20 + v12)

    II = [F[:, 0], F[:, 1], F[:, 2], F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
    JJ = [F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    VV = [v00, v11, v22, v01, v12, v20, v01, v12, v20]

    if multicol:
        II = jnp.stack(II)
        JJ = jnp.stack(JJ)
        VV = jnp.stack(VV)
    else:
        II = jnp.concatenate(II)
        JJ = jnp.concatenate(JJ)
        VV = jnp.concatenate(VV)

    n = V.shape[0]
    return II, JJ, VV, n


def grid_to_double_covered_mesh(W, H):
    """Triangles are CCW (right-hand rule, outward-pointing)."""
    x = jnp.linspace(0, 1.0 * (W - 1), W)
    y = jnp.linspace(0, 1.0 * (H - 1), H)
    xv, yv = jnp.meshgrid(x, y, indexing='xy')

    V = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

    I = jnp.arange(W - 1, dtype=jnp.int64)
    J = jnp.arange(H - 1, dtype=jnp.int64)

    ind00 = I[None, :] + J[:, None] * W
    ind10 = (I + 1)[None, :] + J[:, None] * W
    ind01 = I[None, :] + (J + 1)[:, None] * W
    ind11 = (I + 1)[None, :] + (J + 1)[:, None] * W

    ind00 = ind00.flatten()
    ind10 = ind10.flatten()
    ind01 = ind01.flatten()
    ind11 = ind11.flatten()

    F = jnp.concatenate([
        jnp.stack([ind00, ind10, ind01], axis=-1),
        jnp.stack([ind11, ind01, ind10], axis=-1),
        jnp.stack([ind10, ind11, ind00], axis=-1),
        jnp.stack([ind01, ind00, ind11], axis=-1),
    ], axis=0)

    return V, F, None


def vertex2face(F, uv):
    batch_size = uv.shape[0]
    f = F.shape[0]
    uf = jnp.zeros((batch_size, f) + uv.shape[2:])
    for i in range(batch_size):
        uf = uf.at[i].set(
            (uv[i, F[:, 0], ...] + uv[i, F[:, 1], ...] + uv[i, F[:, 2], ...]) / 3
        )
    return uf


def edges(F):
    assert F.shape[-1] == 3
    E = [
        F[..., 0:2],
        F[..., 1:3],
        jnp.stack([F[..., 2], F[..., 0]], axis=-1),
    ]
    E = jnp.concatenate(E, axis=-2)
    return E


def sparse_dirac_operator_2D(F, E=None, n=None):
    """Circular boundary gradient / discrete area form (Wang-Guo-Solomon 2023).

    JAX has no native COO-sparse type that matches torch.sparse_coo_tensor;
    we return a (indices, values, shape) triple. Callers that need a dense
    matrix can densify via jnp.zeros + .at[].add.
    """
    if E is None:
        E = edges(F)
    if n is None:
        n = int(F.max()) + 1

    assert len(E.shape) == 2

    indices = jnp.stack([E.flatten(), (E[..., [1, 0]]).flatten()])
    dtype = jnp.float32
    values = 0.5 * jnp.concatenate([
        -jnp.ones_like(E[..., 0], dtype=dtype).flatten(),
        jnp.ones_like(E[..., 0], dtype=dtype).flatten(),
    ])
    return indices, values, (n, n)


def edge_lengths_from_edge_list(V, E):
    P0 = V[E[:, 0], :]
    P1 = V[E[:, 1], :]
    return jnp.sqrt(jnp.sum(jnp.square(P1 - P0), axis=-1))


def value_vertex2edge(E, VV):
    return (VV[:, E[:, 0]] + VV[:, E[:, 1]]) / 2.0


def assemble_lap_values(au, g):
    """g shared across batch; au varies per-element.

    au: [..., f, 3] -> permuted to [3, f, ...]
    g:  [3, f, 2]
    """
    au = jnp.transpose(au, [2, 1, 0])

    def _inner_prod(p, q):
        return p[:, 0, None] * au[0] * q[:, 0, None] \
            + p[:, 0, None] * au[1] * q[:, 1, None] \
            + p[:, 1, None] * au[1] * q[:, 0, None] \
            + p[:, 1, None] * au[2] * q[:, 1, None]

    v01 = _inner_prod(g[0], g[1])
    v12 = _inner_prod(g[1], g[2])
    v20 = _inner_prod(g[2], g[0])
    v00 = -(v01 + v20)
    v11 = -(v12 + v01)
    v22 = -(v20 + v12)

    VV = [v00, v11, v22, v01, v12, v20, v01, v12, v20]
    return jnp.concatenate(VV)
