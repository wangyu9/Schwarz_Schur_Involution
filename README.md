# Schwarz-Schur Involution or Sparse Inversion

This repo is the official implementation of the ICML 2025 paper 

*    **Schwarz–Schur Involution: Lightspeed Differentiable Sparse Linear Solvers**
  
        by *Yu Wang, S. Mazdak Abulnaga, Yaël Balbastre, Bruce Fischl.*
     
        *International Conference on Machine Learning (ICML) 2025.*

<img width="3200" height="1600" alt="poster-icml25" src="https://github.com/user-attachments/assets/0249964a-cb50-47cb-b07c-2c8158dfc180" />

(Title in Chinese: Schwarz-Schur翻卷, 稀疏矩阵逆)

[OpenReview](https://openreview.net/pdf?id=RKbanvzycr)

[Poster](https://www.dropbox.com/scl/fi/l3d7y338wk1nthm9zw3cy/poster-icml25.pdf?rlkey=56rf2a9ft68sitptdsx5280y7&st=h3sb7yxz&dl=0)

# System requirements

While our algorithm can run on any hardware supporting recent version of pytorch, we suggest running it on Hopper Generation & Ada Lovelace (RTX 4090, A6000 ada, H100) or newer (the Blackwell Generation: RTX 5090, B200). Earlier Ampere Generation (A100, RTX 3090) can have a noticeable performance gap. 

# Dependencies

`pytorch`: tested with pytorch 2.7. Installing Pytorch with GPU support is strongly recommended to achieve the reported level of performance.  

# Example

### Minimal example

```python
# setup the problem
import numpy as np
import torch
import sinv.torch_sparse_involution as tsi
import utility

torch.set_default_device('cuda')
dtype = torch.float32          # or torch.float64
torch.set_default_dtype(dtype)

image_hwc = utility.simple_image()

ab_wh = ((128, 128), (5, 5))   # -> grid 513 x 513

problem, alpha, beta, bc, A_scipy, B_scipy, info = \
    utility.simple_matting_laplacian(ab_wh, image_hwc)
```

After preparing the left and right hand sides in a "patchfied" form---`alpha`/`beta`, the problem can be solved using: 

```python
# Solve A x = b:
x_BWHC = tsi.sinv2d(alpha, beta, 'Neumann', ab_wh)   # [1, 513, 513, C]
```

### Faster solver via JAX (`sinv2d_via_jax`)

The Jax-compiled solver can be significantly faster than the torch implementation. 
We provide a Torch-to-Jax wrapper: `sinv_torch_via_jax.sinv2d_via_jax` runs the same
problem through the JAX single-jit fast path and hands the result back as a
torch tensor. It is a drop-in for `sinv2d` on the same `alpha`/`beta`/`ab_wh`
and returns the same `[batch, W, H, C]` layout.

The first call compiles the kernel (slow). Call `precompile_solver` once up
front so later solves hit the compiled cache:

```python
from sinv_torch_via_jax import sinv2d_via_jax, precompile_solver

# JAX version compiles the solver on first use — do it once up front.
precompile_solver(ab_wh, batch_size=1, num_rhs=3, dtype=dtype)
```

Every later call at the same shape/dtype hits the compiled cache — this is the
fast path:

```python
# with the same alpha, beta, and ab_wh
import time

torch.cuda.synchronize()
t0 = time.time()

N = 100
for i in range(N):                       # values of alpha, beta do not have to be the same over iterations.  
    x_BWHC = sinv2d_via_jax((i+1)*alpha, (i+1)*beta, ab_wh)
torch.cuda.synchronize()

print(f"solver time: {(time.time() - t0) * 1e3 / N:.1f} ms "
      f"for A.shape={A_scipy.shape}, b.shape={B_scipy.shape}")
# solver time: 10.3 ms for A.shape=(263169, 263169), b.shape=(263169, 3)
```

# Usage: `schwarz_schur_involution` and `sinv2d`

This guide covers the two public entry points to the Schwarz–Schur involution
solver in `sinv/torch_sparse_involution.py`:

- **`sinv2d`** — the simple, autograd-enabled solve of `A x = b`. Use this by default.
- **`schwarz_schur_involution`** — the lower-level solver that returns a rich
  solution object (multiple output layouts, the factorization for reuse, etc.).

> This guide describes only the plain linear solve `A x = b`. The augmented /
> bordered system (arguments `beta_u`, `gamma_v`, `omega`, `sigma`) is still
> under construction and is intentionally omitted here.

Throughout, assume:

```python
import torch
import sinv.torch_sparse_involution as tsi
torch.set_default_device('cuda:0')
```

---

## 1. The data model

The solver never sees a flat `N×N` sparse matrix. Instead the `W×H` grid is
partitioned into an `a × b` array of overlapping `w × h` patches, and the
operator/RHS are stored **patchwise**.

`ab_wh = ((a, b), (w, h))` is the single object that pins all dimensions.
From it:

```
a, b, w, h, W, H = tsi.unpack_dims(ab_wh)
W = a*(w-1) + 1        # global grid width
H = b*(h-1) + 1        # global grid height
```

**Constraints on `ab_wh`:**
- `a` and `b` should be **equal and both powers of 2** (the recursion alternates
  axes and asserts even sizes at each level).
- `w`, `h` are typically `5`. Other sizes work in the core but some helper
  builders assume `w = h = 5`.

### Input tensor shapes

| name    | shape                          | meaning                                  |
|---------|--------------------------------|------------------------------------------|
| `alpha` | `[batch, a, b, w*h, w*h]`      | per-patch local operator blocks (the `A`)|
| `beta`  | `[batch, a, b, w*h, channels]` | per-patch right-hand side (the `b`)       |

`channels` is the number of RHS columns solved simultaneously (e.g. 3 for an
RGB image, 1 for a scalar field).

### Boundary condition

`boundary_condition` (aka `BC`) is a string. For the plain solve use:

- `'Neumann'` — the usual choice; internally mapped to `'Neumann-scatter'`.
- `'Neumann-full'` — the non-eliminated Neumann variant.

---

## 2. `sinv2d` — the simple solve

```python
def sinv2d(alpha, beta, boundary_condition, ab_wh, options=''):
    ...
    return X_BWHC
```

**Returns** a single tensor `X_BWHC` of shape `[batch, W, H, channels]` — the
solution on the global grid, in `(W, H)` (width-major) layout.


### Autograd

`sinv2d` is a differentiable op. Set `requires_grad_(True)` on `alpha` and/or
`beta`, run a scalar loss through `x_BWHC`, and call `.backward()`:

```python
alpha.requires_grad_(True)
beta.requires_grad_(True)

x = tsi.sinv2d(alpha, beta, 'Neumann', ab_wh)
loss = x.sum()
loss.backward()

alpha.grad   # d loss / d alpha
beta.grad    # d loss / d beta
```

### Checking the residual

Convert the first batch element back to the flat grid ordering and compare
against a scipy reference:

```python
x_np = tsi.image_flatten(x_BWHC)[0].detach().cpu().numpy()   # [N, C]
rel_err = tsi.error_measure_using_scipy(A_scipy, B_scipy, x_np)
print(rel_err)   # ~1e-14 fp64, ~1e-5 fp32
```

---

## 3. `schwarz_schur_involution` — the lower-level solver

```python
def schwarz_schur_involution(
    alpha,
    beta,
    boundary_condition,
    wh,                              # NOTE: (w, h) only, NOT the full ab_wh
    beta_u=None, gamma_v=None, omega=None, sigma=None,   # augmented (omit)
    debug=False,
    prefactored_solution=None,       # reuse a previous factorization
    transposed_solve=False,
    bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER,
):
    return <solution object>
```

Use this when you need more than the raw `X_BWHC` tensor — the alternative
output layouts, the residual helpers, or (most importantly) the **factorization
for reuse**.

### Key argument difference from `sinv2d`

- `sinv2d` takes the full **`ab_wh`** `((a, b), (w, h))`.
- `schwarz_schur_involution` takes only **`wh`** `= (w, h)`. The `(a, b)` part is
  inferred from `alpha.shape[1:3]`.

### The solution object

The returned `sol` exposes the solution in several layouts plus reuse handles:

| attribute        | shape / type            | meaning                                        |
|------------------|-------------------------|------------------------------------------------|
| `sol.X_uv`       | `[batch, W, H, C]`      | solution, width-major grid (same as `sinv2d`)  |
| `sol.X`          | `[batch, N, C]`         | solution flattened to `(H, W)` row-major order |
| `sol.lam`        | `None` (plain solve)    | Lagrange multiplier — augmented case only      |
| `sol.ab_wh`      | `((a,b),(w,h))`         | dimensions used                                |
| `sol.dtype`      | torch dtype             |                                                |
| `sol.batch_size` | int                     |                                                |

`sol.X` is already in the order `A_scipy` expects, so residual checks are
direct:

```python
sol = tsi.schwarz_schur_involution(alpha, beta, 'Neumann', wh=ab_wh[1])

x_np = sol.X[0].detach().cpu().numpy()          # [N, C], (H,W) order
rel_err = tsi.error_measure_using_scipy(A_scipy, B_scipy, x_np)
```

### Reusing a factorization (`prefactored_solution`)

The expensive part of the solve is factorizing `alpha`. To solve the **same
operator** against **new right-hand sides**, pass the previous `sol` back in as
`prefactored_solution` and set `alpha=None`:

```python
# First solve — full factorization.
sol = tsi.schwarz_schur_involution(alpha, beta1, 'Neumann', wh=ab_wh[1])

# Subsequent solves — reuse the factorization, only the RHS changes.
sol2 = tsi.schwarz_schur_involution(
    None, beta2, 'Neumann', wh=ab_wh[1], prefactored_solution=sol,
)
x2 = sol2.X_uv
```

When `prefactored_solution` is given, `alpha` **must** be `None`; the batch
size, `ab_wh`, and dtype are taken from the prefactored object.

---

## 4. Which one should I use?

| you want…                                          | use                              |
|----------------------------------------------------|----------------------------------|
| just solve `A x = b`, get a tensor back            | `sinv2d`                         |
| autograd through the solve                         | `sinv2d` (it's an autograd op)   |
| multiple RHS against the same `A` (amortize setup) | `schwarz_schur_involution` + `prefactored_solution` |
| the flat `[N, C]` solution for a scipy residual    | `schwarz_schur_involution` → `sol.X` |
| both grid and flat layouts, or the factorization   | `schwarz_schur_involution`       |

`sinv2d` is a thin, autograd-friendly wrapper; `schwarz_schur_involution` is the
full solver it delegates to. Start with `sinv2d` and drop down only when you
need reuse or the extra outputs.

---

## 5. Common pitfalls

- **`wh` vs `ab_wh`.** `schwarz_schur_involution(wh=...)` wants `(w, h)`;
  `sinv2d(ab_wh=...)` wants `((a, b), (w, h))`. Passing the wrong one triggers
  shape assertions.
- **`a == b`, powers of 2.** Non-power-of-2 `a`/`b` fail the recursion's
  even-size asserts.
- **Layout when checking residuals.** `sinv2d`'s `X_BWHC` and `sol.X_uv` are
  `(W, H)` width-major; `error_measure_using_scipy` expects the `(H, W)`
  row-major flattening. Use `tsi.image_flatten(x_BWHC)[0]` or `sol.X[0]` — do
  not `.reshape` the grid tensor by hand.
- **dtype.** Set `torch.set_default_dtype(torch.float64)` for tight residuals;
  fp32 tops out around `1e-5` relative error.
- **Reuse requires `alpha=None`.** With `prefactored_solution`, passing a
  non-`None` `alpha` asserts.


# FAQs:

> Q: This looks great but why the method is not proposed decades ago?

A: Please refer to the paper for an extensive discussion. In short, advances in the GPU capability thanks to deep learning shift the best practice
towards algorithms like ours that better exploit parallelisms.

> Q: Does the solver support irregular meshes in addition to regular grids?

A: Not yet, but we have been working on that! Stay tuned and check back.


# Reference 

```
@inproceedings{
wang2025schwarzschur,
title={Schwarz{\textendash}Schur Involution: Lightspeed Differentiable Sparse Linear Solvers},
author={Yu Wang and Mazdak Abulnaga and Ya{\"e}l Balbastre and Bruce Fischl},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://github.com/wangyu9/Schwarz_Schur_Involution}
}
```
