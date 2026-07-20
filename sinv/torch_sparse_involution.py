"""
Author: Yu Wang (wangyu9@alum.mit.edu)

This file implements the ICML 2025 paper: 
  "Schwarz–Schur Involution: Lightspeed Differentiable Sparse Linear Solvers"
  International Conference on Machine Learning (ICML) 2025. https://openreview.net/pdf?id=RKbanvzycr
    by *Yu Wang, S. Mazdak Abulnaga, Yaël Balbastre, Bruce Fischl*
"""

import numpy as np
import torch

import time

import math

from . import algorithms as alg

from typing import Optional, Union

# Type aliases used in sinv_solve_from_scipy (and friends).
# NumpyArrayOrNone documents "a numpy array, or None". For runtime isinstance()
# checks use the NumpyArrayOrNone_types tuple below (typing constructs like
# Optional can't be passed to isinstance).
NumpyArray = np.ndarray
NumpyArrayOrNone = Optional[np.ndarray]
NumpyArrayOrNone_types = (np.ndarray, np.matrix, type(None))

DEFAULT_BOUNDARY_FIRST_ORDER = False # True # False

#######################################################

def schwarz_schur_involution(
  alpha, 
  beta, 
  boundary_condition, 
  wh, 
  beta_u=None,
  gamma_v=None,
  omega=None,
  sigma=None,
  debug=False, 
  prefactored_solution=None, 
  transposed_solve=False, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
):
  return Schwarz_Schur_involution(
    Alpha=alpha, 
    Beta=beta, 
    BC=boundary_condition, 
    wh=wh, 
    Beta_U=beta_u,
    Gamma_V=gamma_v,
    Omega=omega,
    sigma=sigma,
    debug=debug, 
    prefact_sol=prefactored_solution, 
    transposed_solve=transposed_solve, 
    bfs_order=bfs_order)


def sinv2d_aug(
  alpha, 
  beta, 
  boundary_condition, 
  ab_wh, 
  betaU = None,
  gammaV = None,
  omega = None,
  sigma = None,
  transposed_solve = False,
  options=''
):
  return Sinv2D_transposed_reuse.apply(
    alpha, 
    beta, 
    boundary_condition, 
    ab_wh, 
    betaU,
    gammaV,
    omega,
    sigma,
    transposed_solve,
    options
  )


# sinv2d = Sinv2D_transposed_reuse.apply

def sinv2d(
  alpha, 
  beta, 
  boundary_condition, 
  ab_wh, 
  options=''
):

  betaU = None
  gammaV = None
  omega = None
  sigma = None
  transposed_solve = False

  X_BWHC, lam = Sinv2D_transposed_reuse.apply(
    alpha, 
    beta, 
    boundary_condition, 
    ab_wh, 
    betaU,
    gammaV,
    omega,
    sigma,
    transposed_solve,
    options
  )

  assert lam is None
  
  return X_BWHC


def scipy_reference_solve(
  A, 
  b, 
  boundary_condition='Neumann', 
  x_grad=None, # no x to be computed. 
  U=None, 
  V=None, 
  Omega=None, 
  sigma=None, 
  lam_grad=None, # no lam to be computed.
  transposed_solve=False,
  backward=False, 
):

  from scipy import sparse
  from scipy.sparse.linalg import spsolve

  assert boundary_condition in ['Neumann', 'Neumann-full']

  class Grad():
    name = 'grad'

  def _dense_solve(op, rhs):
    # op: scipy.sparse, rhs: dense numpy [n, c]. Returns dense [n, c].
    sol = np.asarray(spsolve(sparse.csc_matrix(op), rhs))
    return sol.reshape(rhs.shape)

  augmented_system = not (Omega is None)

  # sparsity pattern of A, used to restrict grad.A (see derivatives.tex:
  #   \hat{A} = - \hat{b} \times x \odot (J_1^T J_2), i.e. the outer product of
  #   the adjoint rhs and the solution, masked by A's nonzero pattern).
  Acoo = sparse.coo_matrix(A)
  rr, cc = Acoo.row, Acoo.col

  grad = None

  if augmented_system:

    # forward operator of the augmented (KKT) system, cf. method.tex:
    #   [[A, U], [V, Omega]] [x; lam] = [b; sigma]
    M = sparse.block_array([[A, U], [V, Omega]])
    op = M.T if transposed_solve else M

    N = A.shape[0]
    rhs = np.block([[b], [sigma]])
    xak = _dense_solve(op, rhs)
    x = xak[:N]
    lam = xak[N:]
    tol = np.linalg.norm(rhs - op @ xak) / np.linalg.norm(rhs)

    if backward:
      assert x_grad is not None
      lam_grad_ = np.zeros_like(lam) if lam_grad is None else lam_grad

      # adjoint solve against the transpose of the forward operator:
      #   [g_b; g_sigma] = op^{-T} [x_grad; lam_grad]
      g_rhs = _dense_solve(op.T, np.block([[x_grad], [lam_grad_]]))
      g_b = g_rhs[:N]
      g_sig = g_rhs[N:]

      grad = Grad()
      grad.b = g_b
      grad.sigma = g_sig

      # operator gradients are the (sparsity-masked) outer product -g_rhs (x) y.
      # For a transposed solve op = M^T, so the gradient w.r.t. M is transposed.
      if not transposed_solve:
        gA_vals = -(g_b[rr] * x[cc]).sum(axis=1)
        grad.U = -(g_b @ lam.T)
        grad.V = -(g_sig @ x.T)
        grad.Omega = -(g_sig @ lam.T)
      else:
        gA_vals = -(g_b[cc] * x[rr]).sum(axis=1)
        grad.U = -(x @ g_sig.T)
        grad.V = -(lam @ g_b.T)
        grad.Omega = -(lam @ g_sig.T)

      grad.A = sparse.csr_matrix((gA_vals, (rr, cc)), shape=A.shape)
      grad.bc = None

  else:

    op = A.T if transposed_solve else A
    x = _dense_solve(op, b)
    tol = np.linalg.norm(b - op @ x) / np.linalg.norm(b)
    lam = None

    if backward:
      assert x_grad is not None

      # df/db = A^{-T} (df/dx); for a transposed solve op = A^T.
      g_b = _dense_solve(op.T, x_grad)

      grad = Grad()
      grad.b = g_b

      if not transposed_solve:
        gA_vals = -(g_b[rr] * x[cc]).sum(axis=1)
      else:
        gA_vals = -(g_b[cc] * x[rr]).sum(axis=1)

      grad.A = sparse.csr_matrix((gA_vals, (rr, cc)), shape=A.shape)
      grad.U = None
      grad.V = None
      grad.Omega = None
      grad.sigma = None
      grad.bc = None

  return x, lam, grad, (tol)


def sinv_solve_from_scipy(
  A: 'scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.coo_matrix',
  b: NumpyArray,
  boundary_condition='Neumann',
  ab_wh=((128,128),(5,5)),
  U: NumpyArrayOrNone = None,
  V: NumpyArrayOrNone = None,
  Omega: NumpyArrayOrNone = None,
  sigma: NumpyArrayOrNone = None,
  debug=False,
  # prefact_sol=None,
  transposed_solve=False,
  compare_gradients=False,
  check_sparsity=True,
):

  assert boundary_condition in ['Neumann', 'Neumann-full']
  BC = boundary_condition

  import scipy.sparse
  assert scipy.sparse.issparse(A) and A.format in ('csr', 'csc', 'coo'), \
      f'A must be a scipy.sparse csr/csc/coo matrix, got {type(A)}' \
      + (f' with format {A.format!r}' if scipy.sparse.issparse(A) else '')
  assert isinstance(b, (np.ndarray, np.matrix)), \
      f'b must be a np.ndarray or np.matrix, got {type(b)}'
  for _name, _v in [('U', U), ('V', V), ('Omega', Omega), ('sigma', sigma)]:
      assert isinstance(_v, NumpyArrayOrNone_types), \
          f'{_name} must be a np.ndarray/np.matrix or None, got {type(_v)}'

  alpha = alpha_from_scipy_sparse(A, ab_wh)
  beta =  beta_from_numpy(b, ab_wh)

  if check_sparsity:
    # make sure that the same A2 is obtained when converted back to scipy from alpha. 
    A2 = alpha_to_scipy(alpha, ab_wh, adj='all')[0]
    residual = abs(A-A2).sum()
    # print(f'residual={residual}')
    if residual > 1e-16:
      raise ValueError(f'A has unsupported sparsity: residual={residual}')

  augmented_system = not (Omega is None)

  errors = {}

  if augmented_system:

    O_scipy = Omega
    s_scipy = sigma

    dtype = torch.get_default_dtype()
    betaU =  beta_from_numpy(U, ab_wh)
    gammaV = gamma_from_numpy(V, ab_wh)
    # omega = batch_diag(torch.ones([batch_size, ranks])[...,None])
    omega = torch.tensor(Omega, dtype=dtype)[None,...]
    sigma = torch.tensor(sigma, dtype=dtype)[None,...]

  else:
    O_scipy = None
    s_scipy = None

    betaU = None
    gammaV = None
    omega = None
    sigma = None
  
  if compare_gradients:
      for t in [alpha, beta, BC, betaU, gammaV, omega, sigma]:
          if (t is not None) and not isinstance(t, str):
              t.requires_grad_(True)
              # assert t.requires_grad==True

  if True:
    # new API:
    assert BC=='Neumann-full'

    x_BWHC, lam = sinv2d_aug(alpha, beta, BC, ab_wh, 
        betaU=betaU, gammaV=gammaV, omega=omega, sigma=sigma, 
        transposed_solve=transposed_solve)

    x_scipy, lam_scipy = sinv2d_aug_numpy(x_BWHC, lam, index=0)
    
    sol = None
        
  else:
    
    sol = schwarz_schur_involution(alpha, beta, boundary_condition=BC, 
        beta_u=betaU, gamma_v=gammaV, omega=omega, sigma=sigma, 
        wh=ab_wh[1], transposed_solve=transposed_solve)

    x_scipy, lam_scipy = sol2numpy(sol, index=0, require_lam=True)

  if compare_gradients:

      assert BC in ['Neumann-full'] # otherwise not implemented.

      x_BWHC_grad = torch.ones_like(x_BWHC, dtype=x_BWHC.dtype)
      lam_grad = None if lam is None else torch.ones_like(lam, dtype=lam.dtype)

      grad_torch = autograd_gradients(  alpha, beta, BC, x_BWHC=x_BWHC, x_BWHC_grad=x_BWHC_grad,
              betaU=betaU,gammaV=gammaV,omega=omega, sigma=sigma, lam=lam, lam_grad=lam_grad
          )

      # Recompute the same gradients through pure scipy and compare with grad_torch.
      
      x_grad_np = image_flatten(x_BWHC_grad.detach())[0].cpu().numpy()
      lam_grad_np = None if lam_grad is None else lam_grad[0].cpu().numpy()

      # x_grad_np = np.ones_like(x_scipy)
      # lam_grad_np = None if lam_scipy is None else np.ones_like(lam_scipy)

      _, _, grad_scipy, _ = scipy_reference_solve(
          A, b, boundary_condition='Neumann',
          x_grad=x_grad_np, lam_grad=lam_grad_np,
          U=U, V=V, Omega=O_scipy, sigma=s_scipy,
          transposed_solve=transposed_solve, backward=True,
      )

      errors = compare_gradients_with_scipy(grad_torch, grad_scipy, ab_wh, transposed_solve=transposed_solve, alpha_mask='9pt') 

  tol = error_measure_using_scipy(A, b, x_scipy, 
            U=U, V=V, Omega=O_scipy, sigma=s_scipy, lam=lam_scipy, 
            transposed_solve=transposed_solve)

  errors['tol'] = tol
  # print('shapes', x_scipy.shape, None if lam_scipy is None else lam_scipy.shape)

  return x_scipy, lam_scipy, sol, errors


#######################################################

def convert_to_boundary_first_ordering(alpha, wh):
  '''
  spiral ordering for boundary nodes, and [w-2, h-2] array ordering for interior nodes.
  see Figure 26 in the paper. 
  '''

  PB, PUB = alg.matrix_bdr_interior_split(wh=wh)

  #w, h = wh[0], wh[1]
  #PB = alg.matrix_spiral_bdr_list(w,h)
  #PUB = alg.matrix_interior_list(w,h)

  ind = torch.tensor(PB + PUB, dtype=torch.int64)

  # print(ind)
  # tensor([ 0,  1,  2,  3,  4,  9, 14, 19, 24, 23, 22, 21, 20, 15, 10,  5,  6,  7,
  #        8, 11, 12, 13, 16, 17, 18], device='cuda:0')

  alpha_permu = alpha[...,ind,:][...,:,ind]
  # the indexing [...,ind,:][...,:,ind]  is only for right-hand-side, wrong for lhs

  return alpha_permu


def convert_to_boundary_first_ordering_rhs(beta, wh, beta_type=True):

  PB, PUB = alg.matrix_bdr_interior_split(wh=wh)

  ind = torch.tensor(PB + PUB, dtype=torch.int64)

  if beta_type:
    permu = beta[...,ind,:]
  else:
    permu = beta[...,:,ind]

  return permu


def convert_to_lexicographic_sweeping_order(alpha, wh):

  B, UB = alg.matrix_bdr_interior_split(wh=wh)
  
  invBUB = alg.inverse_permutation(B+UB)

  # print(invBUB)

  invd = torch.tensor(invBUB, dtype=torch.int64)

  # syntax not supported with wrong output shape: IV = DA[..., invd, invd]
  alpha_permu = alpha[...,invd,:][...,:,invd]

  return alpha_permu


def torch_batch_cartesian_prod(A, B):

  # assert torch.is_complex(B)==False
  if torch.is_complex(B) or torch.is_complex(A): 
    print('warning: double check with your use case')
  assert A.shape[-1]==1
  assert B.shape[-1]==1
  dims = len(A.shape[:-2])
  assert dims == len(B.shape[:-2])
  a = A.shape[-2]
  b = B.shape[-2]
  BT = B.transpose(-1, -2)
  ''' this transpose should not be hermitian---purely for the purpose of layout change'''

  return A.expand([-1]*dims + [-1] + [b]), BT.expand([-1]*dims + [a] + [-1])


def hermitian(XX):
  '''
  take an extra conjugate for complex-valued tensors
  renamed to hermitian from transpose
  '''

  # both alpha, beta, gamma can use this function. 

  return torch.conj(torch.transpose(XX, -1, -2))


def torch_batch_outer(A, B):

  assert len(A.shape)==len(B.shape)
  assert A.shape==B.shape
  
  return A @ hermitian(B)


def interpolate(BCHW, WH_s, WH):

  upsampled = torch.nn.functional.interpolate(BCHW, size=(WH[1],WH[0]),  mode='bilinear', align_corners=True)
  # nearest, bilinear
  # numpy grammar: upsampled = upsampled.transpose([0,3,2,1])

  return upsampled


def vector_interpolate(XX_s, WH_s, WH):
  # XX_s = torch.tensor(XX_s, dtype=torch.float64) # if input is not torch.tensor

  XX_s = XX_s[None,...] # B, H*W, C

  assert len(XX_s.shape)==3

  BCHW = image_unflatten( XX_s, WH_s[0], WH_s[1]).permute([0,3,2,1])

  # print(f'BCHW.shape={BCHW.shape}')

  BCHW_upsampled = torch.nn.functional.interpolate(BCHW, size=(WH[1],WH[0]),  mode='bilinear', align_corners=True)

  BWHC = BCHW_upsampled.permute([0,3,2,1])

  XX = image_flatten( BWHC )

  XX = XX[0,...]

  return XX


def inv_default_exact(xx):

  # invf = lambda xx : torch.linalg.inv(xx)
  # invf = lambda xx : torch.linalg.inv_ex(xx)[0]
  
  # result = torch.linalg.inv(xx)
  result, info = torch.linalg.inv_ex(xx) # same result but faster. 
  # E = torch.eye(xx.shape[-1])[...]
  # result, info = torch.linalg.solve_ex(xx, E);
  # result = torch.linalg.solve(xx, E);
  # LU, pivots, info = torch.linalg.lu_factor_ex(xx); result = torch.linalg.lu_solve(LU, pivots, E)

  return result


def fetch_DtN_exact(xx):
  return xx


def fetch_DtN_approx(xx):
  win = 256 # 64
  if xx.shape[-1] < (win-1):
    yy = xx
  else:
    n = xx.shape[-1]
    yy = xx - torch.tril(xx, diagonal=-win) - torch.triu(xx, diagonal=win) \
        + torch.tril(xx, diagonal=-(n-win)) + torch.triu(xx, diagonal=(n-win)) 
  return yy


def inv_default_approx(xx):

  # this works surprisingly well, without modifying collapse_subdomains. 
  # problem=matting_lap_0.01_coffee	ab_wh=((256, 256), (5, 5)):  
  # error 1.5168593563430477e-05 --> 2.9678683858946897e-05 
  # problem=matting_lap_0.01_coffee	ab_wh=((128, 128), (5, 5)):  
  # error 1.0e-05 --> 1.3e-05

  print('approx inv called')

  win = 256 # 64
  if xx.shape[-1] < (win-1):
    result, info = torch.linalg.inv_ex(xx) 
  else:
    n = xx.shape[-1]
    yy = xx - torch.tril(xx, diagonal=-win) - torch.triu(xx, diagonal=win) \
        + torch.tril(xx, diagonal=-(n-win)) + torch.triu(xx, diagonal=(n-win)) 
    result, info = torch.linalg.inv_ex(yy)  
    # result = torch.linalg.inv(yy)  

    if False:
      result = result - torch.tril(result, diagonal=-win) - torch.triu(result, diagonal=win) \
          + torch.tril(result, diagonal=-(n-win)) + torch.triu(result, diagonal=(n-win)) 

  return result


# inv_default = inv_default_approx
# fetch_DtN = fetch_DtN_approx
inv_default = inv_default_exact
fetch_DtN = fetch_DtN_exact

# def inv_default(xx):
#   return inv_default_exact(xx)


def unpack_dims(ab_wh):

  ab = ab_wh[0]
  wh = ab_wh[1]
  w = wh[0]
  h = wh[1]
  a = ab[0]
  b = ab[1]
  W = a*(w-1)+1
  H = b*(h-1)+1

  return a,b,w,h,W,H


def image_to_torch_batch(image):
  """
  assume image is C x H x W
  """

  if len(image.shape)==2:
    image = image[np.newaxis,...]

  image_batch = image[np.newaxis,...]

  torch_image = torch.tensor(image_batch, requires_grad=False, dtype=torch.get_default_dtype())

  return torch_image


class Level():
  name = 'Level_'


def cf(arr, same=True):

  if not same: 
    arr = torch.flip(arr, [-2]) # assume the last channel is feature.

  return arr


def cf_gamma(arr, same=True):

  if not same: 
    arr = torch.flip(arr, [-1]) # assume the second-to-last channel is feature.

  return arr


def df(arr, same=[True,False]):

  dims = []
  if not same[0]:
    dims.append(-2)
  if not same[1]:
    dims.append(-1)
  if len(dims)>0:
    arr = torch.flip(arr, dims)

  return arr


def image_flatten(ZZ):
  
  return torch.reshape(ZZ.transpose(-2,-3), list(ZZ.shape[:-3]) + [ZZ.shape[-2] * ZZ.shape[-3], -1])
  # if len(ZZ.shape)==4:
  #   return torch.reshape(ZZ.transpose(1,2), [ZZ.shape[0], ZZ.shape[1] * ZZ.shape[2], -1])
  # else:
  #   return torch.reshape(ZZ.transpose(-2,-3), list(ZZ.shape[:-3]) + [ZZ.shape[-2] * ZZ.shape[-3], -1])


def image_unflatten(Z, dim0, dim1):
  '''
  reshapes a tensor from [..., dim0*dim1, C] to [..., dim0, dim1, C]
  for an W x H x C image but make data contiguous along the W channel. 
  '''
  # old implementation: 
  # if len(Z.shape)==3:
  #   return torch.reshape(Z, [Z.shape[0], dim1, dim0, -1]).transpose(1,2)
  # else:
  #   assert len(Z.shape)==2
  #   return torch.reshape(Z, [dim1, dim0, -1]).transpose(0,1)
  #
  # another older version.
  # if len(Z.shape)==2: 
  #   return torch.reshape(Z, [dim1, dim0, -1]).transpose(0,1)
  # elif len(Z.shape)==3:
  #   return torch.reshape(Z, [Z.shape[0], dim1, dim0, -1]).transpose(1,2)
  # else:
  #   return torch.reshape(Z, list(Z.shape[:-2]) + [dim1, dim0, -1]).transpose(-2,-3)

  return torch.reshape(Z, list(Z.shape[:-2]) + [dim1, dim0, -1]).transpose(-2,-3)


def get_boundary(X_uv, size):
  
  m = size[0]
  n = size[1]

  return torch.cat([
      X_uv[..., 0:m-1,  0,  :],
      X_uv[..., m-1, 0:n-1, :],
      torch.flip( X_uv[..., 1:m, n-1, :], dims=[-2]),  # not -2 since singleton dim removed
      torch.flip( X_uv[..., 0, 1:n, :], dims=[-2]) # not 1 since dim-0 removed
    ],dim=-2)


def flatten_and_permute(X_uv, pq):

  """ 
  X_uv: [..., p, q, channels]
  flatten the tensor X_uv, and permute the elements to boundary first ordering.
  """

  p, q = pq[0], pq[1]

  assert X_uv.shape[-3]==p
  assert X_uv.shape[-2]==q

  X_B = get_boundary(X_uv, size=[p,q])

  X_UB = X_uv[..., 1:p-1, 1:q-1, :].transpose(-3,-2)

  X_UB = torch.reshape(X_UB, X_UB.shape[:-3]+torch.Size([X_UB.shape[-3]*X_UB.shape[-2], X_UB.shape[-1]]))

  return torch.cat([X_B, X_UB], dim=-2)


def fillin_bfs_order(BC, Z, p, q):

  # the image is p x q, enlarged from (p-1,q-1)
  # todo: rename p,q to w,h

  Z_uv = image_unflatten(Z, p-2, q-2)

  dtype = Z.dtype

  if True:
    sz = list(Z_uv.shape)
    # X_uv = torch.zeros(size=[sz[0],sz[1]+2,sz[2]+2,sz[3]], dtype=dtype)
    sz[-2] = sz[-2] + 2
    sz[-3] = sz[-3] + 2
    X_uv = torch.zeros(size=sz, dtype=dtype)
    X_uv[...,1:-1,1:-1,:] = Z_uv
    if False:
      '''
      do not work as negative stepsize not supported yet. Future Torch version?
      '''
      X_uv[...,0:p-1,   0, :] = BC[..., 0:p-1, :]
      X_uv[..., -1, 0:q-1, :] = BC[..., p-1:p+q-2, :]
      X_uv[..., p-1:0:-1, -1, :] = BC[..., p+q-2:2*p+q-3, :]
      X_uv[...,  0, q-1:0:-1, :] = BC[..., 2*p+q-3:, :]
    else:
      X_uv[...,0:p-1,   0, :] = BC[..., 0:p-1, :]
      X_uv[..., -1, 0:q-1, :] = BC[..., p-1:p+q-2, :]
      X_uv[..., 1:p, -1, :] = torch.flip(BC[..., p+q-2:2*p+q-3, :], dims=[-2])
      X_uv[...,  0, 1:q, :] = torch.flip(BC[..., 2*p+q-3:, :], dims=[-2])
  else:
    assert False
    # TODO: more efficient slicing.

  X = image_flatten(X_uv)

  return X, X_uv, Z_uv


def solution_fillin(sol, BC, Z, p, q):
  
  sol.X_p = torch.concat([BC, Z], axis=-2)

  sol.X, sol.X_uv, sol.Z_uv = fillin_bfs_order(BC, Z, p, q)

  return


def gather_contributions(S):
  return S.sum(dim=[1,2])


def scatter_boundary_eliminate_pre(
  dim_pq, DtN_ori, RHS_ori, 
  pre_sol=None, 
  transposed_solve=False, 
  cache=True, 
  invf=inv_default
):
  """
  history: see c61d0f9 for old versions---scatter_boundary_eliminate: 
    def scatter_boundary_eliminate_pre_old(dim_pq, DtN_ori, RHS_ori, pre_sol=None, cache=True):
      return *scatter_boundary_eliminate(dim_pq, DtN_ori, RHS_ori), None 
  """

  class Solution:
    name = 'scatter_boundary_eliminate_pre'

  # invf = lambda xx : torch.linalg.inv(xx)
  # invf = lambda xx : torch.linalg.inv_ex(xx)[0]

  sol = Solution()

  if cache==False:
    pre_sol = None
    assert transposed_solve==False

  if pre_sol is None:
    DtN = DtN_ori.clone()
  else:
    DtN = None
  RHS = RHS_ori.clone()

  #DtN = DtN_ori
  #RHS = RHS_ori

  # https://en.wikipedia.org/wiki/Schur_complement#Application_to_solving_linear_equations
  #
  # [A, B;  [x]=[u]
  #  C, D;] [y] [v]

  # DtN is [batch_size, a, b, xxx, xxx]
  # RHS is [batch_size, a, b, xxx, c]

  # frame: 
  # [2p+q-3,...,p+q-2]
  # [2p+q-2,...,p+q-1]
  #
  # [2p+2q-5, ...  ,p]
  # [0,1,....p-2, p-1]

  p = dim_pq[0]
  q = dim_pq[1]

  # print(f'scatter_boundary_eliminate: ({p,q})')

  irs = 2*(p+q)-4

  # remove bottom wires for DtN[:, :, 0]
  # keep: 0 and [p-1,end), remove: [1,p-1)
  ind = 0
  ka = slice(0,1)
  kb = slice(p-1,None)
  rm = slice(1,p-1)
  if pre_sol is None:
    Block = DtN[:, :, ind] # Block is the *reference* to the subblock, modifying Block changes DtN.
    invA1 = invf(DtN[:, :, ind, rm, rm])
    # important, must clone it here. having tested that without clone it yields very wrong results.
    Ba1 = Block[..., rm, ka].clone()
    Bb1 = Block[..., rm, kb].clone()
    Ca1 = Block[..., ka, rm]
    Cb1 = Block[..., kb, rm]
    if cache:
      Ca1 = Ca1.clone()
      Cb1 = Cb1.clone()
  else:
    invA1 = pre_sol.invA1
    Ba1 = pre_sol.Ba1
    Bb1 = pre_sol.Bb1
    Ca1 = pre_sol.Ca1
    Cb1 = pre_sol.Cb1
  uu1 = RHS[:, :, ind, rm, :].clone()
  def back_fill1(nBC, transposed_solve=False):
    ind = 0
    ka = slice(0,1)
    kb = slice(p-1,None)
    rm = slice(1,p-1)
    nBC = nBC.clone()
    #print(f'in:{nBC[:,:,0,1:4,:].norm()}')
    ya = nBC[:, :, ind, ka, :]
    yb = nBC[:, :, ind, kb, :]
    if not transposed_solve:
      nBC[:, :, ind, rm, :] = invA1 @ (uu1 - Ba1 @ ya - Bb1 @ yb)
    else:
      nBC[:, :, ind, rm, :] = hermitian( ( hermitian(uu1) - hermitian(ya) @ Ca1 - hermitian(yb) @ Cb1) @ invA1 )
    #print(f'out:{nBC[:,:,0,1:4,:].norm(), rm, (nBC[:, :, ind, rm, :]).norm(), yb.norm()}')
    return nBC
  if pre_sol is None:
    Block[..., ka, ka] -= Ca1 @ invA1 @ Ba1
    Block[..., ka, kb] -= Ca1 @ invA1 @ Bb1
    Block[..., kb, ka] -= Cb1 @ invA1 @ Ba1
    Block[..., kb, kb] -= Cb1 @ invA1 @ Bb1
    #DtN[:, :, ind, rm, :] = 0
    #DtN[:, :, ind, :, rm] = 0
  # A: W, B: Z, C: Y, D: X, uu: w, y: x. 
  if not transposed_solve:
    RHS[:, :, ind, ka, :] -=    Ca1 @ (invA1 @ RHS[:, :, ind, rm, :])
    RHS[:, :, ind, kb, :] -=    Cb1 @ (invA1 @ RHS[:, :, ind, rm, :])
  else:
    RHS[:, :, ind, ka, :] -=    hermitian( ( hermitian(RHS[:, :, ind, rm, :]) @ invA1 ) @ Ba1 )
    RHS[:, :, ind, kb, :] -=    hermitian( ( hermitian(RHS[:, :, ind, rm, :]) @ invA1 ) @ Bb1 )
  #RHS[:, :, ind, rm, :] = 0

  # remove up wires for DtN[:, :, -1]
  # remove: [p+q-1,2*p+q-3), the other
  ind = -1
  ka = slice(0, p+q-1)
  kb = slice(2*p+q-3,None)
  rm = slice(p+q-1,2*p+q-3)
  if pre_sol is None:
    Block = DtN[:, :, ind]
    invA2 = invf(DtN[:, :, ind, rm, rm])
    Ba2 = Block[..., rm, ka].clone()
    Bb2 = Block[..., rm, kb].clone()
    Ca2 = Block[..., ka, rm]
    Cb2 = Block[..., kb, rm]
    if cache:
      Ca2 = Ca2.clone()
      Cb2 = Cb2.clone()
  else:
    invA2 = pre_sol.invA2
    Ba2 = pre_sol.Ba2
    Bb2 = pre_sol.Bb2
    Ca2 = pre_sol.Ca2
    Cb2 = pre_sol.Cb2
  uu2 = RHS[:, :, ind, rm, :].clone()
  def back_fill2(nBC, transposed_solve=False):
    ind = -1
    ka = slice(0, p+q-1)
    kb = slice(2*p+q-3,None)
    rm = slice(p+q-1,2*p+q-3)
    nBC = nBC.clone()
    ya = nBC[:, :, ind, ka, :]
    yb = nBC[:, :, ind, kb, :]
    if not transposed_solve:
      nBC[:, :, ind, rm, :] = invA2 @ (uu2 - Ba2 @ ya - Bb2 @ yb)
    else:
      nBC[:, :, ind, rm, :] = hermitian( ( hermitian(uu2) - hermitian(ya) @ Ca2 - hermitian(yb) @ Cb2) @ invA2 )
    return nBC
  if pre_sol is None:
    DtN[:, :, ind, ka, ka] -= Ca2 @ invA2 @ Ba2
    DtN[:, :, ind, ka, kb] -= Ca2 @ invA2 @ Bb2
    DtN[:, :, ind, kb, ka] -= Cb2 @ invA2 @ Ba2
    DtN[:, :, ind, kb, kb] -= Cb2 @ invA2 @ Bb2
    #DtN[:, :, ind, rm, :] = 0
    #DtN[:, :, ind, :, rm] = 0
  # A: W, B: Z, C: Y, D: X, uu: w, y: x.
  if not transposed_solve:
    RHS[:, :, ind, ka, :] -=    Ca2 @ (invA2 @ RHS[:, :, ind, rm, :])
    RHS[:, :, ind, kb, :] -=    Cb2 @ (invA2 @ RHS[:, :, ind, rm, :])
  else:
    RHS[:, :, ind, ka, :] -=    hermitian( ( hermitian(RHS[:, :, ind, rm, :]) @ invA2 ) @ Ba2 )
    RHS[:, :, ind, kb, :] -=    hermitian( ( hermitian(RHS[:, :, ind, rm, :]) @ invA2 ) @ Bb2 )
  #RHS[:, :, ind, rm, :] = 0


  # remove right wires for DtN[:, -1, :]
  # remove: [p,p+q-2)   the other
  ind = -1
  ka = slice(0, p)
  kb = slice(p+q-2,None)
  rm = slice(p,p+q-2)
  if pre_sol is None:
    Block = DtN[:, ind, :]
    invA3 = invf(DtN[:, ind, :, rm, rm])
    Ba3 = Block[..., rm, ka].clone()
    Bb3 = Block[..., rm, kb].clone()
    Ca3 = Block[..., ka, rm]
    Cb3 = Block[..., kb, rm]
    if cache:
      Ca3 = Ca3.clone()
      Cb3 = Cb3.clone()
  else:
    invA3 = pre_sol.invA3
    Ba3 = pre_sol.Ba3
    Bb3 = pre_sol.Bb3
    Ca3 = pre_sol.Ca3
    Cb3 = pre_sol.Cb3
  uu3 = RHS[:, ind, :, rm, :].clone()
  def back_fill3(nBC, transposed_solve=False):
    ind = -1
    ka = slice(0, p)
    kb = slice(p+q-2,None)
    rm = slice(p,p+q-2)
    nBC = nBC.clone()
    ya = nBC[:, ind, :, ka, :]
    yb = nBC[:, ind, :, kb, :]
    if not transposed_solve:
      # note the indices are different from 1 and 2 for 3 and 4
      nBC[:, ind, :, rm, :] = invA3 @ (uu3 - Ba3 @ ya - Bb3 @ yb)
    else:
      nBC[:, ind, :, rm, :] = hermitian( ( hermitian(uu3) - hermitian(ya) @ Ca3 - hermitian(yb) @ Cb3) @ invA3 )
    return nBC
  if pre_sol is None:
    Block[..., ka, ka] -= Ca3 @ invA3 @ Ba3
    Block[..., ka, kb] -= Ca3 @ invA3 @ Bb3
    Block[..., kb, ka] -= Cb3 @ invA3 @ Ba3
    Block[..., kb, kb] -= Cb3 @ invA3 @ Bb3
    #DtN[:, ind, :, rm, :] = 0
    #DtN[:, ind, :, :, rm] = 0
  # A: W, B: Z, C: Y, D: X, uu: w, y: x.
  if not transposed_solve:
    # note the indices are different from 1 and 2 for 3 and 4, also for the multiplied RHS[:, ind, :, rm, :]
    RHS[:, ind, :, ka, :] -=    Ca3 @ (invA3 @ RHS[:, ind, :, rm, :])
    RHS[:, ind, :, kb, :] -=    Cb3 @ (invA3 @ RHS[:, ind, :, rm, :])
  else:
    RHS[:, ind, :, ka, :] -=    hermitian( ( hermitian(RHS[:, ind, :, rm, :]) @ invA3 ) @ Ba3 )
    RHS[:, ind, :, kb, :] -=    hermitian( ( hermitian(RHS[:, ind, :, rm, :]) @ invA3 ) @ Bb3 )
  #RHS[:, ind, :, rm, :] = 0

  # remove left wires for DtN[:, 0, :]
  # remove: [p,p+q-2)   the other
  ind = 0
  ka = slice(0, 1)
  kb = slice(1, 2*p+q-2)
  rm = slice(2*p+q-2, None)
  if pre_sol is None:
    Block = DtN[:, ind, :]
    invA4 = invf(DtN[:, ind, :, rm, rm])
    Ba4 = Block[..., rm, ka].clone()
    Bb4 = Block[..., rm, kb].clone()
    Ca4 = Block[..., ka, rm]
    Cb4 = Block[..., kb, rm]
    if cache:
      Ca4 = Ca4.clone()
      Cb4 = Cb4.clone()
  else:
    invA4 = pre_sol.invA4
    Ba4 = pre_sol.Ba4
    Bb4 = pre_sol.Bb4
    Ca4 = pre_sol.Ca4
    Cb4 = pre_sol.Cb4
  uu4 = RHS[:, ind, :, rm, :].clone()
  def back_fill4(nBC, transposed_solve=False):
    ind = 0
    ka = slice(0, 1)
    kb = slice(1, 2*p+q-2)
    rm = slice(2*p+q-2, None)
    nBC = nBC.clone()
    ya = nBC[:, ind, :, ka, :]
    yb = nBC[:, ind, :, kb, :]
    if not transposed_solve:
      # note the indices are different from 1 and 2 for 3 and 4
      nBC[:, ind, :, rm, :] = invA4 @ (uu4 - Ba4 @ ya - Bb4 @ yb)
    else:
      nBC[:, ind, :, rm, :] = hermitian( ( hermitian(uu4) - hermitian(ya) @ Ca4 - hermitian(yb) @ Cb4) @ invA4 )
    return nBC
  if pre_sol is None:  
    Block[..., ka, ka] -= Ca4 @ invA4 @ Ba4
    Block[..., ka, kb] -= Ca4 @ invA4 @ Bb4
    Block[..., kb, ka] -= Cb4 @ invA4 @ Ba4
    Block[..., kb, kb] -= Cb4 @ invA4 @ Bb4
    #DtN[:, ind, :, rm, :] = 0
    #DtN[:, ind, :, :, rm] = 0
  # A: W, B: Z, C: Y, D: X, uu: w, y: x.
  if not transposed_solve:
    # note the indices are different from 1 and 2 for 3 and 4, also for the multiplied RHS[:, ind, :, rm, :]
    RHS[:, ind, :, ka, :] -=    Ca4 @ (invA4 @ RHS[:, ind, :, rm, :])
    RHS[:, ind, :, kb, :] -=    Cb4 @ (invA4 @ RHS[:, ind, :, rm, :])
  else:
    RHS[:, ind, :, ka, :] -=    hermitian( ( hermitian(RHS[:, ind, :, rm, :]) @ invA4 ) @ Ba4 )
    RHS[:, ind, :, kb, :] -=    hermitian( ( hermitian(RHS[:, ind, :, rm, :]) @ invA4 ) @ Bb4 )
  #RHS[:, ind, :, rm, :] = 0

  sol.invA1 = invA1
  sol.Ba1 = Ba1
  sol.Bb1 = Bb1
  sol.Ca1 = Ca1
  sol.Cb1 = Cb1

  sol.invA2 = invA2
  sol.Ba2 = Ba2
  sol.Bb2 = Bb2
  sol.Ca2 = Ca2
  sol.Cb2 = Cb2

  sol.invA3 = invA3
  sol.Ba3 = Ba3
  sol.Bb3 = Bb3
  sol.Ca3 = Ca3
  sol.Cb3 = Cb3

  sol.invA4 = invA4
  sol.Ba4 = Ba4
  sol.Bb4 = Bb4
  sol.Ca4 = Ca4
  sol.Cb4 = Cb4

  def nBC_back_fill(nBC, transposed_solve):
    # nBC: [batch_size, a, b, 2*(p+q-2), c]
    # in reverse order: important
    nBC = back_fill1(back_fill2(back_fill3(back_fill4(nBC, transposed_solve), transposed_solve), transposed_solve), transposed_solve)
    return nBC

  return DtN, RHS, nBC_back_fill, sol


def batch_Dirichlet_solve_pre(dim_pq, dLA, Beta_U, Gamma_V, Omega, invf=inv_default): 
  """
  assume rows/cols of dLA are in boundary first ordering.  
  """
  # invf = lambda xx : torch.linalg.inv_ex(xx)[0]

  augmented_system = (Omega!=None)

  p = dim_pq[0]
  q = dim_pq[1]

  irs = 2*(p+q)-4

  dLA_rr = dLA[..., 0:irs, 0:irs] # X
  dLA_rs = dLA[..., 0:irs, irs:]  # Y
  dLA_sr = dLA[..., irs:,  0:irs] # Z
  dLA_ss = dLA[..., irs:,  irs:]  # W

  def split_RHS(RHS):
    RHS_r = RHS[..., 0:irs, :]  # x
    RHS_s = RHS[..., irs:, :]   # z
    return RHS_s, RHS_r

  '''
  if False:
    # CL = torch.linalg.cholesky(self.dLA_ss) # Extract Cholesky decomposition
    dLA_ss_CL, info = torch.linalg.cholesky_ex(dLA_ss, upper=True)

    # note right hand side goes first: 
    IX = torch.cholesky_solve(dLA_sr, dLA_ss_CL)
    def rhs_process(RHS_s):
      IRHS = torch.cholesky_solve(RHS_s, dLA_ss_CL)
      return IRHS
    # work in progress.
    assert False # only for symmetric matrix.
  else:
    IX = torch.linalg.solve(A=dLA_ss, B=dLA_sr)
    def rhs_process(RHS_s):
      IRHS = torch.linalg.solve(A=dLA_ss, B=RHS_s)
      return IRHS
  '''

  inv_dLA_ss = invf(dLA_ss)

  reused_prod = dLA_rs @ inv_dLA_ss # Y * inv(W)
  # DtN = dLA_rr - dLA_rs @ IX
  DtN = dLA_rr - reused_prod @ dLA_sr

  if augmented_system:

    Beta_U_r = Beta_U[..., 0:irs, :] # Ux
    Beta_U_s = Beta_U[..., irs:, :]  # Uz
    
    Gamma_V_r = Gamma_V[..., :, 0:irs] # Vx
    Gamma_V_s = Gamma_V[..., :, irs:]  # Vz
    
    reused_prod_gamma = Gamma_V_s @ inv_dLA_ss

    mBeta_U = Beta_U_r - reused_prod @ Beta_U_s
    mGamma_V = Gamma_V_r - reused_prod_gamma @ dLA_sr # inv(W) * Z
    mOmega = Omega - gather_contributions(Gamma_V_s @ inv_dLA_ss @ Beta_U_s) 

  else:

    mBeta_U = None
    mGamma_V = None
    mOmega = None

  def get_new_RHS(RHS, CRHS, transposed_solve=False):
    """
    the oracle to get the new right hand side, after the block elimination.
    """

    RHS_s, RHS_r = split_RHS(RHS)

    if not transposed_solve:
      '''
        # old code to be removed:
        IRHS = rhs_process(RHS_s)
        nRHS = RHS_r - dLA_rs @ IRHS
      '''

      nRHS = RHS_r - reused_prod @ RHS_s # y − Y W^−1 w
      nCRHS = CRHS - gather_contributions(reused_prod_gamma @ RHS_s) if augmented_system else None

    else:

      # (y⊺ − w⊺ W^−1 Z ) ⊺
      nRHS = RHS_r - hermitian( ( hermitian(RHS_s) @ inv_dLA_ss ) @ dLA_sr )
      nCRHS = CRHS - gather_contributions( hermitian( ( hermitian(RHS_s) @ inv_dLA_ss ) @ Beta_U_s ) ) if augmented_system else None

    return nRHS, nCRHS

  class Solution:
    name = 'Dirichlet_solve: solution'

  def batchwise_Dirichlet_solver(BC, RHS, lam, compute_energy=True, transposed_solve=False):
    """
    back-sub that fills in the interior values of each patch domain. 
    """

    augmented_system = (lam!=None)

    sol = Solution()

    sol.Y = BC.to_dense() # x, which is known (from an upper level solver)

    RHS_s, RHS_r = split_RHS(RHS) # w, y

    if not transposed_solve:
      # z = W^−1 [w − Z x - Uz λ]
      rhs = RHS_s - (dLA_sr @ sol.Y)
      if augmented_system:
        rhs = rhs - Beta_U_s @ lam # Uz λ
      # sol.Z = torch.cholesky_solve( rhs, dLA_ss_CL)
      # sol.Z, info = torch.linalg.solve_ex(A=dLA_ss, B=rhs)
      sol.Z = inv_dLA_ss @ rhs
    else:
      # z = ( ( w⊺ - x⊺ Y - λ⊺ Vz) W^−1 ) ⊺
      rhs = hermitian(RHS_s) - hermitian(sol.Y) @ dLA_rs
      if augmented_system:
        rhs = rhs - hermitian(lam) @ Gamma_V_s
      sol.Z =  hermitian( ( rhs ) @ inv_dLA_ss )

    solution_fillin(sol, sol.Y, sol.Z, p=p, q=q)

    if compute_energy:

      def eval_ADirichlet():
        assert transposed_solve == False
        assert torch.is_complex(sol.Y) == False 
        '''have not checked this for complex, unsymmetric cases'''
        AD = hermitian(sol.Y) @ (dLA_rr @ sol.Y + dLA_rs @ sol.Z) 
        assert(AD.shape[-2]==2)
        assert(AD.shape[-1]==2)
        return AD[...,0,0] + AD[...,1,1]

      sol.eval_ADirichlet = eval_ADirichlet 

    return sol

  return DtN, get_new_RHS, batchwise_Dirichlet_solver, mBeta_U, mGamma_V, mOmega


def merge_blocks_split_value(CX, dim=0, coeff=0.5):

  if dim==0:
    # print(f'shape is {CX[:,0::2,:,:-1,...].shape} and {CX[:,0::2,:,-1:-1,...].shape}')
    return torch.concat(
      [ CX[:,0::2,:,:-1,...], \
        coeff*(CX[:,0::2,:,[-1],...]+CX[:,1::2,:,[0],...]), \
        CX[:,1::2,:,1:,...]], dim=3)
  else:
    assert dim==1
    return torch.concat(
      [ CX[:,:,0::2,:,:-1,...], \
        coeff*(CX[:,:,0::2,:,[-1],...]+CX[:,:,1::2,:,[0],...]), \
        CX[:,:,1::2,:,1:,...]], dim=4)
  
  return # with CX modified


def bottom_up(XX, value_sum=True): 

  """
  XX: [batch_size, a, b, p, q, c]
  this does not apply to Alpha or Beta.
  """

  size = XX.shape
  a = size[1]
  b = size[2]
  
  N_iter = math.floor(math.log2(a)+math.log2(b))

  CX = XX

  coeff = 1 if value_sum else 0.5

  for i in range(N_iter):
    # print(f'i={i}')
    if i%2==0:
      CX = merge_blocks_split_value(CX, dim=0, coeff=coeff)
    else:
      CX = merge_blocks_split_value(CX, dim=1, coeff=coeff)
    # print(f'CX:{CX.shape}')

  return CX


def top_down(XX, 
  div_level='Auto', # Note: keep div_level the second variable.
  value_divide=True,
  ab=None
  ):
  '''
  # input:
  # XX: [batch_size, 1, 1, W, H, C]
  # output:
  # [batch_size,a,b,w,h,c]
  '''
  
  if div_level == 'Auto':
    a = ab[0]
    b = ab[1]
    div_level = (math.log2(a)+math.log2(b))/2
    assert 12==12.0
    # assert div_level==int(div_level)
    div_level = int(div_level)

  CX = XX

  for i in range(div_level):
    cp = CX.shape[3]-1
    cq = CX.shape[4]-1
    assert cp%2==0
    assert cq%2==0

    cp = cp//2
    cq = cq//2

    # y before x, (reverse order of bottom up)

    # y axis: 
    CX = torch.permute(CX, (0, 5, 1, 3, 2, 4))
    if value_divide: # divide value at the middle point to be duplicated.
      # This is buggy! CX[..., cq] /= 2.0 
      CX[..., cq] = CX[..., cq] / 2.0
    CX = CX.unfold(-1, size=cq+1, step=cq) # this will add an dim at last. 
    CX = CX.flatten(start_dim=4, end_dim=5)
    CX = torch.permute(CX, (0, 2, 4, 3, 5, 1)) # get it back.

    # x axis:
    CX = torch.permute(CX, (0, 5, 2, 4, 1, 3))
    if value_divide:
      # This is buggy! CX[..., cp] /= 2.0
      CX[..., cp] = CX[..., cp] / 2.0
    CX = CX.unfold(-1, size=cp+1, step=cp) # this will add an dim at last. 
    CX = CX.flatten(start_dim=4, end_dim=5)
    CX = torch.permute(CX, (0, 4, 2, 5, 3, 1)) # get it back.

  return CX


# sync_across_patches
def beta_sync_across_patches(Beta, wh):
  '''
  Beta and Chi have the same shape. 
  Beta: border elements to be added together.
  Chi:  border elements are duplicated, already added together.
  Beta is used to represent right-hand side of each patch's contribution.
  Chi is used to represent the solution, the border values are duplicated but not divided. 

  Beta.shape: [B,a,b,w*h,1]
  Chi.shape:  [B,a,b,w*h,1]
  '''

  a = Beta.shape[1]
  b = Beta.shape[2]

  Chi = \
    image_flatten(
        top_down(
            bottom_up(
                image_unflatten(
                    Beta, 
                wh[0], wh[1]
                ), 
            value_sum=True 
            ),
        value_divide=False, ab=(a,b) 
        )
    )

  return Chi


chi_from_beta = beta_sync_across_patches


def collapse_subdomains(
  DtN, RHS, 
  Beta_U,
  Gamma_V,
  Omega,
  CRHS, 
  p, q, old_sol, 
  dim=0, 
  pre_sol=None, 
  transposed_solve=False, 
  debug_time=False, 
  invf=None
):
  '''
  history: replace the old functions: 
    tip.recursive
    tip.recursive_back_substitute
  '''

  # note the p,q in this function are w+1,h+1

  sol = Level()

  # input DtN, p, q for rect of p x q

  # DtN: [batch_size, m, n, 2*(p+q), 2*(p+q)]

  augmented_system = (Omega != None) # NOTE: use augmented_system instead of (Omega != None) to decide.
  if augmented_system:
    ranks = Omega.shape[-1]
    assert Omega.shape[-1]==Omega.shape[-2]

  if pre_sol is None:
    sol.name = 'Level_num_fact'
  else:
    sol.name = 'Level_back_sub'
    DtN = pre_sol.DtN
    Beta_U = pre_sol.Beta_U
    Gamma_V = pre_sol.Gamma_V
    Omega = pre_sol.Omega

  size = DtN.shape
  dtype = DtN.dtype
  device = DtN.device

  batch_size = size[0]
  m = size[1]
  n = size[2]

  c = RHS.shape[-1]

  # assert DtN.shape[:-1] == RHS.shape[:-2]

  assert size[-1] == size[-2]
  assert size[-1] == 2 * (p+q)

  if pre_sol is None:
    pass
  else:
    DtN = None
    mDtN = None

    Beta_U = None
    Gamma_V = None
    Omega = None
    mBeta_U = None
    mGamma_V = None
    mOmega = None

  if dim==0:

    assert m%2==0

    if pre_sol is None:
      SS = torch.zeros([batch_size, m//2, n, q-1, q-1], dtype=dtype, device=device)
      RR = torch.zeros([batch_size, m//2, n, 2*(2*p+q), 2*(2*p+q)], dtype=dtype, device=device)

      SR = torch.zeros([batch_size, m//2, n, q-1, 2*(2*p+q)], dtype=dtype, device=device)
      RS = torch.zeros([batch_size, m//2, n, 2*(2*p+q), q-1], dtype=dtype, device=device)

      DtNa = fetch_DtN( DtN[..., 0::2, :, :, :] )
      DtNb = fetch_DtN( DtN[..., 1::2, :, :, :] )

      if augmented_system:

        Beta_U_R =  torch.zeros([batch_size, m//2, n, 2*(2*p+q), ranks], dtype=dtype, device=device)
        Beta_U_S =  torch.zeros([batch_size, m//2, n, q-1, ranks], dtype=dtype, device=device)
        Gamma_V_R = torch.zeros([batch_size, m//2, n, ranks, 2*(2*p+q)], dtype=dtype, device=device)
        Gamma_V_S = torch.zeros([batch_size, m//2, n, ranks, q-1], dtype=dtype, device=device)

        Beta_U_a = Beta_U[..., 0::2, :, :, :]
        Beta_U_b = Beta_U[..., 1::2, :, :, :]
        Gamma_V_a = Gamma_V[..., 0::2, :, :, :]
        Gamma_V_b = Gamma_V[..., 1::2, :, :, :]
        # there is no Omega_a Omega_b since Omega is global. 

    fS = torch.zeros([batch_size, m//2, n, q-1, c], dtype=dtype, device=device)
    fR = torch.zeros([batch_size, m//2, n, 2*(2*p+q), c], dtype=dtype, device=device)

    RHSa = RHS[..., 0::2, :, :, :]
    RHSb = RHS[..., 1::2, :, :, :]

    m0 = 0
    m1 = p
    m2 = 2*p
    m3 = 2*p+q
    m4 = 3*p+q
    m5 = 4*p+q

    # torch only supports contiguous slicing with positive increment. 

    # also copied to topdown_solve below.

    ai = slice(0,p+1) # [0,p+1)=[0,p]
    aj = slice(p+1,p+q) # [p+1,p+q-1]=[p+1,p+q)
    ak = slice(p+q,None) 

    bu = slice(0, 2*p+q+1) # [0,2*p+q] = [2*p+q+1)
    bv = slice(2*p+q+1, None) # bv flip to the ordering of aj or s!

    ri = slice(0,m1+1) # ri and ru overlapped at m1
    ru = slice(m1, m4+1) # ru and rk overlapped at m4
    rk = slice(m4, None)

    # sj is all of the indexing.
    # sv is all of the indexing but in reverse order.   

    if pre_sol is None:
      SS += DtNa[..., aj, aj] + df( DtNb[..., bv, bv], [False, False])
      # ajj, bvv

      RR[..., ri, ri] += DtNa[..., ai, ai] # aii
      RR[..., rk, rk] += DtNa[..., ak, ak] # akk
      RR[..., ri, rk] += DtNa[..., ai, ak] # aik
      RR[..., rk, ri] += DtNa[..., ak, ai] # aki

      RS[..., ri, :]  += DtNa[..., ai, aj] # aij
      RS[..., rk, :]  += DtNa[..., ak, aj] # akj
      SR[..., :, ri]  += DtNa[..., aj, ai] # aji
      SR[..., :, rk]  += DtNa[..., aj, ak] # ajk

      RR[..., ru, ru] += DtNb[..., bu, bu] # buu
      RS[..., ru, :] += df( DtNb[..., bu, bv], [True, False]) # buv
      SR[..., :, ru] += df( DtNb[..., bv, bu], [False, True]) # bvu

      if augmented_system:

        # for beta-type keep it same as how fR, fS are transformed.
        Beta_U_R[..., ri, :] += Beta_U_a[..., ai, :]
        Beta_U_R[..., rk, :] += Beta_U_a[..., ak, :]

        Beta_U_S[..., :, :] += Beta_U_a[..., aj, :]

        Beta_U_R[..., ru, :] += Beta_U_b[..., bu, :]
        Beta_U_S[..., :, :]  += cf( Beta_U_b[..., bv, :], False)

        # for gamma-type slice into the last dim. 
        Gamma_V_R[..., :, ri] += Gamma_V_a[..., :, ai]
        Gamma_V_R[..., :, rk] += Gamma_V_a[..., :, ak]

        Gamma_V_S[..., :, :] += Gamma_V_a[..., :, aj]

        Gamma_V_R[..., :, ru] += Gamma_V_b[..., :, bu]
        Gamma_V_S[..., :, :]  += cf_gamma( Gamma_V_b[..., :, bv], False)

    fR[..., ri, :] += RHSa[..., ai, :]
    fR[..., rk, :] += RHSa[..., ak, :]

    fS[..., :, :] += RHSa[..., aj, :]

    fR[..., ru, :] += RHSb[..., bu, :]
    fS[..., :, :]  += cf( RHSb[..., bv, :], False)

    if pre_sol is None:
      mDtN = torch.zeros([batch_size, m//2, n, 4*p+2*q, 4*p+2*q], dtype=dtype, device=device)

      if augmented_system:
        mBeta_U = torch.zeros([batch_size, m//2, n, 4*p+2*q, ranks], dtype=dtype, device=device)
        mGamma_V = torch.zeros([batch_size, m//2, n, ranks, 4*p+2*q], dtype=dtype, device=device)
        mOmega = torch.zeros([batch_size, m//2, n, ranks, ranks], dtype=dtype, device=device)
      else:
        mBeta_U = None
        mGamma_V = None
        mOmega = None

    # NOTE: probably should remove from here since it's not used anymore. 
    mRHS = torch.zeros(torch.Size([batch_size, m//2, n, 4*p+2*q]) + RHS.shape[-1:], dtype=dtype, device=device) # -1: to make it a list.
    if augmented_system:
      mCRHS = torch.zeros(torch.Size([batch_size, m//2, n, ranks]) + RHS.shape[-1:], dtype=dtype, device=device)
    else:
      mCRHS = None

  else:

    assert dim==1

    assert n%2==0

    if pre_sol is None:

      SS = torch.zeros([batch_size, m, n//2, p-1, p-1], dtype=dtype, device=device)
      RR = torch.zeros([batch_size, m, n//2, 2*(p+2*q), 2*(p+2*q)], dtype=dtype, device=device)

      SR = torch.zeros([batch_size, m, n//2, p-1, 2*(p+2*q)], dtype=dtype, device=device)
      RS = torch.zeros([batch_size, m, n//2, 2*(p+2*q), p-1], dtype=dtype, device=device)

      DtNa = fetch_DtN( DtN[..., :, 0::2, :, :] )
      DtNb = fetch_DtN( DtN[..., :, 1::2, :, :] )

      if augmented_system:

        Beta_U_R =  torch.zeros([batch_size, m, n//2, 2*(p+2*q), ranks], dtype=dtype, device=device)
        Beta_U_S =  torch.zeros([batch_size, m, n//2, p-1, ranks], dtype=dtype, device=device)
        Gamma_V_R = torch.zeros([batch_size, m, n//2, ranks, 2*(p+2*q)], dtype=dtype, device=device)
        Gamma_V_S = torch.zeros([batch_size, m, n//2, ranks, p-1], dtype=dtype, device=device)

        Beta_U_a = Beta_U[..., :, 0::2, :, :]
        Beta_U_b = Beta_U[..., :, 1::2, :, :]
        Gamma_V_a = Gamma_V[..., :, 0::2, :, :]
        Gamma_V_b = Gamma_V[..., :, 1::2, :, :]
        # there is no Omega_a Omega_b since Omega is global. 

    fS = torch.zeros([batch_size, m, n//2, p-1, c], dtype=dtype, device=device)
    fR = torch.zeros([batch_size, m, n//2, 2*(p+2*q), c], dtype=dtype, device=device)

    RHSa = RHS[..., :, 0::2, :, :]
    RHSb = RHS[..., :, 1::2, :, :]

    if debug_time:
      torch.cuda.synchronize()
      start_time = time.time()

    m0 = 0
    m1 = p
    m2 = p+q
    m3 = p+2*q
    m4 = 2*p+2*q
    m5 = 2*p+3*q

    # torch only supports contiguous slicing with positive increment. 

    # also copied to topdown_solve below.

    ai = slice(0,p+q+1) # [0,p+q+1)=[0,p+q]
    aj = slice(p+q+1,2*p+q) # [p+q+1,2*p+q-1]=[p+q+1,2*p+q)
    ak = slice(2*p+q,None) 

    bu = slice(0, 1) # a single point
    bv = slice(1, p) # bv flip to the ordering of aj or s!
    bw = slice(p, None)

    ri = slice(0,m2+1) # ri and rw overlapped at m2
    rw = slice(m2, m5)
    ru = slice(m5, m5+1) # ru and rk overlapped at m5
    rk = slice(m5, None)

    # sj is all of the indexing.
    # sv is all of the indexing but in reverse order.   

    # print(f'shapes: {DtNa[..., aj, aj].shape, SS.shape}')

    if pre_sol is None:

      SS += DtNa[..., aj, aj] + df( DtNb[..., bv, bv], [False, False])
      # ajj, bvv

      RR[..., ri, ri] += DtNa[..., ai, ai] # aii
      RR[..., rk, rk] += DtNa[..., ak, ak] # akk
      RR[..., ri, rk] += DtNa[..., ai, ak] # aik
      RR[..., rk, ri] += DtNa[..., ak, ai] # aki

      RS[..., ri, :]  += DtNa[..., ai, aj] # aij
      RS[..., rk, :]  += DtNa[..., ak, aj] # akj
      SR[..., :, ri]  += DtNa[..., aj, ai] # aji
      SR[..., :, rk]  += DtNa[..., aj, ak] # ajk

      RR[..., ru, ru] += DtNb[..., bu, bu] # buu
      RR[..., rw, rw] += DtNb[..., bw, bw] # bww
      RR[..., ru, rw] += DtNb[..., bu, bw] # buw
      RR[..., rw, ru] += DtNb[..., bw, bu] # bwu
      RS[..., ru, :] += df( DtNb[..., bu, bv], [True, False]) # buv
      SR[..., :, ru] += df( DtNb[..., bv, bu], [False, True]) # bvu
      RS[..., rw, :] += df( DtNb[..., bw, bv], [True, False]) # bwv
      SR[..., :, rw] += df( DtNb[..., bv, bw], [False, True]) # bvw

      if augmented_system:

        # for beta-type keep it same as how fR, fS are transformed.
        Beta_U_R[..., ri, :] += Beta_U_a[..., ai, :]
        Beta_U_R[..., rk, :] += Beta_U_a[..., ak, :]

        Beta_U_S[..., :, :] += Beta_U_a[..., aj, :]

        Beta_U_R[..., ru, :] += Beta_U_b[..., bu, :]
        Beta_U_R[..., rw, :] += Beta_U_b[..., bw, :]
        Beta_U_S[..., :, :]  += cf( Beta_U_b[..., bv, :], False)

        # for gamma-type slice into the last dim. 
        
        Gamma_V_R[..., :, ri] += Gamma_V_a[..., :, ai]
        Gamma_V_R[..., :, rk] += Gamma_V_a[..., :, ak]

        Gamma_V_S[..., :, :] += Gamma_V_a[..., :, aj]

        Gamma_V_R[..., :, ru] += Gamma_V_b[..., :, bu]
        Gamma_V_R[..., :, rw] += Gamma_V_b[..., :, bw]
        Gamma_V_S[..., :, :]  += cf_gamma( Gamma_V_b[..., :, bv], False)
        
    fR[..., ri, :] += RHSa[..., ai, :]
    fR[..., rk, :] += RHSa[..., ak, :]

    fS[..., :, :] += RHSa[..., aj, :]

    fR[..., ru, :] += RHSb[..., bu, :]
    fR[..., rw, :] += RHSb[..., bw, :]
    fS[..., :, :]  += cf( RHSb[..., bv, :], False)

    if debug_time:
      torch.cuda.synchronize()
      print(f'matrix slicer time: {(time.time()-start_time)*1000} ms')

    # NOTE: consider get rid of the initialization
    if pre_sol is None:
      mDtN = torch.zeros([batch_size, m, n//2, 2*p+4*q, 2*p+4*q], dtype=dtype, device=device)

      if augmented_system:
        mBeta_U = torch.zeros([batch_size, m, n//2, 2*p+4*q, ranks], dtype=dtype, device=device)
        mGamma_V = torch.zeros([batch_size, m, n//2, ranks, 2*p+4*q], dtype=dtype, device=device)
        mOmega = torch.zeros([batch_size, m, n//2, ranks, ranks], dtype=dtype, device=device)
      else:
        mBeta_U = None
        mGamma_V = None
        mOmega = None

    mRHS = torch.zeros(torch.Size([batch_size, m, n//2, 2*p+4*q]) + RHS.shape[-1:], dtype=dtype, device=device) # -1: to make it a list.
    if augmented_system:
      mCRHS = torch.zeros(torch.Size([batch_size, m, n//2, ranks]) + RHS.shape[-1:], dtype=dtype, device=device)
    else:
      mCRHS = None

  if debug_time:
    torch.cuda.synchronize()
    start_time = time.time()
  
  # X: RR, Y: RS, Z: SR, W: SS
  # y: fR, w: fS
  # Ux: Beta_U_R
  # Uz: Beta_U_S
  # Vx: Gamma_V_R
  # Vz: Gamma_V_S

  if pre_sol is None:

    if True:
      if invf is None:
        invf = lambda xx : torch.linalg.inv_ex(xx)[0]
      size = SS.shape 
      assert len(size)==5
      # batch_eye = torch.eye(size[-1])[None, None, None, ...]
      # invSS, info = torch.linalg.solve_ex(SS, batch_eye) 
      # invSS, info = torch.linalg.inv_ex(SS) 
      # invSS = torch.linalg.inv(SS) 
      invSS = invf(SS)

      invSSmSR = invSS @ SR # W^{-1} Z
      mDtN += RR - RS @ invSSmSR # RR - RS @ invSS @ SR

      if augmented_system:
        mBeta_U = Beta_U_R - RS @ invSS @ Beta_U_S
        mGamma_V = Gamma_V_R - Gamma_V_S @ invSSmSR
        mOmega = Omega - gather_contributions(Gamma_V_S @ invSS @ Beta_U_S)
      else:
        mBeta_U = None
        mGamma_V = None
        mOmega = None

      def update_mRHS(ffR, ffS, cRHS, transposed_solve):
        if not transposed_solve:
          # y − Y W^−1 w
          mmRHS = ffR - RS @ (invSS @ ffS)
          mmCRHS = cRHS - gather_contributions(Gamma_V_S @ (invSS @ ffS)) if augmented_system else None
          return mmRHS, mmCRHS
        else:
          # (y⊺ − w⊺ W^−1 Z ) ⊺
          mmRHS = ffR - hermitian( (hermitian(ffS) @ invSS ) @ SR )
          mmCRHS = cRHS - gather_contributions( hermitian( (hermitian(ffS) @ invSS ) @ Beta_U_S ) ) if augmented_system else None
          return mmRHS, mmCRHS

    else:
      assert False # not implemented
      assert augmented_system==False

      if False:
        # mDtN = RR + RS * (SS\SR)
        LL, info = torch.linalg.cholesky_ex(SS, upper=True)
        mDtN += RR - RS @ torch.cholesky_solve(SR, LL)
        mRHS += fR - RS @ torch.cholesky_solve(fS, LL)

      else:
        LL, info = torch.linalg.cholesky_ex(SS, upper=True)
        size = SS.shape 
        assert len(size)==5
        batch_eye = torch.eye(size[-1], device=device)[None, None, None, ...]
        invSS = torch.cholesky_solve( batch_eye , LL) 
        invSSmSR = invSS @ SR
        mDtN += RR - RS @ invSSmSR # RR - RS @ invSS @ SR
        mRHS += fR - RS @ (invSS @ fS)

    # has setup the function update_mRHS

  else:

    update_mRHS = pre_sol.update_mRHS

  mRHS, mCRHS = update_mRHS(fR, fS, CRHS, transposed_solve=transposed_solve) # CRHS does not need to be splitted.
  sol.update_mRHS = update_mRHS

  if debug_time:
    torch.cuda.synchronize()
    print(f'linear solve time: {(time.time()-start_time)*1000} ms')

  ############# back-fill: assemble the callable topdown_solve_Dirichlet #############

  sol.DtN = DtN
  sol.Beta_U = Beta_U
  sol.Gamma_V = Gamma_V
  sol.Omega = Omega

  sol.batch_size = batch_size
  sol.p = p
  sol.q = q
  sol.m = m
  sol.n = n

  if pre_sol is None:
    sol.invSS = invSS
    sol.invSSmSR = invSSmSR
    sol.RS = RS
    if augmented_system:
      sol.Beta_U_S = Beta_U_S
    def update_uS(ffS, uuR, lam, transposed_solve):
      if not transposed_solve:
        # z = W^−1 [w − Z x - Uz λ]
        tmp = ffS
        if augmented_system:
          tmp = tmp - Beta_U_S @ lam
        return invSS @ tmp - invSSmSR @ uuR
      else:
        # z = ( ( w⊺ - x⊺ Y - λ⊺ Vz ) W^−1 ) ⊺
        tmp = hermitian(ffS)
        if augmented_system:
          tmp = tmp - hermitian(lam) @ Gamma_V_S
        return hermitian( ( tmp - hermitian(uuR) @ RS ) @ invSS )
  else:
    sol.invSS = pre_sol.invSS
    sol.invSSmSR = pre_sol.invSSmSR
    sol.RS = pre_sol.RS
    if augmented_system:
      sol.Beta_U_S = pre_sol.Beta_U_S
    update_uS = pre_sol.update_uS

  # update_uS is the back fill function.
  sol.update_uS = update_uS

  if dim==0:
    def topdown_solve_Dirichlet(BC, lam, transposed_solve):
      c = BC.shape[-1]
      nBC = torch.zeros([batch_size, m, n, 2*(p+q), c], dtype=dtype, device=device)
      # nBCa = nBC[..., 0::2, :, :, :]
      # nBCb = nBC[..., 1::2, :, :, :]
      uR = BC
      uS = update_uS(fS, uR, lam=lam, transposed_solve=transposed_solve)
      nBC[..., 0::2, :, ai, :] = uR[..., ri, :] # nBCa[..., ai, :]
      nBC[..., 0::2, :, ak, :] = uR[..., rk, :] # nBCa[..., ak, :]
      nBC[..., 0::2, :, aj, :] = uS[..., :, :]  # nBCa[..., aj, :]
      nBC[..., 1::2, :, bu, :] = uR[..., ru, :] # nBCb[..., bu, :]
      nBC[..., 1::2, :, bv, :] = cf(uS[..., :, :], False) # nBCb[..., bv, :]
      return nBC
  else:
    def topdown_solve_Dirichlet(BC, lam, transposed_solve):
      c = BC.shape[-1]
      nBC = torch.zeros([batch_size, m, n, 2*(p+q), c], dtype=dtype, device=device)
      # nBCa = nBC[..., :, 0::2, :, :]
      # nBCb = nBC[..., :, 1::2, :, :]
      uR = BC
      uS = update_uS(fS, uR, lam=lam, transposed_solve=transposed_solve)
      nBC[..., :, 0::2, ai, :] = uR[..., ri, :] # nBCa[..., ai, :]
      nBC[..., :, 0::2, ak, :] = uR[..., rk, :] # nBCa[..., ak, :]
      nBC[..., :, 0::2, aj, :] = uS[..., :, :]  # nBCa[..., aj, :]
      nBC[..., :, 1::2, bu, :] = uR[..., ru, :] # nBCb[..., bu, :]
      nBC[..., :, 1::2, bw, :] = uR[..., rw, :] # nBCb[..., bw, :]
      nBC[..., :, 1::2, bv, :] = cf(uS[..., :, :], False) # nBCb[..., bv, :]
      return nBC
    
  sol.topdown_solve_Dirichlet = topdown_solve_Dirichlet    

  '''
  level.mDtN
  level.mRHS
  level.mp = 2*p
  level.mq = q

  return level
  '''
  # output merged DtN
  # print((p, q), (m, n))

  if dim==0:
    return mDtN, mRHS, mBeta_U, mGamma_V, mOmega, mCRHS, 2*p, q, sol
  else:
    return mDtN, mRHS, mBeta_U, mGamma_V, mOmega, mCRHS, p, 2*q, sol


def topdown_solve(BC, sol):

  c = BC.shape[-1]

  batch_size = sol.batch_size
  p = sol.p
  q = sol.q
  m = sol.m
  n = sol.n

  dtype = BC.dtype

  nBC = torch.zeros([batch_size, m, n, 2*(p+q), c], dtype)

  # nBCa = nBC[..., 0::2, :, :, :]
  # nBCb = nBC[..., 1::2, :, :, :]

  # copied from above.

  ai = slice(0,p+1) # [0,p+1)=[0,p]
  aj = slice(p+1,p+q) # [p+1,p+q-1]=[p+1,p+q)
  ak = slice(p+q,None) 

  bu = slice(0, 2*p+q+1) # [0,2*p+q] = [2*p+q+1)
  bv = slice(2*p+q+1, None) # bv flip to the ordering of aj or s!

  ri = slice(0,m1+1) # ri and ru overlapped at m1
  ru = slice(m1, m4+1) # ru and rk overlapped at m4
  rk = slice(m4, None)

  # sj is all of the indexing.
  # sv is all of the indexing but in reverse order.   

  uR = BC
  uS = sol.invSS @ sol.mRHS - invSSmSR @ uR

  nBC[..., 0::2, :, ai, :] = uR[..., ri, :] # nBCa[..., ai, :]
  nBC[..., 0::2, :, ak, :] = uR[..., rk, :] # nBCa[..., ak, :]
  nBC[..., 0::2, :, aj, :] = uS[..., :, :]  # nBCa[..., aj, :]

  nBC[..., 1::2, :, bu, :] = uR[..., ru, :] # nBCb[..., bu, :]
  nBC[..., 1::2, :, bv, :] = cf(uS[..., :, :], False) # nBCb[..., bv, :]

  return nBC


def midpoint_reflective_reduce(m, n, ps, qs):

  assert m%2==0
  assert n%2==0
  mph = (m*ps)//2
  nqh = (n*qs)//2

  FR = torch.zeros([4*(mph+nqh), 2*(mph+nqh)+1])

  ua = 0
  ub = mph
  uc = 2*mph
  ud = 2*mph + nqh
  ue = 2*mph + 2*nqh
  uf = 3*mph + 2*nqh
  ug = 4*mph + 2*nqh
  uh = 4*mph + 3*nqh
  uab = slice(ua+1, ub)
  ubc = slice(ub+1, uc)
  ucd = slice(uc+1, ud)
  ude = slice(ud+1, ue)
  uef = slice(ue+1, uf)
  ufg = slice(uf+1, ug)
  ugh = slice(ug+1, uh)
  uha = slice(uh+1, None)

  # u = slice(u+1, u)

  va = 0 # a is put first in v
  vb = mph
  vd = mph + nqh
  vf = 2*mph + nqh
  vh = 2*mph + 2*nqh
  vab = slice(va+1, vb)
  vcd = slice(vb+1, vd)
  vef = slice(vd+1, vf)
  vgh = slice(vf+1, vh)

  FR[ua, va] = 1
  FR[uc, va] = 1
  FR[ue, va] = 1
  FR[ug, va] = 1

  FR[ub, vb] = 1
  FR[ud, vd] = 1
  FR[uf, vf] = 1
  FR[uh, vh] = 1

  def slen(sslice):
    return sslice.stop - sslice.start

  FR[uab, vab] = torch.eye(slen(vab))
  FR[ucd, vcd] = torch.eye(slen(vcd))
  FR[uef, vef] = torch.eye(slen(vef))
  FR[ugh, vgh] = torch.eye(slen(vgh))

  FR[ubc, vab] = torch.flip(torch.eye(slen(vab)), dims=[0])
  FR[ude, vcd] = torch.flip(torch.eye(slen(vcd)), dims=[0])
  FR[ufg, vef] = torch.flip(torch.eye(slen(vef)), dims=[0])
  FR[uha, vgh] = torch.flip(torch.eye(slen(vgh)), dims=[0])

  F_reduce = FR[None,...]

  return F_reduce


def apply_all_schur_steps(
  DtN, RHS, BC, 
  Beta_U,
  Gamma_V,
  Omega,
  CRHS,
  m, n, ps, qs, 
  transposed_solve=False, 
  debug=False, 
  layers_pre=None, 
  lastInvLHS_reuse=None, 
  invf=inv_default
):
  '''
  history: replace the old functions: 
    tip.routine_ps_condense
    tip.routine_ps_back_substitute
  '''

  augmented_system = (Omega != None)
  if augmented_system:
    pass
  else:
    lam = None
  
  # invf = lambda xx : torch.linalg.inv_ex(xx)[0]
  # invf = lambda xx : torch.linalg.inv(xx)
  
  layers = [(DtN, RHS, Beta_U, Gamma_V, Omega, CRHS,  ps, qs, [])]

  N_iter = math.floor(math.log2(m)+math.log2(n))
  
  for i in range(N_iter): 

    dim = 0 if i%2==0 else 1
    # iter_fun = tip.recursive if i%2==0 else tip.recursive_last

    if debug:
      import time
      start_time = time.time()

    # print(f'{list(layers[i][0].shape[1:3])+ [layers[i][2], layers[i][3]]},')

    if layers_pre is None:
      # DtN_new, RHS_new, ps_new, qs_new, sol = recursive(*layers[i], dim=dim)
      DtN_new, RHS_new, Beta_U_new, Gamma_V_new, Omega_new, CRHS_new, ps_new, qs_new, sol = \
        collapse_subdomains(*layers[i], dim=dim, \
          pre_sol=None, transposed_solve=transposed_solve)
    else:
      DtN_new, RHS_new, Beta_U_new, Gamma_V_new, Omega_new, CRHS_new, ps_new, qs_new, sol = \
        collapse_subdomains(*layers[i], dim=dim, \
          pre_sol=layers_pre[i+1][-1], transposed_solve=transposed_solve) # sol is the last one

    if debug:
      print(f'routine_ps_condense: {(ps_new,qs_new)} time[{i}]: {(time.time()-start_time)*1000} ms')

    sol.name += str(i)

    layers.append([DtN_new, RHS_new, Beta_U_new, Gamma_V_new, Omega_new, CRHS_new, ps_new, qs_new, sol])

  lastInvLHS = None

  if BC is None:

    assert False # use instead 'Neumann'

  if BC=='Neumann-scatter':

    assert not augmented_system # not implemented
    lam = None

    if layers_pre is None:
      lhs = layers[-1][0]
    else:
      lhs = layers_pre[-1][0]
    rhs = layers[-1][1]
    assert ps==qs # otherwise not implemented
    ss = slice(0,None,ps)
    # lastInvLHS = 'not implemented'
    BC = torch.zeros(lhs.shape[:-1]+rhs.shape[-1:], dtype=lhs.dtype, device=lhs.device)
    if layers_pre is None:
      lastInvLHS = invf(lhs[...,ss,ss])
    else:
      lastInvLHS = lastInvLHS_reuse
    if not transposed_solve:
      BC[...,ss,:] = lastInvLHS @ rhs[...,ss,:]
    else:
      BC[...,ss,:] = hermitian( hermitian(rhs[...,ss,:]) @ lastInvLHS )
    
  elif BC=='Neumann' or BC=='Neumann-full':

    #print(f'routine_ps_condense(): No BC provided.')

    dd = layers[-1][1]

    if augmented_system:
      invOmega = torch.linalg.inv(Omega_new)
      prod_reused = Beta_U_new @ invOmega

    # setup lastInvLHS
    if layers_pre is None:
      lhs = layers[-1][0]
      if augmented_system:
        lhs = lhs - prod_reused @ Gamma_V_new
      # lastInvLHS = tsl.recur_inv(layers[-1][0], cutoff=1000)
      # lastInvLHS, info = torch.linalg.inv_ex(layers[-1][0])
      lastInvLHS = invf(lhs)
    else:
      lastInvLHS = lastInvLHS_reuse

    if not transposed_solve:
      # rhs-update
      rhs = dd
      if augmented_system:
        rhs = rhs - prod_reused @ CRHS_new ## i.e.: Beta_U_new @ invOmega @ CRHS_new
      BC = lastInvLHS @ rhs
      # back-fill:
      lam = invOmega @ (CRHS_new - Gamma_V_new @ BC) if augmented_system else None 
    else:
      herm_rhs = hermitian(dd)
      if augmented_system:
        herm_rhs = herm_rhs - ( hermitian(CRHS_new) @ invOmega ) @ Gamma_V_new 
      BC = hermitian( herm_rhs @ lastInvLHS )
      lam = hermitian( ( hermitian(CRHS_new) - hermitian(BC) @ Beta_U_new ) @ invOmega ) if augmented_system else None

  elif BC=='midpoint-reflective':

    assert not augmented_system
    lam = None
    assert transposed_solve==False
    F_reduce = midpoint_reflective_reduce(m, n, ps, qs)
    if layers_pre is None:
      lhs = layers[-1][0]
    else:
      lhs = layers_pre[-1][0]
    rhs = layers[-1][1]
    assert torch.is_complex(F_reduce)==False
    lhs = F_reduce.transpose(-1,-2) @ lhs @ F_reduce
    rhs = F_reduce.transpose(-1,-2) @ rhs
    # lastInvLHS = 'not implemented'
    if layers_pre is None:
      lastInvLHS = invf(lhs)
    else:
      lastInvLHS = lastInvLHS_reuse
    if not transposed_solve:
      BC = lastInvLHS @ rhs
    else:
      BC = hermitian( hermitian(rhs) @ lastInvLHS )
    BC = F_reduce @ BC
  
  else: # Dirichlet boundary condition

    assert torch.is_tensor(BC)
    assert not augmented_system
    lam = None

  nBC = BC
  for i in range(N_iter):

    sol = layers[-(i+1)][-1]
    nBC = sol.topdown_solve_Dirichlet(nBC, lam=lam, transposed_solve=transposed_solve)

  #print(f'time: {(time.time()-start_time)*1000} ms')

  if layers_pre is None:
    size = layers[-1][0].shape
    assert size[1]==1 or size[2]==1

  return nBC, layers, lastInvLHS, lam


def simple_WHC(ab_wh, dtype=None, fun=torch.ones):
  ''' output is of shape WHC, not BWHC'''

  a,b,w,h,W,H = unpack_dims(ab_wh)

  if dtype is None:
    dtype = torch.get_default_dtype()

  if type(fun)==str:
    assert fun=='xy_coordinates'
    image_whc = meshgrid_coordinates(WH=[W,H], dtype=dtype)
  else:
    image_whc = fun([W,H,1], dtype=dtype)
  
  return image_whc


def BWHC_of_shape(
  ab_wh, 
  batch_size=1,
  columns=1, 
  fun=torch.zeros,
  dtype=None,
):

  a,b,w,h,W,H = unpack_dims(ab_wh)

  if dtype is None:
    dtype = torch.get_default_dtype()

  BWHC = fun([batch_size, W, H, columns], dtype=dtype)

  return BWHC


def beta_of_shape(
  ab_wh, 
  batch_size=1,
  columns=1, 
  fun=torch.zeros,
  dtype=None,
):

  BWHC = BWHC_of_shape(
    ab_wh=ab_wh, 
    batch_size=batch_size,
    columns=columns, 
    fun=fun, 
    dtype=dtype,
  )

  print('beta_of_shape:', BWHC.shape, ab_wh)

  beta = beta_from_BWHC(BWHC, ab_wh)

  return beta


def chi_of_shape(
  ab_wh, 
  batch_size=1,
  columns=1, 
  fun=torch.zeros,
  dtype=None,
):

  a,b,w,h,W,H = unpack_dims(ab_wh)

  if dtype is None:
    dtype = torch.get_default_dtype()

  chi = fun([batch_size, a, b, w*h, columns], dtype=dtype)

  return chi


def zero_beta_for(
  Alpha, 
  columns=1, 
):
  '''this is only for zero right-hand side, torch.zeros, other wise some across-patch division is needed'''
  return torch.zeros(Alpha.shape[:-1]+torch.Size([columns]), dtype=Alpha.dtype, device=Alpha.device)


def default_chi_for(
  Alpha, 
  columns=1, 
  fun=torch.zeros
):
  return fun(Alpha.shape[:-1]+torch.Size([columns]), dtype=Alpha.dtype, device=Alpha.device)




class Helpers():
  '''work in progress: do not use this class'''
  name = 'Helpers on how to use solution of Schwarz_Schur_involution'
  flatten = image_flatten
  unflatten = image_unflatten


def autograd_gradients(
  alpha,
  beta,
  BC, 
  x_BWHC, 
  lam, 
  x_BWHC_grad=None,
  betaU=None,
  gammaV=None,
  omega=None, 
  sigma=None, 
  lam_grad=None,
  ):

    augmented_system = (omega is not None)

    grads = {'alpha': None, 'beta': None, 'BC': None,  
      'betaU': None, 'gammaV': None, 'omega': None, 'sigma': None}

    diff_inputs = [alpha, beta] # inputs to torch.autograd.grad cannot be None or str
    names = ['alpha', 'beta'] 

    if augmented_system:
        diff_inputs += [betaU, gammaV, omega, sigma]
        names += ['betaU', 'gammaV', 'omega', 'sigma']

    if torch.is_tensor(BC):
        diff_inputs += [BC]
        names += ['BC']
    
    for t in diff_inputs:
        assert t.requires_grad==True

    # Construct a scalar-equivalent loss by seeding each output's cotangent
    # with ones, i.e. loss = sum(x_BWHC) + sum(lam). grad_outputs are the
    # upstream gradients dLoss/dOutput, which are all-ones for this loss.
    outputs = [x_BWHC]
    grad_outputs = [x_BWHC_grad if torch.is_tensor(x_BWHC_grad) else torch.ones_like(x_BWHC, dtype=x_BWHC.dtype) ]
    if lam is not None:
        outputs.append(lam)
        grad_outputs.append(lam_grad if torch.is_tensor(lam_grad) else torch.ones_like(lam, dtype=lam.dtype))

    gradients = torch.autograd.grad(
        outputs=tuple(outputs),
        inputs=tuple(diff_inputs),
        grad_outputs=tuple(grad_outputs),
        allow_unused=True,
    )

    for name, g in zip(
        names,
        gradients,
    ):
        #print('grad', name, None if g is None else tuple(g.shape))
        grads[name] = g

    return grads


def compare_gradients_with_scipy(
    grad_torch, 
    grad_scipy, 
    ab_wh, 
    transposed_solve, 
    alpha_mask=None, # per patch, masking alpha before comparison.
    ):
    """
    e.g.: 
    
    not good use, since alpha can happen to be zero:
        alpha_mask = alpha.detach().abs() > 0
    """

    if isinstance(alpha_mask, str):
        if alpha_mask=='9pt':
            alpha_mask = alpha_uniform_laplacian(ab_wh).detach().abs() > 0
        else:
            raise ValueError(f"Unknown alpha_mask of type {alpha_mask}")

    augmented_system = (grad_torch['omega'] is not None)

    a_,b_,w_,h_,W_,H_ = unpack_dims(ab_wh)
    
    errors = dict()

    # --- convert each scipy (global) gradient to the patchwise representation ---

    # alpha lives on the patchwise sparse matrix; convert grad.A back with
    # value_divide=False so the shared boundary DOFs are *gathered* (adjoint of
    # the scatter-add forward map) rather than averaged by patch multiplicity.
    # NOTE: this only agrees with grad_torch on A_scipy's sparsity pattern (the
    # 9pt stencil). grad_torch also has *real* nonzeros off that pattern (an
    # unconstrained w*h x w*h block), which scipy masks to zero; so restrict the
    # comparison to the stencil. In transposed_solve mode the sinv2d_aug backward
    # emits the alpha-block gradient without the transpose that d/dA of A^T\b
    # induces, so transpose the scipy result to match.
    galpha_scipy = alpha_from_scipy_sparse(grad_scipy.A, ab_wh, value_divide=False)
    if transposed_solve:
        galpha_scipy = galpha_scipy.transpose(-1, -2)
      
    # beta / betaU / gammaV are RHS-type: gather the global rows into patches
    # using the hierarchical mesh flat indices FD = FX + FY*W.
    FXY = mesh_hier(ab_wh, flattened=True)          # [a,b,w*h,2]
    FD = (FXY[..., 0] + FXY[..., 1] * W_).long().reshape(-1)  # [a*b*w*h]

    def gather_beta(g_np):
        # g_np: [N, nch] -> [1, a, b, w*h, nch]
        g = torch.tensor(g_np, dtype=torch.get_default_dtype())
        return g[FD].reshape(a_, b_, w_ * h_, -1)[None]

    gbeta_scipy = gather_beta(grad_scipy.b)

    def rel_err(g_torch, g_scipy, mask=None):
        if g_torch is None or g_scipy is None:
            return None
        if mask is not None:
            g_torch = g_torch[mask]
            g_scipy = g_scipy[mask]
        gt = g_torch.detach().cpu().numpy() if torch.is_tensor(g_torch) else g_torch
        gs = g_scipy.detach().cpu().numpy() if torch.is_tensor(g_scipy) else g_scipy
        return float(np.abs(gt - gs).sum() / (np.abs(gt).sum() + 1e-30))

    # print('grad rel-err (torch vs scipy):')
    errors['alpha'] = rel_err(grad_torch['alpha'], galpha_scipy, alpha_mask) # (on-stencil) error
    errors['alpha_on_mask'] = errors['alpha']
    errors['beta'] = rel_err(grad_torch['beta'],  gbeta_scipy)

    if augmented_system:
        # betaU/omega are RHS/dense-type. Under a transposed solve the KKT
        # block operator is transposed, which swaps the U<->V roles and
        # transposes Omega in the gradient, so pick the matching scipy block.
        gU_np = grad_scipy.V.T if transposed_solve else grad_scipy.U
        gV_np = grad_scipy.U.T if transposed_solve else grad_scipy.V
        gOmega_np = grad_scipy.Omega.T if transposed_solve else grad_scipy.Omega

        gbetaU_scipy = gather_beta(gU_np)
        ggammaV_scipy = gather_beta(gV_np.T).transpose(-1, -2)
        gomega_scipy = torch.tensor(gOmega_np,
                                    dtype=torch.get_default_dtype())[None, ...]
        gsigma_scipy = torch.tensor(grad_scipy.sigma,
                                    dtype=torch.get_default_dtype())[None, ...]

        errors['betaU'] = rel_err(grad_torch['betaU'],  gbetaU_scipy)
        errors['gammaV'] = rel_err(grad_torch['gammaV'], ggammaV_scipy)
        errors['omega'] = rel_err(grad_torch['omega'],  gomega_scipy)
        errors['sigma'] = rel_err(grad_torch['sigma'],  gsigma_scipy)

    return errors


def error_measure_using_scipy(
  A, 
  b, 
  x, 
  U=None, 
  V=None, 
  Omega=None, 
  sigma=None, 
  lam=None, # only for augmented systems
  transposed_solve=False,
  ):

  from scipy import sparse

  assert len(b.shape)==len(x.shape)

  augmented_system = not (Omega is None)

  if augmented_system:

    lhs = sparse.block_array([[A, U], [V, Omega]])
    if transposed_solve:
      lhs = lhs.T
    xak = np.block([[x], [lam]])
    rhs = np.block([[b], [sigma]])
    tol = np.linalg.norm(rhs - lhs @ xak) / np.linalg.norm(rhs)

  else:
    lhs = A.T if transposed_solve else A 
    tol = np.linalg.norm(b - lhs @ x) / np.linalg.norm(b)

  return tol



# using name: planning to change API to Alpha, Beta, BC, xxx 
def Schwarz_Schur_involution(
  Alpha, 
  Beta, 
  BC, 
  wh, 
  Beta_U=None,
  Gamma_V=None,
  Omega=None,
  sigma=None,
  debug=False, 
  prefact_sol=None, 
  transposed_solve=False, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
):

  """
  Inputs
  ------
  Alpha : {torch.tensor, None}
    ``Alpha`` should be ``None`` if given ``prefact_sol``
  input: 
    Alpha: [batch_size, a, b, (w*h), (w*h)]     or None 
     Beta: [batch_size, a, b, (w*h), channels]
    
    Beta_U:  [batch_size, a, b, (w*h), ranks]
    Gamma_V: [batch_size, a, b, ranks, (w*h)]
    Omega: [batch_size, ranks, ranks]
    sigma: [batch_size, ranks, channels]

  History
  -------
  replace the old functions: 
  >>> import xxx as tip
  >>> tip.condense_general_solve
  >>> tip.condense_back_substitute
  """


  ############# check input sizes #############

  if Beta is None:
    assert torch.is_tensor(BC)
    Beta = zero_beta_for(Alpha, columns=BC.shape[-1])

  if prefact_sol is None:
    p = wh[0]
    q = wh[1]
    batch_size = Alpha.shape[0]
    a = Alpha.shape[1]
    b = Alpha.shape[2]
    assert len(Alpha.shape)==5
    assert Alpha.shape[3]==(p*q)
    assert Alpha.shape[4]==(p*q)
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

  assert len(Beta.shape)==5
  assert Beta.shape[0]==batch_size 
  assert Beta.shape[1]==a
  assert Beta.shape[2]==b
  assert Beta.shape[3]==(p*q)

  channels = Beta.shape[4]

  augmented_system = (Omega != None)

  if augmented_system:
    ranks = Omega.shape[1]
    assert Omega.shape == (batch_size, ranks, ranks)
    assert Beta_U.shape == (batch_size, a, b, (p*q), ranks)
    assert Gamma_V.shape == (batch_size, a, b, ranks, (p*q))
    assert sigma.shape == (batch_size, ranks, channels)
  else:
    ranks = 0
    assert Beta_U is None
    assert Gamma_V is None
    assert sigma is None

  ##########################

  if (not bfs_order):
    if Alpha!=None:
      Alpha = convert_to_boundary_first_ordering(Alpha, wh=wh)
    Beta    = convert_to_boundary_first_ordering_rhs(Beta, wh=wh, beta_type=True)
    if augmented_system:
      Beta_U  = convert_to_boundary_first_ordering_rhs(Beta_U, wh=wh, beta_type=True)
      Gamma_V = convert_to_boundary_first_ordering_rhs(Gamma_V, wh=wh, beta_type=False) # Gamma type.

  # if BC is None, it will use Neumann BC. 
  RHS = Beta

  if BC is None:
    print('warning: use Neumann for BC==None')
    BC = 'Neumann'

  if BC=='Neumann':
    BC = 'Neumann-scatter'

  # RHS test None
  
  if debug:
    start_time = time.time()

  new_version = True # merge_Bab = False

  """
  ## The system 
  (Alpha, Beta=RHS, Beta_U, Gamma_V, Omega, sigma, BC)
    first becomes
  (DtN, nRHS, nBeta_U, nGamma_V, nOmega, nCRHS, BC)
    and then becomes
  (sDtN, sRHS, sBeta_U, sGamma_V, sOmega, sCRHS, sBC)
    and then recursive Schur merge steps. 
  """

  ###### [1]: Schwarz step: num_fact ######
  if prefact_sol is None:

    if new_version:

      DtN, get_nRHS_step1, batch_Dirichlet_solver, nBeta_U, nGamma_V, nOmega = \
          batch_Dirichlet_solve_pre(dim_pq=(p,q), dLA=Alpha, Beta_U=Beta_U, Gamma_V=Gamma_V, Omega=Omega)  

      def get_nRHS(RHS, cRHS, transposed_solve):
        nRHS, nCRHS = get_nRHS_step1(RHS, cRHS, transposed_solve=transposed_solve)
        return nRHS, nCRHS

    else: # the old implementation with limited capacity. 

      assert not augmented_system
      assert transposed_solve==False # might work, haven't debugged.

      # the flattening ordering does not matter. It is only used here. 
      Alpha = torch.reshape(Alpha, torch.Size([batch_size*a*b]) + Alpha.shape[-2:])
      DtN, get_nRHS_step1, batch_Dirichlet_solver, _, _, _ = \
          batch_Dirichlet_solve_pre(dim_pq=(p,q), dLA=Alpha, Beta_U=None, Gamma_V=None, Omega=None)  
      
      DtN = torch.reshape(DtN, torch.Size([batch_size, a, b]) + DtN.shape[-2:])

      def get_nRHS(RHS, cRHS):
        assert cRHS is None
        tmp_RHS = torch.reshape(RHS, [batch_size*a*b] + list(RHS.shape[-2:]))
        nRHS, _ = get_nRHS_step1(tmp_RHS, cRHS, transposed_solve=transposed_solve)
        nRHS = torch.reshape(nRHS, torch.Size([batch_size, a, b]) + nRHS.shape[-2:])
        return nRHS, None

    class Solution:
      name = 'Schwarz_Schur_involution'

  else:

    class Solution:
      name = 'Schwarz_Schur_involution_back_sub'

    get_nRHS = prefact_sol.get_nRHS
    batch_Dirichlet_solver = prefact_sol.batch_Dirichlet_solver

    DtN = prefact_sol.DtN
    nBeta_U  = prefact_sol.nBeta_U
    nGamma_V = prefact_sol.nGamma_V
    nOmega   = prefact_sol.nOmega

  if debug:
    print(f'DtN solve time: {(time.time()-start_time)*1000} ms')

  sol = Solution()

  nRHS, nCRHS = get_nRHS(RHS, sigma, transposed_solve)
  if not (BC=='Neumann-full' or BC=='Neumann-scatter' or BC=='midpoint-reflective'):
    assert nRHS.shape == (DtN.shape[:-1] + BC.shape[-1:])

  ###### [1.1]: Schwarz step - scatter eliminate: num_fact ######

  if BC=='Neumann-scatter':
    assert not augmented_system # not implemented.
    if prefact_sol is None:
      sBeta_U, sGamma_V, sOmega = None, None, None
      sCRHS = None
      sDtN, sRHS, nBC_back_fill, sol_scatter = \
        scatter_boundary_eliminate_pre((p,q), DtN, nRHS, transposed_solve=transposed_solve)
    else:
      sDtN = prefact_sol.sDtN
      sBeta_U, sGamma_V, sOmega = prefact_sol.sBeta_U, prefact_sol.sGamma_V, prefact_sol.sOmega
      sCRHS = None
      _, sRHS, nBC_back_fill, sol_scatter = \
        scatter_boundary_eliminate_pre((p,q), DtN, nRHS, pre_sol=prefact_sol.sol_scatter, transposed_solve=transposed_solve)      
  else:
    sDtN, sRHS, sCRHS = DtN, nRHS, nCRHS
    sBeta_U, sGamma_V, sOmega = nBeta_U, nGamma_V, nOmega
    sol_scatter = None
  
  ###### [2 & -2]: Schur steps: num_fact and back_sub ######
  
  if prefact_sol is None:
    # print(f'shapes {sDtN.shape}, {sRHS.shape}')
    # sBC, layers, lastInvLHS = routine_ps_condense(sDtN, sRHS, BC, a, b, p-1, q-1, debug)
    sBC, layers, lastInvLHS, lam = \
      apply_all_schur_steps(sDtN, sRHS, BC, sBeta_U, sGamma_V, sOmega, sCRHS, a, b, p-1, q-1, 
        debug=debug, transposed_solve=transposed_solve)
  else:
    sBC, layers, lastInvLHS, lam = \
      apply_all_schur_steps(sDtN, sRHS, BC, sBeta_U, sGamma_V, sOmega, sCRHS, a, b, p-1, q-1, 
        debug=debug, transposed_solve=transposed_solve, 
        layers_pre=prefact_sol.layers, lastInvLHS_reuse=prefact_sol.lastInvLHS)

  ###### [- 1.1]: Schwarz step - scatter eliminate: back_sub ######

  # print(f'norm={sBC[:,:,0,1:4].norm()}')
  if BC=='Neumann-scatter':
    assert not augmented_system
    nBC = nBC_back_fill(sBC, transposed_solve=transposed_solve)
  else:
    nBC = sBC
  #print(f'norm={nBC[:,:,0,1:4].norm()}')

  if debug:
    start_time = time.time()

  ###### [- 1]: Schwarz step: back_sub ######

  if new_version:
    sol_blockwise = batch_Dirichlet_solver(
        BC=nBC, RHS=RHS, lam=lam, 
        compute_energy=True, transposed_solve=transposed_solve
      )
  else:
    # sol_blockwise = batch_Dirichlet_solver(BC=torch.reshape(nBC, [batch_size * a * b, 2*(p+q-2), nBC.shape[-1]]), RHS=RHS, compute_energy=True)
    # sol_blockwise = batch_Dirichlet_solver(BC=torch.reshape(nBC, [batch_size, a, b, 2*(p+q-2), nBC.shape[-1]]), 
    #       RHS=torch.reshape(RHS, [batch_size, a, b] + list(RHS.shape[-2:])), 
    #       compute_energy=True)
    assert not augmented_system
    sol_blockwise = batch_Dirichlet_solver(
        BC=torch.reshape(nBC, [batch_size * a * b, 2*(p+q-2), nBC.shape[-1]]), 
        RHS=torch.reshape(RHS, [batch_size * a * b] + list(RHS.shape[-2:])), 
        lam=None,
        compute_energy=True, transposed_solve=transposed_solve
      )

  if debug:
    print(f'post solve time: {(time.time()-start_time)*1000} ms'); 

  X_uv = bottom_up(torch.reshape(sol_blockwise.X_uv, [batch_size, a, b, p, q, -1]), value_sum=False)
  sol.X_uv = X_uv[:,0,0,...] # remove dims of size 1
  # sol.X = torch.reshape(sol.X_uv.clone().transpose(-2,-3), [batch_size,-1, sol.X_uv.shape[-1]]) 
  sol.X = image_flatten(sol.X_uv.clone())
  sol.lam = lam[:,0,0,...] if augmented_system else None

  # this for back-sub
  sol.batch_Dirichlet_solver = batch_Dirichlet_solver
  sol.get_nRHS = get_nRHS
  sol.DtN = DtN
  sol.nBeta_U, sol.nGamma_V, sol.nOmega = nBeta_U, nGamma_V, nOmega
  sol.lastInvLHS = lastInvLHS
  sol.ab_wh = ((a,b), (p,q))
  sol.batch_size = batch_size
  sol.dtype = dtype
  sol.layers = layers
  sol.sol_scatter = sol_scatter
  sol.sDtN = sDtN
  sol.sBeta_U, sol.sGamma_V, sol.sOmega = sBeta_U, sGamma_V, sOmega
  # end for back-sub

  sol.BC = BC
  # do not use this anymore: sol.RHS = RHS
  sol.Beta = Beta
  # sol.Chi = sol.X_uv # stop using this confusing name. 

  #def flatten(XXX):
  #  return torch.reshape(XXX.transpose(-2,-3), [batch_size,-1, XXX.shape[-1]]) 

  # solutions: 
  sol.sol_blockwise = sol_blockwise

  '''
  def BHWC():
    return sol.X_uv.transpose(-2,-3)

  def BWHC():
    return sol.X_uv

  def BCHW():
    return sol.X_uv.clone().permute(0,3,2,1)

  sol.BHWC = BHWC
  sol.BWHC = BWHC
  sol.BCHW = BCHW
  '''

  sol.helpers = Helpers()


  # print(f'debug: condense_Dirichlet_solve(): {sol.X}')

  return sol


import gc

# Per-call torch.cuda.synchronize()+gc.collect()+empty_cache() caps peak memory
# on very large grids, but for many small solves (e.g. attention) it dominates
# runtime: ~16 of ~22 ms at ab_wh=((8,8),(8,8)) is spent here, not solving.
# Default off (~4.9x faster fwd+bwd at attention size, bit-identical results);
# set SINV_RECLAIM_MEMORY=True on huge grids if you need the old behaviour.
SINV_RECLAIM_MEMORY = False

def _sinv_reclaim_memory():
  if SINV_RECLAIM_MEMORY:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

class Sinv2D_transposed_reuse(torch.autograd.Function):

  @staticmethod
  def forward(ctx, 
    alpha, 
    beta, 
    boundary_condition, 
    ab_wh, 
    betaU,
    gammaV,
    omega,
    sigma,
    transposed_solve,
    options
  ):

    ctx.options = options
    ctx.transposed = transposed_solve

    #alpha = alpha.detach()
    #beta  = beta.detach()

    if torch.is_tensor(boundary_condition):
      bc = boundary_condition.detach()
      assert False 
      '''grad w.r.t. Dirichlet BC is not implemented'''
    else:
      bc = boundary_condition
      ctx.BC = bc

    # ctx.save_for_backward only for inputs and outputs:
    # https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483

    # sol_forward = schwarz_schur_involution(alpha, beta, boundary_condition=bc, wh=ab_wh[1], transposed_solve=ctx.transposed)

    sol_forward = schwarz_schur_involution(alpha, beta, boundary_condition=bc, wh=ab_wh[1], 
        beta_u=betaU, gamma_v=gammaV, omega=omega, sigma=sigma, 
        transposed_solve=ctx.transposed
    )

    X_BWHC = sol_forward.X_uv
    lam = sol_forward.lam

    # X_BWHC = X_BWHC.detach()
    del sol_forward

    _sinv_reclaim_memory()

    a,b,w,h,W,H = unpack_dims(ab_wh)

    ctx.ab_wh = ab_wh
    ctx.save_for_backward(alpha, beta, X_BWHC, 
      betaU, gammaV, omega, sigma, lam
      )

    return X_BWHC, lam

  @staticmethod
  def backward(ctx, grad, grad_lam):
    '''
    the grad should be a beta-type tensor instead of a chi-type tensor.
    '''

    # x = A \ b, dx = A \ db, for symmetric A
    # (df/db) = A \ (df/dx)

    device = grad.device

    with torch.device(device):

      # A, b, x = ctx.saved_tensors
      alpha, beta, X_BWHC, \
        betaU, gammaV, omega, sigma, lam \
          = ctx.saved_tensors

      augmented_system = not (omega is None)

      if not augmented_system:
        assert grad_lam is None

      #print(f'grad: {device, grad.device, alpha.device}')

      ab_wh = ctx.ab_wh
      BC = ctx.BC
      options = ctx.options

      if False:
        '''
        dummy tests for memory leakage
        https://github.com/pytorch/pytorch/issues/7343
        '''
        R1 = torch.zeros_like(alpha)
        R2 = torch.zeros_like(beta)
        del alpha, beta, X_BWHC
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return R1, R2, None, None, None

      # ab_wh = ((a,b),(w,h))

      '''TypeError: save_for_backward can only save variables, but argument 0 is of type tuple'''

      
      assert grad.shape==X_BWHC.shape

      # grad_abnc = top_down(grad[:,None,None,...], value_divide=True, ab=ab_wh[0]) # BWHC to Chi-type tensor
      beta_grad = beta_from_BWHC(grad, ab_wh=ab_wh)

      # print(f'beta_grad: {beta_grad.device}')

      if False:

        # print(f'Sinv2D_transposed_reuse, {alpha.device}, {beta.device}')
        # with torch.device(device):
        '''TODO: should be reused from forward, currently sol structure cannot be stored in ctx'''
        sol_forward = schwarz_schur_involution(alpha, beta, boundary_condition=BC, wh=ab_wh[1], transposed_solve=ctx.transposed)

        sol_backward  = schwarz_schur_involution(None, beta_grad, boundary_condition=BC, wh=None, prefactored_solution=sol_forward, transposed_solve=(not ctx.transposed))

        assert not augmented_system

      else:

        # sol_backward  = schwarz_schur_involution(alpha, beta_grad, boundary_condition=BC, wh=ab_wh[1], transposed_solve=(not ctx.transposed))

        sol_backward = schwarz_schur_involution(
            alpha, beta_grad, boundary_condition=BC, wh=ab_wh[1], 
            beta_u=betaU, gamma_v=gammaV, omega=omega, sigma=grad_lam, 
            transposed_solve=(not ctx.transposed)
        )

      # sol_backward_BC # TODO.

      gradBeta  = sol_backward.X_uv
      grad_sigma = sol_backward.lam

      # print(f'gradBeta: {gradBeta.device}')

      # gradBeta = gradBeta.detach()
      if 'sol_forward' in locals():
        del sol_forward
      del sol_backward

      '''TODO: not sure why I have to do this: '''
      # gradBeta = gradBeta.to(X_BWHC.device) 

      # critical to use value_divide=False to propogate gradient: 
      gradBeta = top_down(gradBeta[:,None,None,...], value_divide=False, ab=ab_wh[0])
      # gradBeta [B,W,H,c] --> [B,a,b,w,h,c]

      gradBeta = image_flatten(gradBeta)
      # [B,a,b,w,h,c] --> [B,a,b,w*h,c]

      X_Babwhc = top_down(X_BWHC[:,None,None,...], value_divide=False, ab=ab_wh[0])
      X_Babnc = image_flatten(X_Babwhc)
      # [B,W,H,c] --> [B,a,b,w,h,c] --> [B,a,b,w*h,c]

      #print(f'Sinv2D_transposed_reuse, {beta_grad.device}, {gradBeta.device}, {X_Babnc.device}')
      #print(f'Sinv2D_transposed_reuse, {gradBeta.shape}, {X_Babnc.shape}')
      # NOTE: here input to torch_batch_outer should be two chi-type tensors, since alpha adds up to yield the final system at overlapped pixels.
      # NOTE: the order matters for torch_batch_outer.
      gradAlpha = - torch_batch_outer(
          gradBeta, X_Babnc
          ) 

      assert gradAlpha.shape==alpha.shape

      _sinv_reclaim_memory()

      grad_bc = None

      if augmented_system:

        # NOTE: here input to torch_batch_outer should be one chi-type tensor with a global-type tensor, since betaU, gammaV adds up to yield the final system at overlapped pixels.

        def torch_batch_outer2(A, B):
          assert len(A.shape)==len(B.shape)
          # assert A.shape==B.shape # got rid of this condition check for broadcasting. 
          return A @ hermitian(B)

        grad_betaU  = - torch_batch_outer2(gradBeta, lam[:,None,None,...]) # [B,a,b,w*h,c] x [B,1,1,ranks,c] --> [B,a,b,w*h,ranks]
        grad_gammaV = - torch_batch_outer2(grad_sigma[:,None,None,...], X_Babnc) # [B,1,1,ranks,c] x [B,a,b,w*h,c] --> [B,a,b,ranks,w*h]
        grad_omega  = - torch_batch_outer(grad_sigma, lam) # [B,ranks,c] x [B,ranks,c] --> [B,ranks,ranks]
        # checked broadcast rules, e.g. assert (torch.ones([10,6,7,43,2]) @ torch.ones([10,1,1,2,7])).shape == torch.Size([10, 6, 7, 43, 7])
        assert grad_omega.shape==omega.shape
        
      else:
        grad_betaU = None
        grad_gammaV = None
        grad_omega = None
        grad_sigma = None

      return \
        gradAlpha, \
        gradBeta, \
        grad_bc, \
        None, \
        grad_betaU, \
        grad_gammaV, \
        grad_omega, \
        grad_sigma, \
        None, \
        None # the last None is for options.


def sol2numpy(sol, index=None, require_lam=False):
  
  XX  = sol.X if (index is None) else sol.X[index]
  if sol.lam==None:
    if require_lam:
      return XX.detach().cpu().numpy(), None
    else:
      return XX.detach().cpu().numpy()
  else:
    lam = sol.lam if (index is None) else sol.lam[index]
    return XX.detach().cpu().numpy(), lam.detach().cpu().numpy()


def sinv2d_aug_numpy(x_BWHC, lam, index=None):
  
  # index = 0
  assert not (index is None)
  x_scipy = image_flatten(x_BWHC)[index].detach().cpu().numpy()
  lam_scipy = None if lam is None else lam[index].detach().cpu().numpy()
  
  return x_scipy, lam_scipy
  

def Smul2D(Alpha, Chi, wh, Chi_like_output=True):

  # this multiplies the sparse matrix A with dense matrix X
  # that alpha and chi represent resp. 
  '''
  beta's values have been arbitary divided at the pacth border. 
  Alpha @ Beta is incorrect since this it has no communication between adjacent patches yet. 
  '''

  '''
  '''
  
  w, h = wh[0], wh[1]

  channels = Chi.shape[-1]

  Beta = Alpha @ Chi # patchwise product

  if Chi_like_output:
    Chi2 = beta_sync_across_patches(Beta, wh=wh)
    return Chi2
  else:
    return Beta


def uniform_lap_patch(p, q):

  import pyamg
  import time
  import math

  #sten = pyamg.gallery.diffusion_stencil_2d(epsilon=1,theta=0.0,type='FD')
  sten = np.array([[ 0., -1., -0.], [-1.,  4., -1.], [-0., -1.,  0.]])

  print(f'sten:{sten}')
  # sten = np.array([[-1,-1,-1], [-1,8.0,-1], [-1,-1,-1]])

  # due to the y-first ordering for pyamg
  # https://pyamg.readthedocs.io/en/latest/_modules/pyamg/gallery/stencil.html#stencil_grid
  if True:
    A = pyamg.gallery.stencil_grid(sten.transpose(), (q, p))
  else:  
    A = pyamg.gallery.stencil_grid(sten, (p, q))

  d = A.shape[0]

  DA = A.toarray()

  np.fill_diagonal(DA, 0)

  diag = np.sum(DA, axis=1)

  np.fill_diagonal(DA, -diag)

  print(np.linalg.norm(DA @ np.ones(d)))

  return DA


def dense_patchwise_laplacian_rand_walk(images):

  assert(False) # work in progress

  # images: [..., w, h, channels]
  # ... can be batch, or batch,a,b

  batch_dims = len(images.shape) - 3

  batch_sizes = list(images.shape[0:batch_dims])

  c = images.shape[-1]
  h = images.shape[-2]
  w = images.shape[-3]

  dL_unflatten = torch.zeros(batch_sizes + [w,h,w,h])

  dL_unflatten 

  images[...,:,:,:]

  return


def _beta_topdown(beta_abuv, ab_wh, value_divide=True):

  # beta_abuv: [batch_size, 1, 1, W, H, C]

  ab = ab_wh[0]
  wh = ab_wh[1]
  a = ab[0]
  b = ab[1]
  w = wh[0]
  h = wh[1]

  import math

  div_level = (math.log2(a)+math.log2(b))/2
  assert 12==12.0
  assert div_level==int(div_level)
  div_level = int(div_level)

  RHS_bdr = top_down(beta_abuv.clone(), div_level=div_level, value_divide=value_divide)

  # mRHS = flatten_and_permute(RHS_bdr, (w, h)) # the permute function has been moved to outside if necessary.
  mRHS = image_flatten(RHS_bdr)

  return mRHS


def _core_beta_or_chi_from_BWHC(
  BWHC, 
  ab_wh, 
  value_divide=True, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
  ):
  '''
  old name was beta_from_image_whc
  '''

  RHS_abuv = BWHC[:,None,None,...]

  beta = _beta_topdown(RHS_abuv, ab_wh=ab_wh, value_divide=value_divide)

  # beta = torch.reshape(beta, torch.Size([-1]) + beta.shape[-2:])

  if bfs_order:
    beta = convert_to_boundary_first_ordering_rhs(beta, wh=ab_wh[1])

  return beta


def beta_from_BWHC(
  BWHC, 
  ab_wh, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
  ):

  return _core_beta_or_chi_from_BWHC(BWHC, ab_wh=ab_wh, value_divide=True, bfs_order=bfs_order)


def chi_from_BWHC(
  BWHC, 
  ab_wh, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
  ):

  return _core_beta_or_chi_from_BWHC(BWHC, ab_wh=ab_wh, value_divide=False, bfs_order=bfs_order)


def diag_alpha_from_beta(beta):
  '''
  construct a diag alpha from a single column beta

  Example: 

  beta = torch.ones([1,3,4,9,1])
  beta[...,0,0] = 2
  beta[...,1,0] = 3
  alpha = tsi.diag_alpha_from_beta(beta=beta)
  print(alpha[...,0,0], alpha[...,1,1])
  '''

  assert beta.shape[-1]==1

  n = beta.shape[-2]

  alpha = torch.zeros(beta.shape[:-2] + torch.Size([n,n]), dtype=beta.dtype, device=beta.device)

  # print(alpha.shape, alpha[..., torch.arange(n), torch.arange(n)].shape, beta.shape)

  alpha[..., torch.arange(n), torch.arange(n)] = beta[..., 0]

  return alpha


def beta_from_alpha_diag(alpha):
  '''
  construct a beta from the diagonal of alpha

  Example: 

  alpha = torch.zeros([1,3,4,9,9])
  alpha[...,0,0] = 2
  alpha[...,1,1] = 3
  beta = tsi.beta_from_alpha_diag(alpha=alpha)
  print(beta[...,0,0], beta[...,1,0], beta.shape)
  '''

  assert alpha.shape[-1]==alpha.shape[-2]

  n = alpha.shape[-2]

  beta = torch.zeros(alpha.shape[:-2] + torch.Size([n,1]), dtype=alpha.dtype, device=alpha.device)

  beta[..., 0] = alpha[..., torch.arange(n), torch.arange(n)]

  return beta


def _core_alpha_from_phi(phi, ab_wh): 

  '''consider merging it with patch_laplacian_adjust_boundary'''

  assert DEFAULT_BOUNDARY_FIRST_ORDER==False

  
  a,b,w,h,W,H = unpack_dims(ab_wh) 

  assert w==5
  assert h==5

  '''
  bottom = [0,1,2,3,4]
  left = [20,15,10,5,0]
  right = [4,9,14,19,24]
  top = [24,23,22,21,20] 

  left = torch.tensor(left, dtype=torch.int64)
  right = torch.tensor(right, dtype=torch.int64)
  bottom = torch.tensor(bottom, dtype=torch.int64)
  top = torch.tensor(top, dtype=torch.int64)
  '''

  bottom = slice(0,5)
  left  = slice(0,25,5)
  right = slice(4,25,5)
  top  =  slice(20,25)

  
  # this does not work: the left-value cannot have two brackets
  # alpha[..., 1:, :, left, :][...,:,left] = alpha[..., 1:, :,left, :][...,:,left] / 2.0
  # alpha[..., :-1, :, right, :][...,:,right] = alpha[..., :-1, :,right, :][...,:,right] / 2.0
  # alpha[..., :, 1:, bottom, :][...,:,bottom] = alpha[..., :, 1:,bottom, :][...,:,bottom] / 2.0
  # alpha[..., :, :-1, top, :][...,:,top] = alpha[..., :, :-1,top, :][...,:,top] / 2.0


  mask = torch.ones([w*h, w*h])
  mask[bottom, bottom] = mask[bottom, bottom] * 0.5
  mask[top, top] = mask[top, top] * 0.5
  mask[right, right] = mask[right, right] * 0.5
  mask[left, left] = mask[left, left] * 0.5

  # print(mask)

  '''this way of slicing only works for slice, not int tensor'''

  return phi * mask


def alpha_from_phi(phi, ab_wh, in_place=False): 

  if in_place:
    alpha = phi
  else:
    alpha = phi.clone()

  patch_laplacian_adjust_boundary(alpha, ab_wh)
  # alpha = _core_alpha_from_phi(alpha, ab_wh)

  return alpha


def beta_from_numpy(
  numpyRHS, 
  ab_wh, 
  dtype=None, 
  value_divide=True, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
  ):
  """
  numpyRHS is xxxx by c (c is usually 1), i.e., the flattened rhs. 
  ab_wh: ((a,b),(w,h))
  """
  
  if dtype is None:
    dtype = torch.get_default_dtype()
  
  a,b,w,h,W,H = unpack_dims(ab_wh)

  assert(len(numpyRHS.shape)==2)

  RHS_f = torch.tensor(numpyRHS, dtype=dtype)[None,...] 

  BWHC = image_unflatten(RHS_f, W, H)

  assert value_divide==True # the False option has been removed.
  beta = beta_from_BWHC(BWHC, ab_wh, bfs_order=bfs_order)

  return beta


def gamma_from_numpy(
  numpyRHS, 
  ab_wh, 
  column_vectors=False, # this means numpyRHS
  dtype=None, 
  value_divide=True, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER
  ):

  assert(len(numpyRHS.shape)==2)

  return beta_from_numpy(
      numpyRHS if column_vectors else numpyRHS.T, 
      ab_wh, 
      dtype=dtype, 
      value_divide=value_divide, 
      bfs_order=bfs_order
      ).transpose(-1,-2)


def meshgrid_coordinates(WH, dtype):
  '''
  input: 
      can use dtype=torch.get_default_dtype()
  output: [W,H,2]
      can do 
      XY[None,None,None,...] to convert to [1,1,1,W,H,2]
  '''
  grid_x, grid_y = torch.meshgrid(
      torch.arange(WH[0], dtype=dtype), 
      torch.arange(WH[1], dtype=dtype), 
      indexing='ij')

  XY = torch.stack((grid_x, grid_y), dim=-1)

  return XY


def mesh_hier(ab_wh, flattened=True):
  '''
  output: [a, b, w, h, channels==2]
  '''

  a,b,w,h,W,H = unpack_dims(ab_wh)

  grid_x, grid_y = torch.meshgrid(torch.arange(W, dtype=torch.int64), 
                                  torch.arange(H, dtype=torch.int64), 
                                  indexing='ij')
  
  # idx = slice(45,50)
  # idy = slice(400,406)
  # print(grid_x.shape, grid_x[idx, idy], grid_y[idx, idy])

  FX = grid_x[None,None,None,...,None]
  FY = grid_y[None,None,None,...,None]

  # print(FX.shape)

  FXY = torch.cat([FX,FY], dim=-1)

  FXY = top_down(FXY, ab=(a,b), value_divide=False)
  # FY = top_down(FY, ab=(a,b), value_divide=False)
  # out: [batch_size, a, b, w, h, 2]
  # removed batch_size dim, and merge the W,H channels.  
  FXY = FXY[0]
  # FY = FY[0]
  if flattened:
    FXY = torch.reshape(FXY.transpose(-2,-3), FXY.shape[:-3] + torch.Size([-1]) + FXY.shape[-1:])
  # FY = torch.reshape(FY.transpose(-2,-3),  FY.shape[:-3] + torch.Size([-1]) +  FY.shape[-1:])

  return FXY


def patch_adjacency_mask(wh, nebr):

  # the output adj_mask is boolean
  XY = mesh_hier(((1,1), wh))[0,0] * 1.0 
  if nebr=='5pt':
    # 5pt
    adj_mask = (torch.cdist(XY, XY, p=2)**2) <= 1.01 # w*h by w*h
  else:
    assert nebr=='9pt'
    # 9pt
    adj_mask = (torch.cdist(XY, XY, p=2)**2) <= 2.01 # w*h by w*h
  return adj_mask


def _torch_bottom_up_sparse_indices(ab_wh):

  a,b,w,h,W,H = unpack_dims(ab_wh)

  FXY = mesh_hier(ab_wh=ab_wh, flattened=True)

  FX = FXY[...,0:1]
  FY = FXY[...,1:2]

  # print(IV.shape, FX.shape, IV.shape, ID.shape, FX[-1,0][5:10], FY[-1,0][5:10])

  FD = FX + FY * W

  IX, IY = torch_batch_cartesian_prod(FD, FD)

  # print(IX[-1,0][5:10,0], IY[-1,0][0,15:20])
  # print(IV.shape, IX.shape, IY.shape) # all have the same shape

  return IX, IY


def _torch_bottom_up_sparse_VIJ(DA, wh, numpy_out=False, adj='all'):

  # DA is [..., a,b,pwh,pwh] # pwh = w*h
  # output 
  # if adj=='all' IV,IX,IY is three variables with the same shape as DA: 
  #   the shape [..., a,b,w*h,w*h] 
  # if not: the shape is [..., a,b,e]

  w = wh[0]
  h = wh[1]

  a = DA.shape[-4]
  b = DA.shape[-3]

  IV = DA

  IX, IY = _torch_bottom_up_sparse_indices(ab_wh=((a,b),(w,h)))

  if adj=='9pt' or adj=='5pt':
    adj_mask = patch_adjacency_mask(wh=(w,h), nebr='9pt')
    # print('torch_bottom_up_sparse_VIJ(): shapes:', IV.shape, IX.shape, IY.shape)
    # print(adj_mask.shape, adj_mask*1.0)
    ii, jj = torch.where(adj_mask)
    # print(ii.shape, ii.shape[0] / (w*h), ii, jj, adj_mask[ii,jj])

    if True:
      # double check
      ii2, jj2 = torch.where(torch.logical_not(adj_mask))
      removed = IV[..., ii2, jj2]
      # print(removed.shape, removed.abs().sum())
      assert 0.0==removed.abs().sum()

    IX = IX[..., ii, jj]
    IY = IY[..., ii, jj]
    IV = IV[..., ii, jj]
    # print(IV.shape, IX.shape, IY.shape)

  else:
    assert adj=='all'

  if numpy_out:
    return IV.detach().cpu().numpy(), IX.detach().cpu().numpy(), IY.detach().cpu().numpy()
  else:
    return IV, IX, IY


def alpha_to_scipy(
  Alpha, 
  ab_wh, 
  shape=None, 
  boundary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER, 
  adj='all'
  ):
  """ 
  adj {'all', '9pt', '5pt'}
  """

  assert Alpha.shape[0]==1 
  '''otherwise not implemented'''

  DA = Alpha[0]
  
  import scipy
  # input: DA: [a*b, (w*h), (w*h)]

  a,b,w,h,W,H = unpack_dims(ab_wh)

  DA_abxx = torch.reshape(DA, [a,b]+list(DA.shape[-2:]))
  # DA_abxx: [a,b,(w*h), (w*h)]

  if boundary_first_order:
    # apply *inverse permutation* of boundary-first ordering

    DA_abxx = convert_to_lexicographic_sweeping_order(DA_abxx, [w,h])

  IV, IX, IY = _torch_bottom_up_sparse_VIJ(DA_abxx, wh=(w,h), numpy_out=True, adj=adj)

  N = W * H
  A_scipy = scipy.sparse.csr_matrix((IV.flatten(), (IX.flatten(), IY.flatten())), shape=(N, N))

  As = list()
  As.append(A_scipy)

  return As


'''renamed, and compatible with old names'''

def torch_bottom_up_sparse_scipy(
  DA, 
  ab_wh, 
  shape=None, 
  boundary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER, 
  adj='all'
  ):
  return alpha_to_scipy(Alpha=DA[None,...], 
    ab_wh=ab_wh, shape=shape, boundary_first_order=boundary_first_order, adj=adj)[0]


def flattenWH(X):
  # input image: 
  # [..., w, h, c]
  # output
  # [..., w*h, c] such that x-coordinates move faster.  
  return torch.reshape(X.transpose(-2,-3), X.shape[:-3] + torch.Size([-1]) + X.shape[-1:])


def patch_laplacian_adjust_boundary(alpha, ab_wh):
  '''this modified alpha'''

  # alpha: (batch_size, a, b, w*h, w*h)
  # alpha is being modified.

  a,b,w,h,W,H = unpack_dims(ab_wh) # tip

  # Row-major flat indices of the four edges of a w×h patch.
  # bottom: row 0,  top: row h-1,  left: col 0,  right: col w-1.
  bottom = torch.arange(w, dtype=torch.int64)
  top    = torch.arange((h - 1) * w, h * w, dtype=torch.int64)
  left   = torch.arange(h, dtype=torch.int64) * w
  right  = torch.arange(h, dtype=torch.int64) * w + (w - 1)

  rr, cc = torch.meshgrid(left, left, indexing='ij')
  alpha[..., 1:, :, rr, cc] *= 0.5
  rr, cc = torch.meshgrid(right, right, indexing='ij')
  alpha[..., :-1, :, rr, cc] *= 0.5
  rr, cc = torch.meshgrid(bottom, bottom, indexing='ij')
  alpha[..., :, 1:, rr, cc] *= 0.5
  rr, cc = torch.meshgrid(top, top, indexing='ij')
  alpha[..., :, :-1, rr, cc] *= 0.5

  return


def batch_dense_laplacian_whwh(X):
  # input image: 
  # [..., w, h, c]
  # output
  # [..., w, h, w, h]
  assert False # TODO: WIP
  return


def single_patch_uniform_laplacian(wh):

  from . import torch_mesh_processing as tmgp

  w, h = wh[0], wh[1]

  batch_size = 1
  n = w*h
  f = (w-1)*(h-1) * 4 # since this uses double-covered mesh

  old_numpy = False # True # False
  
  if old_numpy:

    import mesh_processing as mgp
    mesh = mgp.MeshGrid(w, h)
    np_V, np_F = mesh.V, mesh.F
    II, JJ, _, _ = mgp.lap_entries(np_V, np_F, multicol=False)
    assert np_F.shape[0]==f
    assert np_V.shape[0]==n

    adjs = torch.tensor(np.stack([II,JJ]), dtype=torch.int64, requires_grad=False)

  else:

    # numpy code to be re-implemented in torch.

    V, F, _ = tmgp.grid_to_double_covered_mesh(w, h)
    II, JJ, _, _ = tmgp.lap_entries(V, F, multicol=False)

    adjs = torch.stack([II,JJ])
  
  au = torch.concat([
        torch.ones([batch_size, f, 1]), 
        torch.zeros([batch_size, f, 1]),
        torch.ones([batch_size, f, 1])
      ],axis=-1)

  # dtype = torch.get_default_dtype()

  if old_numpy:
    np_gn_p = mgp.grad_normalized(np_V, np_F)
    gn_p = torch.tensor(np_gn_p, requires_grad=False) # dtype=dtype, 
  else:
    gn_p = tmgp.grad_normalized(V, F)

  # print(f'gn_p: {gn_p.shape, adjs.shape, F.shape}')
  # print('au', au.shape)

  values = tmgp.assemble_lap_values(au, gn_p) #.to(dtype=dtype)

  ## spLA is [n,n,batch_size], last dim is dense---what torch only supports
  spLA = torch.sparse_coo_tensor( 
                indices=adjs, 
                values=values, 
                size=[n, n, batch_size])
                
  dLA = spLA.to_dense().permute(2,0,1) # dtype=torch.float32, not supported

  return dLA[0]


def batch_diag(XX):

  assert XX.shape[-1]==1
  n = XX.shape[-2]

  RR = torch.zeros(list(XX.shape[:-2]) + [n,n], dtype=XX.dtype)

  # this is inefficient. 
  RR[..., np.arange(n), np.arange(n)] = XX[..., 0]

  return RR


def alpha_from_patchwise_features(CX):
  # CX: [batch_size,a,b,w,h,c], e.g., the output of top_down
  
  _, a, b, w, h, c = list(CX.shape)

  assert c==1

  if True:

    import mesh_processing as mgp
    mesh = mgp.MeshGrid(w, h)
    Gx = torch.tensor(mesh.Gx.todense())[None, None, None]
    Gy = torch.tensor(mesh.Gy.todense())[None, None, None]
    F  = torch.tensor(mesh.F, dtype=torch.int64)

  C = flattenWH(CX)
  # [..., n, 1]

  CF = (C[...,F[:,0],:] + C[...,F[:,1],:] + C[...,F[:,2],:]) / 6.0
  # [..., f, 1]

  CF = batch_diag(CF) 

  # print(C.shape, Gx.shape, F.shape)
  # print(CF.shape)

  assert torch.is_complex(Gx)==False
  LA = Gx.transpose(-1,-2) @ CF @ Gx + Gy.transpose(-1,-2) @ CF @ Gy

  alpha = LA

  return alpha
  

def single_patch_mass_diagonal(wh):

  w, h = wh[0], wh[1]

  M = torch.eye(w*h)
  m = 2 * torch.ones([w,h]) # since using double-covered mesh.

  m[..., [0,-1], :] = m[..., [0,-1], :] * 0.5
  m[..., :, [0,-1]] = m[..., :, [0,-1]] * 0.5

  ind = torch.arange(w*h)

  M[...,ind,ind] = m.transpose(-1,-2).flatten()

  return M


######### some examples of alphas #########


def alpha_uniform_laplacian(ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER): 
  '''
  return a regular 4pt Laplacian.
  '''

  a,b,w,h,W,H = unpack_dims(ab_wh)
  alpha_shape = [1,a,b,w*h,w*h]
  alpha = torch.zeros(alpha_shape)

  alpha[...,:,:] = single_patch_uniform_laplacian([w,h])

  if bfs_order:
    alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])

  return alpha


def alpha_coeff_laplacian(image_hwc, ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):

  # modified from alpha_image_laplacian

  image_whc = image_hwc[None,None,None].transpose(3,4)

  a,b,w,h,W,H = unpack_dims(ab_wh)

  CX = top_down(image_whc, div_level='Auto', value_divide=False, ab=ab_wh[0])

  # FCX = flattenWH(CX)

  alpha = alpha_from_patchwise_features(CX)

  alpha_shape = [1,a,b,w*h,w*h]
  assert(list(alpha.shape)==alpha_shape)

  adj_mask = patch_adjacency_mask([w,h], nebr='9pt')

  alpha = alpha * (1.0*adj_mask)

  if bfs_order:
    alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
  
  return alpha


def alpha_uniform_mass_diagonal(ab_wh, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):

  a,b,w,h,W,H = unpack_dims(ab_wh)
  alpha_shape = [1,a,b,w*h,w*h]
  alpha = torch.zeros(alpha_shape)

  alpha[...,:,:] = single_patch_mass_diagonal([w,h])

  if bfs_order:
    alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])

  return alpha


def batch_dense_affinity_image(X, wh, params=None):
  # this returns a patch-dense matrix

  # input X: 
  # [..., n, c]
  # output
  # [..., n, n]

  w, h = wh[0], wh[1]

  shape = X.shape
  c = shape[-1]
  assert shape[-2]==(w*h)
  BS = shape[-2:]

  D = torch.cdist(X, X)

  # print(D.shape)

  sigma = 0.0
  if hasattr(params,'sigma'):
    sigma = params.sigma
  sigma = torch.tensor(sigma)
  # print(sigma)
  
  if hasattr(params,'temp_kernel'):
    # Weights = 1 - D*D # 
    Weights = torch.maximum(torch.exp(- 1.0 * D*D), torch.tensor(0.01))
  else:
    Weights = torch.maximum(torch.exp(-900.0 * D*D), sigma)

  return Weights


def alpha_image_laplacian(
  image, 
  ab_wh, 
  diag_zero_sum=True, 
  adjust_border_patches=False, 
  bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER, 
  params=None
  ):
  '''
  Parameters
  ----------
  adjust_border_patches should be 
    Flase if using FEM, or 
    True only for the purpose of being consistent with external code like pymatting. 
  '''

  # assume the image is  [H,W,C] a torch tensor

  if len(image.shape)==3:
    # [H,W,C] --> [B,H,W,C]
    image = image[None, ...]
  else:
    assert len(image.shape)==4

  image_whc = image[:, None, None, ...].transpose(3,4)
  # [B,1,1,W,H,C]
  # image_whc should be [W,H,C], not the common [H,W,C]

  # print(image_whc.shape)

  a,b,w,h,W,H = unpack_dims(ab_wh)

  CX = top_down(image_whc, div_level='Auto', value_divide=False, ab=ab_wh[0]) # ab=(8,8))# tip

  # print(CX.shape)

  FCX = flattenWH(CX)

  alpha = batch_dense_affinity_image(FCX, ab_wh[1], params=params)
  # print(alpha.shape)
  alpha_shape = [1,a,b,w*h,w*h]
  assert(list(alpha.shape)==alpha_shape)

  adj_mask = patch_adjacency_mask([w,h], nebr='5pt' if hasattr(params,'5pt') else '9pt')
  # print(adj_mask.shape)

  if hasattr(params,'random_sparse_patch'):
    alpha = torch.rand(alpha_shape) - 0.5

  alpha = alpha * (1.0*adj_mask)

  if hasattr(params, 'random_dense_patch'):
    alpha = torch.rand(alpha_shape) - 0.5
    # rand_like gives random number between [0,1]
    #print(alpha[0,0,0])

  if hasattr(params, 'eye'):
    alpha = torch.zeros(alpha_shape)
    ind = torch.arange(w*h)
    alpha[...,ind,ind] = 1

    diag_zero_sum = False
    assert adjust_border_patches==True

  # print(alpha.shape)
  # print(alpha[0,1,1])

  if adjust_border_patches:
    patch_laplacian_adjust_boundary(alpha, ab_wh) # modify alpha for patches at the whole-image border 

  if diag_zero_sum:
    alpha = - alpha
    ind = torch.arange(w*h)
    alpha[...,ind,ind] = 0
    alpha[...,ind,ind] = - alpha.sum([-1])

  # print(alpha[0,1,1])
  # print(alpha[0,0,0])

  if bfs_order:
    alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
  
  return alpha


def alpha_image_matting(image_hwc, ab_wh):

  lam = 0.0001 * 2 # 0.0001 # 0.0001 # 0.01
  # problem += str(lam)
  alpha = alpha_image_laplacian(image_hwc, ab_wh, diag_zero_sum=True, adjust_border_patches=True)
  alpha += alpha_uniform_laplacian(ab_wh) * lam / 2
  alpha += alpha_uniform_mass_diagonal(ab_wh) * 0.000002 / 2

  return alpha


def alpha_image_matting_normalized(image_hwc, ab_wh):

  a,b,w,h,W,H = unpack_dims(ab_wh)

  # lam = 0.0001 * 2 / (W*H/512**2) # 0.0001 # 0.0001 # 0.01
  lam = 0.0001 * 2
  # problem += str(lam)
  alpha = alpha_image_laplacian(image_hwc, ab_wh, diag_zero_sum=True, adjust_border_patches=True)
  alpha += alpha_uniform_laplacian(ab_wh) * lam / 2
  alpha += alpha_uniform_mass_diagonal(ab_wh) * 0.000002 / 2 # 0.01 * lam / 2

  # ones = default_chi_for(alpha, fun=torch.ones)
  # diag_sum = Smul2D(alpha, ones, ab_wh[1])
  # assert diag_sum.min() > 0
  # print(diag_sum.shape)

  if False:
    # this is the shifts 
    alpha += alpha_uniform_mass_diagonal(ab_wh) * 0.000002 / 2


  if False:
    beta = beta_from_alpha_diag(alpha)
    chi = chi_from_beta(beta, wh=ab_wh[1])
    assert chi.min() > 0
    chi = torch.sqrt(chi)

    print(f'chi.shape={chi.shape}') # [..., n, 1] 

    alpha = alpha / chi / chi.transpose(-1,-2)

  return alpha


def scipy_sparse_top_down_slow_recursive(A, 
    ab_wh=None,
    init_shape='not used given ab_wh', 
    div_level='not used given ab_wh',
    value_divide=True, 
    input_x_first=True, 
    permute_bounary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER, 
    wh_if_permute_bounary_first_order=None, # (w,h) or True (in which case find wh in ab_wh)
    beta_shaped=True, # False
    ):
  
  if not (ab_wh is None):
    a,b,w,h,W,H = unpack_dims(ab_wh)

    import math
    div_level = int((math.log2(a)+math.log2(b))/2)

    init_shape=(W,H)

  if not (wh_if_permute_bounary_first_order is None):
    permute_bounary_first_order = True
    w,h = wh_if_permute_bounary_first_order

  assert A.shape[0]==(init_shape[0]*init_shape[1])
  p = init_shape[0]
  q = init_shape[1]

  #   # Cs: Cs[i][j] is array of n by n sparse matrix
  Cs = [[A]]

  for level in range(div_level):

    # y before x, (reverse order of bottom up)

    # y axis: 
    m = len(Cs)
    n = len(Cs[0])
    Ns = [[None for j in range(n*2)] for i in range(m)]
    assert q%2==1
    cp = p
    cq = (q-1)//2
    assert Cs[0][0].shape[0]==p*q

    for i in range(m):
      assert n==len(Cs[i])

      for j in range(n):
        assert Cs[i][j].shape[0]==(p*q)

        x_first = True
        if level==0 and not input_x_first:
          x_first = False
        # division: [0, cq) [cq, cq] [cq+1, 2*cq+1)
        #print(f'tip: shapes {p,q}')
        indices = alg.indices_matrix_2d(p, q, 1, 1, 0, 0, x_first=True)

        shared = indices[:, cq].flatten()
        part1 = indices[:, 0:cq+1].transpose().flatten()
        part2 = indices[:, cq:None].transpose().flatten()

        #if level==0 and i==0 and j==0:
        #  print(f'shared{shared}')

        #print(f'test shared: {shared.shape}')

        S = Cs[i][j].copy()
        #print(f'before shapes:{S.shape}')
        # note https://stackoverflow.com/questions/28574041/modifying-block-matrices-in-python
        # this is wrong: S[:,shared][shared,:] = S[:,shared][shared,:] / 2.0.  somehow S[shared,shared] is just the diagonal.
        if value_divide: 
          S[np.ix_(shared,shared)] /= 2
          #print(f'test size: {(S[:,shared][shared,:]).shape}')
        #print(f'after shapes:{S.shape, (S[part1, part1]).shape, type(part1)}')

        Ns[i][j*2] = S[:,part1][part1,:]
        Ns[i][j*2+1] = S[:,part2][part2,:]

        #print(f'reduced shapes: {part1.shape, Ns[i][j*2].shape, Ns[i][j*2+1].shape}')
    Cs = Ns
    p = cp
    q = cq+1

    # x axis: 
    m = len(Cs)
    n = len(Cs[0])
    Ns = [[None for j in range(n)] for i in range(m*2)]
    assert p%2==1
    cp = (p-1)//2
    cq = q
    #print(f'scipy_sparse_top_down: {Cs[0][0].shape[0], p,q,m,n }')
    assert Cs[0][0].shape[0]==p*q

    for i in range(m):
      assert n==len(Cs[i])

      for j in range(n):

        # division: [0, cp) [cp, cp] [cp+1, 2*cp+1)
        indices = alg.indices_matrix_2d(p, q, 1, 1, 0, 0, x_first=True)
        shared = indices[cp, :].flatten()
        part1 = indices[0:cp+1, :].transpose().flatten()
        part2 = indices[cp:None, :].transpose().flatten()

        S = Cs[i][j].copy()

        #print(f'debug, {level}, {S.shape, i, j, m, n}')
        if value_divide:
          S[np.ix_(shared,shared)] /= 2

        Ns[i*2][j] = S[:,part1][part1,:]
        Ns[i*2+1][j] = S[:,part2][part2,:]
    Cs = Ns
    p = cp+1
    q = cq

  dtype = torch.get_default_dtype()

  DA = torch.stack(
      [torch.stack(
          [torch.tensor(Cs[i][j].todense(), dtype=dtype) for j in range(len(Cs[i]))], axis=0) for i in range(len(Cs))]
    )

  if permute_bounary_first_order:

    # B = alg.matrix_spiral_bdr_list(w,h)
    # UB = alg.matrix_interior_list(w,h)
    # ind = torch.tensor(B + UB, dtype=torch.int64)
    # DA = DA[...,ind,:][...,:,ind] # this is only for right-hand-side

    DA = convert_to_boundary_first_ordering(DA, wh=[w,h])

  if not beta_shaped:
    DA = DA.reshape( torch.Size([-1]) + DA.shape[-2:])

  return Cs, DA


def scipy_sparse_top_down_fast(A,
    ab_wh=None,
    value_divide=True,
    permute_bounary_first_order=DEFAULT_BOUNDARY_FIRST_ORDER,
    wh_if_permute_bounary_first_order=None, # (w,h) or True (find wh in ab_wh)
    beta_shaped=True, # False
    ):
  '''
  Vectorized equivalent of scipy_sparse_top_down.

  scipy_sparse_top_down partitions the WxH grid into a*b overlapping w*h patches
  by recursively slicing A's rows/cols; that recursion is O(div_level) full-matrix
  scipy fancy-index copies plus a python double-loop over patches. The result is
  fully determined and loop-free:

    DA[i,j,k,l] = A[g[i,j,k], g[i,j,l]] * 0.5^s(i,j,k,l)

  where g[i,j,k] is the global index (x + y*W) of local node k of patch (i,j)
  (exactly the top_down node ordering, obtained from mesh_hier), and s counts the
  shared internal patch-boundary lines that both endpoints sit on: an x-line at
  X = m*(w-1), 0<X<W-1, halves entries with Xk==Xl==X (once), similarly for y.
  Each internal interface is halved exactly once because the split is hierarchical
  (the line is duplicated into both sub-blocks and never re-split).

  Returns (None, DA) so it is drop-in for `_, DA = scipy_sparse_top_down(...)`.
  The Cs list of sub-matrices is not reconstructed.
  '''
  import scipy.sparse

  assert ab_wh is not None, 'scipy_sparse_top_down_fast requires ab_wh'
  a, b, w, h, W, H = unpack_dims(ab_wh)

  if wh_if_permute_bounary_first_order is not None:
    permute_bounary_first_order = True
    if wh_if_permute_bounary_first_order is not True:
      assert tuple(wh_if_permute_bounary_first_order) == (w, h)

  N = W * H
  assert A.shape[0] == N and A.shape[1] == N

  # Global index (x + y*W) of every patch node, in top_down local order.
  FXY = mesh_hier(ab_wh, flattened=True).detach().cpu().numpy().astype(np.int64)  # [a,b,w*h,2]
  X = FXY[..., 0]                      # [a,b,w*h]
  Y = FXY[..., 1]
  g = X + Y * W                        # [a,b,w*h]

  # Gather A[g_row, g_col] for all patches at once via sorted-key lookup.
  Acoo = A.tocoo()
  akey = Acoo.row.astype(np.int64) * N + Acoo.col.astype(np.int64)
  order = np.argsort(akey, kind='stable')
  akey_s = akey[order]
  aval_s = np.asarray(Acoo.data)[order]

  gr = g[..., :, None]                 # [a,b,w*h,1]
  gc = g[..., None, :]                 # [a,b,w*h,1] -> broadcast to [...,w*h,w*h]
  qkey = (gr.astype(np.int64) * N + gc.astype(np.int64))
  qkey = np.broadcast_to(qkey, (a, b, w * h, w * h)).reshape(-1)

  pos = np.searchsorted(akey_s, qkey)
  pos = np.clip(pos, 0, len(akey_s) - 1)
  hit = len(akey_s) > 0
  vals = np.where((akey_s[pos] == qkey) if hit else np.zeros_like(qkey, dtype=bool),
                  aval_s[pos] if hit else 0.0,
                  0.0)
  vals = vals.reshape(a, b, w * h, w * h)

  if value_divide:
    # per-node membership on an internal x / y interface line
    sx_node = ((X % (w - 1)) == 0) & (X > 0) & (X < W - 1) if w > 1 else np.zeros_like(X, dtype=bool)
    sy_node = ((Y % (h - 1)) == 0) & (Y > 0) & (Y < H - 1) if h > 1 else np.zeros_like(Y, dtype=bool)

    same_x = X[..., :, None] == X[..., None, :]                         # [a,b,wh,wh]
    same_y = Y[..., :, None] == Y[..., None, :]
    ex = same_x & sx_node[..., :, None] & sx_node[..., None, :]
    ey = same_y & sy_node[..., :, None] & sy_node[..., None, :]
    scale = np.power(0.5, ex.astype(np.int64) + ey.astype(np.int64))
    vals = vals * scale

  dtype = torch.get_default_dtype()
  DA = torch.tensor(vals, dtype=dtype)                                  # [a,b,w*h,w*h]

  if permute_bounary_first_order:
    DA = convert_to_boundary_first_ordering(DA, wh=[w, h])

  if not beta_shaped:
    DA = DA.reshape(torch.Size([-1]) + DA.shape[-2:])

  return None, DA


def alpha_from_scipy_sparse(A_scipy, ab_wh, value_divide=True, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):
  '''
  output: alpha, [batch_size==1, a, b, w*h, w*h]
  '''
  _, alpha = scipy_sparse_top_down_fast(A_scipy, ab_wh=ab_wh, value_divide=value_divide, beta_shaped=True, permute_bounary_first_order=bfs_order)
  alpha = alpha[None, ...]
  return alpha


def scipy_sparse_bottom_up(Cs, shape=None, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):

  assert(False) # work in progress

  import scipy
  p = Cs[0][0].shape[0]
  q = Cs[0][0].shape[1]

  m = len(Cs)
  n = len(Cs[0])

  if shape is None:
    N = (m*(p-1)+1) * (n*(q-1)+1)
    shape = (N, N)

  A = scipy.sparse.csr_matrix(([],([],[])), shape=shape)

  for i in range(m):
    for j in range(n):
      # indices = i*p:i*p+p+1   j*q:j*q+q+1

      indices = alg.indices_matrix_2d(p, q, m, n, i, j, x_first=True)
      A[np.ix_(indices,indices)] += Cs[i][j]

  return A


def test_alpha_image_laplacian(image_numpy, ab_wh=((64,64),(5,5))):

  import image_processing as ip
  
  # assume the image is  [H,W,C]

  a,b,w,h,W,H = unpack_dims(ab_wh)

  image = torch.tensor(image_numpy, dtype=torch.get_default_dtype())

  alpha = alpha_image_laplacian(image, ab_wh, diag_zero_sum=False, adjust_border_patches=True) # old:affinity_only=True

  # print(alpha[0,1,1])

  A_scipy = torch_bottom_up_sparse_scipy(alpha[0], ab_wh=ab_wh) # the sparse matrix A. # tip

  ##

  # alpha[...,slice(5),slice(6,16)].shape

  ind = torch.tensor([1,2,3,6,7,8], dtype=torch.int64)

  alpha[...,ind][...,ind,:].shape

  print(A_scipy.shape)

  # A_scipy_ref = pymatting.rw_laplacian(image)
  A_scipy_ref = ip.random_walk_affinity(image_numpy) 

  import math
  div_level = int((math.log2(a)+math.log2(b))/2)
  _, alpha_ref = scipy_sparse_top_down(A_scipy_ref, init_shape=(W,H), div_level=div_level, 
                                          wh_if_permute_bounary_first_order=(w,h)) # tip
  alpha_ref = alpha_ref[None,...]

  A_scipy_ref2 = torch_bottom_up_sparse_scipy(alpha_ref[0], ab_wh=ab_wh) # tip

  print('errors:',\
    np.abs(A_scipy_ref - A_scipy).max(),\
    np.abs(A_scipy_ref - A_scipy).max(),\
    np.abs(A_scipy_ref - A_scipy_ref2).max()
  )
  # on carmeraman float 64: errors: 1.1102230246251565e-16 1.1102230246251565e-16 5e-324

  print('shapes:', alpha.shape, alpha_ref.shape)

  print('errors:',\
    (alpha - alpha_ref).abs().max(),\
    (alpha - alpha_ref).norm()\
  )
  # errors: tensor(1.1102e-16, device='cuda:0') tensor(1.8922e-14, device='cuda:0')

  return


def torch_bottom_up_scipy_sparse_slow(Cs, wh, shape=None, bfs_order=DEFAULT_BOUNDARY_FIRST_ORDER):

  # TODO: correctness to be tested. 

  import scipy
  pwh = Cs[0][0].shape[0]
  # pwh = Cs[0][0].shape[1]
  assert pwh==Cs[0][0].shape[1]

  w = wh[0]
  h = wh[1]

  assert pwh == (w*h)

  is_torch = False
  if torch.is_tensor(Cs):
    # torch array
    m = Cs.shape[0]
    n = Cs.shape[1]
    is_torch = True
    Cs = Cs.detach().cpu().numpy()
  else:
    # numpy array
    m = len(Cs)
    n = len(Cs[0])

  if shape is None:
    N = (m*(w-1)+1) * (n*(h-1)+1)
    shape = (N, N)

  if bfs_order:
    B, UB = alg.matrix_bdr_interior_split(wh=(w,h))
    invBUB = alg.inverse_permutation(B+UB)

  print(invBUB)

  # ind = torch.tensor(B + UB, dtype=torch.int64)

  if True:
    # slow version
    A = scipy.sparse.csr_matrix(([],([],[])), shape=shape)
    for i in range(m):
      for j in range(n):
        indices = alg.indices_matrix_2d(w, h, m, n, i, j, x_first=True)
        indices = indices.transpose().flatten()
        if i==0 and j==0:
          print('indices:', indices)
        if bfs_order:
          R = Cs[i][j][np.ix_(invBUB,invBUB)]
        else:
          R = Cs[i][j]
        A[np.ix_(indices,indices)] += R
  else:
    # much faster: 
    assert False # in progress

  return A

