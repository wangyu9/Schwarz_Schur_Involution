import numpy as np
import torch

import time

import math

import algorithms as alg


def inv_default(xx):
  result, info = torch.linalg.inv_ex(xx)
  return result


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


  if len(image.shape)==2:
    image = image[np.newaxis,...]

  image_batch = image[np.newaxis,...]

  torch_image = torch.tensor(image_batch, requires_grad=False, dtype=torch.get_default_dtype())

  return torch_image


class Level():
  name = 'Level_'


def cf(arr, same=True):
  if not same: 
    arr = torch.flip(arr, [-2])
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


def image_unflatten(Z, dim0, dim1):

  return torch.reshape(Z, list(Z.shape[:-2]) + [dim1, dim0, -1]).transpose(-2,-3)



def get_boundary(X_uv, size):
  
  m = size[0]
  n = size[1]

  return torch.cat([
      X_uv[..., 0:m-1,  0,  :],
      X_uv[..., m-1, 0:n-1, :],
      torch.flip( X_uv[..., 1:m, n-1, :], dims=[-2]),
      torch.flip( X_uv[..., 0, 1:n, :], dims=[-2])
    ],dim=-2)


def flatten_and_permute(X_uv, pq):

  p, q = pq[0], pq[1]

  assert X_uv.shape[-3]==p
  assert X_uv.shape[-2]==q

  X_B = get_boundary(X_uv, size=[p,q])

  X_UB = X_uv[..., 1:p-1, 1:q-1, :].transpose(-3,-2)

  X_UB = torch.reshape(X_UB, X_UB.shape[:-3]+torch.Size([X_UB.shape[-3]*X_UB.shape[-2], X_UB.shape[-1]]))

  return torch.cat([X_B, X_UB], dim=-2)


def fillin_bfs_order(BC, Z, p, q):


  Z_uv = image_unflatten(Z, p-2, q-2)

  dtype = Z.dtype

  if True:
    sz = list(Z_uv.shape)
    sz[-2] = sz[-2] + 2
    sz[-3] = sz[-3] + 2
    X_uv = torch.zeros(size=sz, dtype=dtype)
    X_uv[...,1:-1,1:-1,:] = Z_uv
    if False:
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

  X = image_flatten(X_uv)

  return X, X_uv, Z_uv


def solution_fillin(sol, BC, Z, p, q):
  
  sol.X_p = torch.concat([BC, Z], axis=-2)

  sol.X, sol.X_uv, sol.Z_uv = fillin_bfs_order(BC, Z, p, q)

  return 


def scatter_boundary_eliminate_pre(dim_pq, DtN_ori, RHS_ori, pre_sol=None, cache=True, invf=inv_default):
  '''
  history: see c61d0f9 for old versions---scatter_boundary_eliminate: 
    def scatter_boundary_eliminate_pre_old(dim_pq, DtN_ori, RHS_ori, pre_sol=None, cache=True):
      return *scatter_boundary_eliminate(dim_pq, DtN_ori, RHS_ori), None 
  '''

  class Solution:
    name = 'scatter_boundary_eliminate_pre'


  sol = Solution()

  if cache==False:
    pre_sol = None

  if pre_sol is None:
    DtN = DtN_ori.clone()
  else:
    DtN = None
  RHS = RHS_ori.clone()





  p = dim_pq[0]
  q = dim_pq[1]


  irs = 2*(p+q)-4

  ind = 0
  ka = slice(0,1)
  kb = slice(p-1,None)
  rm = slice(1,p-1)
  if pre_sol is None:
    Block = DtN[:, :, ind]
    invA1 = invf(DtN[:, :, ind, rm, rm])
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
  def back_fill1(nBC):
    ind = 0
    ka = slice(0,1)
    kb = slice(p-1,None)
    rm = slice(1,p-1)
    nBC = nBC.clone()
    ya = nBC[:, :, ind, ka, :]
    yb = nBC[:, :, ind, kb, :]
    nBC[:, :, ind, rm, :] = invA1 @ (uu1 - Ba1 @ ya - Bb1 @ yb)
    return nBC
  if pre_sol is None:
    Block[..., ka, ka] -= Ca1 @ invA1 @ Ba1
    Block[..., ka, kb] -= Ca1 @ invA1 @ Bb1
    Block[..., kb, ka] -= Cb1 @ invA1 @ Ba1
    Block[..., kb, kb] -= Cb1 @ invA1 @ Bb1
  RHS[:, :, ind, ka, :] -=    Ca1 @ (invA1 @ RHS[:, :, ind, rm, :])
  RHS[:, :, ind, kb, :] -=    Cb1 @ (invA1 @ RHS[:, :, ind, rm, :])

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
  def back_fill2(nBC):
    ind = -1
    ka = slice(0, p+q-1)
    kb = slice(2*p+q-3,None)
    rm = slice(p+q-1,2*p+q-3)
    nBC = nBC.clone()
    ya = nBC[:, :, ind, ka, :]
    yb = nBC[:, :, ind, kb, :]
    nBC[:, :, ind, rm, :] = invA2 @ (uu2 - Ba2 @ ya - Bb2 @ yb)
    return nBC
  if pre_sol is None:
    DtN[:, :, ind, ka, ka] -= Ca2 @ invA2 @ Ba2
    DtN[:, :, ind, ka, kb] -= Ca2 @ invA2 @ Bb2
    DtN[:, :, ind, kb, ka] -= Cb2 @ invA2 @ Ba2
    DtN[:, :, ind, kb, kb] -= Cb2 @ invA2 @ Bb2
  RHS[:, :, ind, ka, :] -=    Ca2 @ (invA2 @ RHS[:, :, ind, rm, :])
  RHS[:, :, ind, kb, :] -=    Cb2 @ (invA2 @ RHS[:, :, ind, rm, :])


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
  def back_fill3(nBC):
    ind = -1
    ka = slice(0, p)
    kb = slice(p+q-2,None)
    rm = slice(p,p+q-2)
    nBC = nBC.clone()
    ya = nBC[:, ind, :, ka, :]
    yb = nBC[:, ind, :, kb, :]
    nBC[:, ind, :, rm, :] = invA3 @ (uu3 - Ba3 @ ya - Bb3 @ yb)
    return nBC
  if pre_sol is None:
    Block[..., ka, ka] -= Ca3 @ invA3 @ Ba3
    Block[..., ka, kb] -= Ca3 @ invA3 @ Bb3
    Block[..., kb, ka] -= Cb3 @ invA3 @ Ba3
    Block[..., kb, kb] -= Cb3 @ invA3 @ Bb3
  RHS[:, ind, :, ka, :] -=    Ca3 @ (invA3 @ RHS[:, ind, :, rm, :])
  RHS[:, ind, :, kb, :] -=    Cb3 @ (invA3 @ RHS[:, ind, :, rm, :])

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
  def back_fill4(nBC):
    ind = 0
    ka = slice(0, 1)
    kb = slice(1, 2*p+q-2)
    rm = slice(2*p+q-2, None)
    nBC = nBC.clone()
    ya = nBC[:, ind, :, ka, :]
    yb = nBC[:, ind, :, kb, :]
    nBC[:, ind, :, rm, :] = invA4 @ (uu4 - Ba4 @ ya - Bb4 @ yb)
    return nBC
  if pre_sol is None:  
    Block[..., ka, ka] -= Ca4 @ invA4 @ Ba4
    Block[..., ka, kb] -= Ca4 @ invA4 @ Bb4
    Block[..., kb, ka] -= Cb4 @ invA4 @ Ba4
    Block[..., kb, kb] -= Cb4 @ invA4 @ Bb4
  RHS[:, ind, :, ka, :] -=    Ca4 @ (invA4 @ RHS[:, ind, :, rm, :])
  RHS[:, ind, :, kb, :] -=    Cb4 @ (invA4 @ RHS[:, ind, :, rm, :])

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

  def nBC_back_fill(nBC):
    nBC = back_fill1(back_fill2(back_fill3(back_fill4(nBC))))
    return nBC

  return DtN, RHS, nBC_back_fill, sol


def batch_Dirichlet_solve_pre(dim_pq, dLA, invf=inv_default): 


  p = dim_pq[0]
  q = dim_pq[1]

  irs = 2*(p+q)-4

  dLA_ss = dLA[..., irs:,  irs:]
  dLA_sr = dLA[..., irs:,  0:irs]
  dLA_rr = dLA[..., 0:irs, 0:irs]
  dLA_rs = dLA[..., 0:irs, irs:]

  def split_RHS(RHS):
    RHS_s = RHS[..., irs:, :]
    RHS_r = RHS[..., 0:irs, :]
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

  class Solution:
    name = 'Dirichlet_solve: solution'

  inv_dLA_ss = invf(dLA_ss)

  reused_prod = dLA_rs @ inv_dLA_ss
  DtN = dLA_rr - reused_prod @ dLA_sr

  def get_nRHS(RHS):
    RHS_s, RHS_r = split_RHS(RHS)
    if True:
      nRHS = RHS_r - reused_prod @ RHS_s
    else:
      IRHS = rhs_process(RHS_s)
      nRHS = RHS_r - dLA_rs @ IRHS
    return nRHS

  def solver(BC, RHS, compute_energy=True):
    sol = Solution()

    sol.Y = BC.to_dense()

    RHS_s, RHS_r = split_RHS(RHS)

    if True:
      sol.Z = inv_dLA_ss @ (RHS_s - dLA_sr @ sol.Y)
    else:
      rhs = RHS_s - (dLA_sr @ sol.Y)
      if True:
        sol.Z = inv_dLA_ss @ rhs
      else:
        if False:
          sol.Z = torch.cholesky_solve( rhs, dLA_ss_CL)
        else:
          sol.Z, info = torch.linalg.solve_ex(A=dLA_ss, B=rhs)

    solution_fillin(sol, sol.Y, sol.Z, p=p, q=q)

    if compute_energy:

      def eval_ADirichlet():
        AD = torch.transpose(sol.Y,-2,-1) @ (dLA_rr @ sol.Y + dLA_rs @ sol.Z) 
        assert(AD.shape[-2]==2)
        assert(AD.shape[-1]==2)
        return AD[...,0,0] + AD[...,1,1]

      sol.eval_ADirichlet = eval_ADirichlet 

    return sol

  return DtN, get_nRHS, solver


def merge_blocks_split_value(CX, dim=0, coeff=0.5):
  if dim==0:

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


def bottom_up(XX, value_sum=True): 


  size = XX.shape
  a = size[1]
  b = size[2]
  
  N_iter = math.floor(math.log2(a)+math.log2(b))

  CX = XX

  coeff = 1 if value_sum else 0.5

  for i in range(N_iter):
    if i%2==0:
      CX = merge_blocks_split_value(CX, dim=0, coeff=coeff)
    else:
      CX = merge_blocks_split_value(CX, dim=1, coeff=coeff)

  return CX


def scipy_sparse_top_down(A, 
    ab_wh=None,
    init_shape='not used given ab_wh', 
    div_level='not used given ab_wh',
    value_divide=True, 
    input_x_first=True, 
    permute_bounary_first_order=True, 
    wh_if_permute_bounary_first_order=None,
    beta_shaped=False,
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

  Cs = [[A]]

  for level in range(div_level):


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
        indices = alg.indices_matrix_2d(p, q, 1, 1, 0, 0, x_first=True)

        shared = indices[:, cq].flatten()
        part1 = indices[:, 0:cq+1].transpose().flatten()
        part2 = indices[:, cq:None].transpose().flatten()



        S = Cs[i][j].copy()
        if value_divide: 
          S[np.ix_(shared,shared)] /= 2

        Ns[i][j*2] = S[:,part1][part1,:]
        Ns[i][j*2+1] = S[:,part2][part2,:]

    Cs = Ns
    p = cp
    q = cq+1

    m = len(Cs)
    n = len(Cs[0])
    Ns = [[None for j in range(n)] for i in range(m*2)]
    assert p%2==1
    cp = (p-1)//2
    cq = q
    assert Cs[0][0].shape[0]==p*q

    for i in range(m):
      assert n==len(Cs[i])

      for j in range(n):

        indices = alg.indices_matrix_2d(p, q, 1, 1, 0, 0, x_first=True)
        shared = indices[cp, :].flatten()
        part1 = indices[0:cp+1, :].transpose().flatten()
        part2 = indices[cp:None, :].transpose().flatten()

        S = Cs[i][j].copy()

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


    B = alg.matrix_spiral_bdr_list(w,h)
    UB = alg.matrix_interior_list(w,h)

    ind = torch.tensor(B + UB, dtype=torch.int64)

    DA = DA[...,ind,:][...,:,ind]

  if not beta_shaped:
    DA = DA.reshape( torch.Size([-1]) + DA.shape[-2:])

  return Cs, DA


def alpha_from_scipy_sparse(A_scipy, ab_wh):
  '''
  output: alpha, [batch_size==1, a, b, w*h, w*h]
  '''
  _, alpha = scipy_sparse_top_down(A_scipy, ab_wh=ab_wh, beta_shaped=True)
  alpha = alpha[None, ...]
  return alpha


def scipy_sparse_bottom_up(Cs, shape=None):

  assert(False)

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

      indices = alg.indices_matrix_2d(p, q, m, n, i, j, x_first=True)
      A[np.ix_(indices,indices)] += Cs[i][j]

  return A


def top_down(XX, 
  div_level='Auto',
  value_divide=True,
  ab=None
  ):


  if div_level == 'Auto':
    a = ab[0]
    b = ab[1]
    div_level = (math.log2(a)+math.log2(b))/2
    assert 12==12.0
    div_level = int(div_level)

  CX = XX

  for i in range(div_level):
    cp = CX.shape[3]-1
    cq = CX.shape[4]-1
    assert cp%2==0
    assert cq%2==0

    cp = cp//2
    cq = cq//2


    CX = torch.permute(CX, (0, 5, 1, 3, 2, 4))
    if value_divide:
      CX[..., cq] = CX[..., cq] / 2.0
    CX = CX.unfold(-1, size=cq+1, step=cq)
    CX = CX.flatten(start_dim=4, end_dim=5)
    CX = torch.permute(CX, (0, 2, 4, 3, 5, 1))

    CX = torch.permute(CX, (0, 5, 2, 4, 1, 3))
    if value_divide:
      CX[..., cp] = CX[..., cp] / 2.0
    CX = CX.unfold(-1, size=cp+1, step=cp)
    CX = CX.flatten(start_dim=4, end_dim=5)
    CX = torch.permute(CX, (0, 4, 2, 5, 3, 1))

  return CX


def collapse_subdomains(DtN, RHS, p, q, old_sol, dim=0, pre_sol=None, debug_time=False, invf=None):
  '''
  history: replace the old functions: 
    tip.recursive
    tip.recursive_back_substitute
  '''


  sol = Level()



  if pre_sol is None:
    sol.name = 'Level_num_fact'
  else:
    sol.name = 'Level_back_sub'
    DtN = pre_sol.DtN

  size = DtN.shape
  dtype = DtN.dtype

  batch_size = size[0]
  m = size[1]
  n = size[2]

  c = RHS.shape[-1]


  assert size[-1] == size[-2]
  assert size[-1] == 2 * (p+q)

  if not (pre_sol is None):
    DtN = None
    mDtN = None

  if dim==0:

    assert m%2==0

    if pre_sol is None:
      SS = torch.zeros([batch_size, m//2, n, q-1, q-1], dtype=dtype)
      RR = torch.zeros([batch_size, m//2, n, 2*(2*p+q), 2*(2*p+q)], dtype=dtype)

      SR = torch.zeros([batch_size, m//2, n, q-1, 2*(2*p+q)], dtype=dtype)
      RS = torch.zeros([batch_size, m//2, n, 2*(2*p+q), q-1], dtype=dtype)

      DtNa = DtN[..., 0::2, :, :, :]
      DtNb = DtN[..., 1::2, :, :, :]

    fS = torch.zeros([batch_size, m//2, n, q-1, c], dtype=dtype)
    fR = torch.zeros([batch_size, m//2, n, 2*(2*p+q), c], dtype=dtype)

    RHSa = RHS[..., 0::2, :, :, :]
    RHSb = RHS[..., 1::2, :, :, :]

    m0 = 0
    m1 = p
    m2 = 2*p
    m3 = 2*p+q
    m4 = 3*p+q
    m5 = 4*p+q



    ai = slice(0,p+1)
    aj = slice(p+1,p+q)
    ak = slice(p+q,None) 

    bu = slice(0, 2*p+q+1)
    bv = slice(2*p+q+1, None)

    ri = slice(0,m1+1)
    ru = slice(m1, m4+1)
    rk = slice(m4, None)


    if pre_sol is None:
      SS += DtNa[..., aj, aj] + df( DtNb[..., bv, bv], [False, False])

      RR[..., ri, ri] += DtNa[..., ai, ai]
      RR[..., rk, rk] += DtNa[..., ak, ak]
      RR[..., ri, rk] += DtNa[..., ai, ak]
      RR[..., rk, ri] += DtNa[..., ak, ai]

      RS[..., ri, :]  += DtNa[..., ai, aj]
      RS[..., rk, :]  += DtNa[..., ak, aj]
      SR[..., :, ri]  += DtNa[..., aj, ai]
      SR[..., :, rk]  += DtNa[..., aj, ak]

      RR[..., ru, ru] += DtNb[..., bu, bu]
      RS[..., ru, :] += df( DtNb[..., bu, bv], [True, False])
      SR[..., :, ru] += df( DtNb[..., bv, bu], [False, True])

    fR[..., ri, :] += RHSa[..., ai, :]
    fR[..., rk, :] += RHSa[..., ak, :]

    fS[..., :, :] += RHSa[..., aj, :]

    fR[..., ru, :] += RHSb[..., bu, :]
    fS[..., :, :]  += cf( RHSb[..., bv, :], False)

    if pre_sol is None:
      mDtN = torch.zeros([batch_size, m//2, n, 4*p+2*q, 4*p+2*q], dtype=dtype)

    mRHS = torch.zeros(torch.Size([batch_size, m//2, n, 4*p+2*q]) + RHS.shape[-1:], dtype=dtype)

  else:

    assert dim==1

    assert n%2==0

    if pre_sol is None:

      SS = torch.zeros([batch_size, m, n//2, p-1, p-1], dtype=dtype)
      RR = torch.zeros([batch_size, m, n//2, 2*(p+2*q), 2*(p+2*q)], dtype=dtype)

      SR = torch.zeros([batch_size, m, n//2, p-1, 2*(p+2*q)], dtype=dtype)
      RS = torch.zeros([batch_size, m, n//2, 2*(p+2*q), p-1], dtype=dtype)

      DtNa = DtN[..., :, 0::2, :, :]
      DtNb = DtN[..., :, 1::2, :, :]


    fS = torch.zeros([batch_size, m, n//2, p-1, c], dtype=dtype)
    fR = torch.zeros([batch_size, m, n//2, 2*(p+2*q), c], dtype=dtype)

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



    ai = slice(0,p+q+1)
    aj = slice(p+q+1,2*p+q)
    ak = slice(2*p+q,None) 

    bu = slice(0, 1)
    bv = slice(1, p)
    bw = slice(p, None)

    ri = slice(0,m2+1)
    rw = slice(m2, m5)
    ru = slice(m5, m5+1)
    rk = slice(m5, None)



    if pre_sol is None:

      SS += DtNa[..., aj, aj] + df( DtNb[..., bv, bv], [False, False])

      RR[..., ri, ri] += DtNa[..., ai, ai]
      RR[..., rk, rk] += DtNa[..., ak, ak]
      RR[..., ri, rk] += DtNa[..., ai, ak]
      RR[..., rk, ri] += DtNa[..., ak, ai]

      RS[..., ri, :]  += DtNa[..., ai, aj]
      RS[..., rk, :]  += DtNa[..., ak, aj]
      SR[..., :, ri]  += DtNa[..., aj, ai]
      SR[..., :, rk]  += DtNa[..., aj, ak]

      RR[..., ru, ru] += DtNb[..., bu, bu]
      RR[..., rw, rw] += DtNb[..., bw, bw]
      RR[..., ru, rw] += DtNb[..., bu, bw]
      RR[..., rw, ru] += DtNb[..., bw, bu]
      RS[..., ru, :] += df( DtNb[..., bu, bv], [True, False])
      SR[..., :, ru] += df( DtNb[..., bv, bu], [False, True])
      RS[..., rw, :] += df( DtNb[..., bw, bv], [True, False])
      SR[..., :, rw] += df( DtNb[..., bv, bw], [False, True])


    fR[..., ri, :] += RHSa[..., ai, :]
    fR[..., rk, :] += RHSa[..., ak, :]

    fS[..., :, :] += RHSa[..., aj, :]

    fR[..., ru, :] += RHSb[..., bu, :]
    fR[..., rw, :] += RHSb[..., bw, :]
    fS[..., :, :]  += cf( RHSb[..., bv, :], False)

    if debug_time:
      torch.cuda.synchronize()
      print(f'matrix slicer time: {(time.time()-start_time)*1000} ms')

    if pre_sol is None:
      mDtN = torch.zeros([batch_size, m, n//2, 2*p+4*q, 2*p+4*q], dtype=dtype)

    mRHS = torch.zeros(torch.Size([batch_size, m, n//2, 2*p+4*q]) + RHS.shape[-1:], dtype=dtype)

  if debug_time:
    torch.cuda.synchronize()
    start_time = time.time()

  if pre_sol is None:
    if True:

      if invf is None:
        invf = lambda xx : torch.linalg.inv_ex(xx)[0]
      size = SS.shape 
      assert len(size)==5
      invSS = invf(SS)
      
      invSSmSR = invSS @ SR
      mDtN += RR - RS @ invSSmSR
      def update_mRHS(ffR, ffS):
        return ffR - RS @ (invSS @ ffS)
      
    else:
      assert False

      if False:
        LL, info = torch.linalg.cholesky_ex(SS, upper=True)
        mDtN += RR - RS @ torch.cholesky_solve(SR, LL)
        mRHS += fR - RS @ torch.cholesky_solve(fS, LL)

      else:
        LL, info = torch.linalg.cholesky_ex(SS, upper=True)
        size = SS.shape 
        assert len(size)==5
        batch_eye = torch.eye(size[-1])[None, None, None, ...]
        invSS = torch.cholesky_solve( batch_eye , LL) 
        invSSmSR = invSS @ SR
        mDtN += RR - RS @ invSSmSR
        mRHS += fR - RS @ (invSS @ fS)

  else:
    update_mRHS = pre_sol.update_mRHS

  mRHS += update_mRHS(fR, fS)
  sol.update_mRHS = update_mRHS

  if debug_time:
    torch.cuda.synchronize()
    print(f'linear solve time: {(time.time()-start_time)*1000} ms')

  sol.DtN = DtN

  sol.batch_size = batch_size
  sol.p = p
  sol.q = q
  sol.m = m
  sol.n = n

  if pre_sol is None:
    sol.invSS = invSS
    sol.invSSmSR = invSSmSR
    def update_uS(ffS, uuR):
      return invSS @ ffS - invSSmSR @ uuR
  else:
    sol.invSS = pre_sol.invSS
    sol.invSSmSR = pre_sol.invSSmSR
    update_uS = pre_sol.update_uS

  sol.update_uS = update_uS

  if dim==0:
    def topdown_solve_Dirichlet(BC):
      c = BC.shape[-1]
      nBC = torch.zeros([batch_size, m, n, 2*(p+q), c], dtype=dtype)
      uR = BC
      uS = update_uS(fS, uR)
      nBC[..., 0::2, :, ai, :] = uR[..., ri, :]
      nBC[..., 0::2, :, ak, :] = uR[..., rk, :]
      nBC[..., 0::2, :, aj, :] = uS[..., :, :]
      nBC[..., 1::2, :, bu, :] = uR[..., ru, :]
      nBC[..., 1::2, :, bv, :] = cf(uS[..., :, :], False)
      return nBC
    
  else:
    def topdown_solve_Dirichlet(BC):
      c = BC.shape[-1]
      nBC = torch.zeros([batch_size, m, n, 2*(p+q), c], dtype=dtype)
      uR = BC
      uS = update_uS(fS, uR)
      nBC[..., :, 0::2, ai, :] = uR[..., ri, :]
      nBC[..., :, 0::2, ak, :] = uR[..., rk, :]
      nBC[..., :, 0::2, aj, :] = uS[..., :, :]
      nBC[..., :, 1::2, bu, :] = uR[..., ru, :]
      nBC[..., :, 1::2, bw, :] = uR[..., rw, :]
      nBC[..., :, 1::2, bv, :] = cf(uS[..., :, :], False)
      return nBC
    
  sol.topdown_solve_Dirichlet = topdown_solve_Dirichlet    

  '''
  level.mDtN
  level.mRHS
  level.mp = 2*p
  level.mq = q

  return level
  '''

  if dim==0:
    return mDtN, mRHS, 2*p, q, sol
  else:
    return mDtN, mRHS, p, 2*q, sol


def topdown_solve(BC, sol):

  c = BC.shape[-1]

  batch_size = sol.batch_size
  p = sol.p
  q = sol.q
  m = sol.m
  n = sol.n

  nBC = torch.zeros([batch_size, m, n, 2*(p+q), c])



  ai = slice(0,p+1)
  aj = slice(p+1,p+q)
  ak = slice(p+q,None) 

  bu = slice(0, 2*p+q+1)
  bv = slice(2*p+q+1, None)

  ri = slice(0,m1+1)
  ru = slice(m1, m4+1)
  rk = slice(m4, None)


  uR = BC
  uS = sol.invSS @ sol.mRHS - invSSmSR @ uR

  nBC[..., 0::2, :, ai, :] = uR[..., ri, :]
  nBC[..., 0::2, :, ak, :] = uR[..., rk, :]
  nBC[..., 0::2, :, aj, :] = uS[..., :, :]

  nBC[..., 1::2, :, bu, :] = uR[..., ru, :]
  nBC[..., 1::2, :, bv, :] = cf(uS[..., :, :], False)

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


  va = 0
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


def apply_all_schur_steps(DtN, RHS, BC, m, n, ps, qs, debug=False, layers_pre=None, lastInvDtN_reuse=None, invf=inv_default):
  '''
  history: replace the old functions: 
    tip.routine_ps_condense
    tip.routine_ps_back_substitute
  '''
  

  if True:
    
    layers = [(DtN, RHS, ps, qs, [])]

    N_iter = math.floor(math.log2(m)+math.log2(n))
    
    for i in range(N_iter): 

      dim = 0 if i%2==0 else 1

      if debug:
        import time
        start_time = time.time()


      if layers_pre is None:
        DtN_new, RHS_new, ps_new, qs_new, sol = collapse_subdomains(*layers[i], dim=dim, pre_sol=None)
      else:
        DtN_new, RHS_new, ps_new, qs_new, sol = collapse_subdomains(*layers[i], dim=dim, pre_sol=layers_pre[i+1][-1])

      if debug:
        print(f'routine_ps_condense: {(ps_new,qs_new)} time[{i}]: {(time.time()-start_time)*1000} ms')

      sol.name += str(i)

      layers.append([DtN_new, RHS_new, ps_new, qs_new, sol])

    lastInvDtN = None

    if BC is None:
      assert False
    if BC=='Neumann-scatter':
      if layers_pre is None:
        lhs = layers[-1][0]
      else:
        lhs = layers_pre[-1][0]
      rhs = layers[-1][1]
      assert ps==qs
      ss = slice(0,None,ps)
      BC = torch.zeros(lhs.shape[:-1]+rhs.shape[-1:])
      if layers_pre is None:
        lastInvDtN = invf(lhs[...,ss,ss])
      else:
        lastInvDtN = lastInvDtN_reuse
      BC[...,ss,:] = lastInvDtN @ rhs[...,ss,:]
    elif BC=='Neumann' or BC=='Neumann-full':
      if layers_pre is None:
        lhs = layers[-1][0]
      else:
        lhs = layers_pre[-1][0]
      rhs = layers[-1][1]
      if layers_pre is None:
        lastInvDtN = invf(lhs)
      else:
        lastInvDtN = lastInvDtN_reuse
      BC = lastInvDtN @ rhs
    elif BC=='midpoint-reflective':
      F_reduce = midpoint_reflective_reduce(m, n, ps, qs)
      if layers_pre is None:
        lhs = layers[-1][0]
      else:
        lhs = layers_pre[-1][0]
      rhs = layers[-1][1]
      lhs = F_reduce.transpose(-1,-2) @ lhs @ F_reduce
      rhs = F_reduce.transpose(-1,-2) @ rhs
      if layers_pre is None:
        lastInvDtN = invf(lhs)
      else:
        lastInvDtN = lastInvDtN_reuse
      BC = lastInvDtN @ rhs
      BC = F_reduce @ BC

    nBC = BC
    for i in range(N_iter):

      sol = layers[-(i+1)][-1]
      nBC = sol.topdown_solve_Dirichlet(nBC)


  if layers_pre is None:
    size = layers[-1][0].shape
    assert size[1]==1 or size[2]==1

  return nBC, layers, lastInvDtN


def Schwarz_Schur_involution(Alpha, Beta, BC, wh, debug=False, prefact_sol=None):

  """
  Inputs
  ------
  Alpha : {torch.tensor, None}
    ``Alpha`` should be ``None`` if given ``prefact_sol``
  input: 
    Alpha: [batch_size, a, b, (w*h), (w*h)]     or None 
     Beta: [batch_size, a, b, (w*h), channels]

  History
  -------
  replace the old functions: 
  >>> import xxx as tip
  >>> tip.condense_general_solve
  >>> tip.condense_back_substitute
  """

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

  RHS = Beta

  if BC is None:
    print('warning: use Neumann for BC==None')
    BC = 'Neumann'

  if BC=='Neumann':
    BC = 'Neumann-scatter'

  
  if debug:
    start_time = time.time()

  merge_Bab = False
  
  if prefact_sol is None:

    if not merge_Bab:
      
      DtN, get_nRHS_step1, batch_Dirichlet_solver = batch_Dirichlet_solve_pre(dim_pq=(p,q), dLA=Alpha)  

      def get_nRHS(RHS):
        nRHS = get_nRHS_step1(RHS)
        return nRHS

    else:

      Alpha = torch.reshape(Alpha, torch.Size([batch_size*a*b]) + Alpha.shape[-2:])
      DtN, get_nRHS_step1, batch_Dirichlet_solver = batch_Dirichlet_solve_pre(dim_pq=(p,q), dLA=Alpha)  
      
      DtN = torch.reshape(DtN, torch.Size([batch_size, a, b]) + DtN.shape[-2:])

      def get_nRHS(RHS):
        tmp_RHS = torch.reshape(RHS, [batch_size*a*b] + list(RHS.shape[-2:]))
        nRHS = get_nRHS_step1(tmp_RHS)
        nRHS = torch.reshape(nRHS, torch.Size([batch_size, a, b]) + nRHS.shape[-2:])
        return nRHS

    class Solution:
      name = 'Schwarz_Schur_involution'

  else:

    class Solution:
      name = 'Schwarz_Schur_involution_back_sub'

    DtN = prefact_sol.DtN

    get_nRHS = prefact_sol.get_nRHS

    batch_Dirichlet_solver = prefact_sol.batch_Dirichlet_solver

  if debug:
    print(f'DtN solve time: {(time.time()-start_time)*1000} ms')

  sol = Solution()

  nRHS = get_nRHS(RHS)
  if not (BC=='Neumann-full' or BC=='Neumann-scatter' or BC=='midpoint-reflective'):
    assert nRHS.shape == (DtN.shape[:-1] + BC.shape[-1:])


  if BC=='Neumann-scatter':
    if prefact_sol is None:
      sDtN, sRHS, nBC_back_fill, sol_scatter = scatter_boundary_eliminate_pre((p,q), DtN, nRHS)
    else:
      sDtN = prefact_sol.sDtN
      _, sRHS, nBC_back_fill, sol_scatter = scatter_boundary_eliminate_pre((p,q), DtN, nRHS, pre_sol=prefact_sol.sol_scatter)      
  else:
    sDtN, sRHS = DtN, nRHS
    sol_scatter = None
  

  if prefact_sol is None:
    sBC, layers, lastInvDtN = apply_all_schur_steps(sDtN, sRHS, BC, a, b, p-1, q-1, debug=debug)
  else:
    if True:
      sBC, layers, lastInvDtN = apply_all_schur_steps(sDtN, sRHS, BC, a, b, p-1, q-1, debug=debug, 
        layers_pre=prefact_sol.layers, lastInvDtN_reuse=prefact_sol.lastInvDtN)
    else:
      print('warning: this has unsolved bugs.')
      layers = prefact_sol.layers
      lastInvDtN = prefact_sol.lastInvDtN
      sBC = routine_ps_back_substitute(layers, sRHS, BC, a, b, p-1, q-1, lastInvDtN, debug)


  if BC=='Neumann-scatter':
    nBC = nBC_back_fill(sBC)
  else:
    nBC = sBC

  if debug:
    start_time = time.time()


  if not merge_Bab:
    sol_blockwise = batch_Dirichlet_solver(BC=nBC, RHS=RHS, compute_energy=True)
  else:
    sol_blockwise = batch_Dirichlet_solver(BC=torch.reshape(nBC, [batch_size * a * b, 2*(p+q-2), nBC.shape[-1]]), 
      RHS=torch.reshape(RHS, [batch_size * a * b] + list(RHS.shape[-2:])), 
      compute_energy=True)

  if debug:
    print(f'post solve time: {(time.time()-start_time)*1000} ms'); 

  X_uv = bottom_up(torch.reshape(sol_blockwise.X_uv, [batch_size, a, b, p, q, -1]), value_sum=False)
  sol.X_uv = X_uv[:,0,0,...]

  sol.batch_Dirichlet_solver = batch_Dirichlet_solver
  sol.get_nRHS = get_nRHS
  sol.DtN = DtN
  sol.lastInvDtN = lastInvDtN
  sol.ab_wh = ((a,b), (p,q))
  sol.batch_size = batch_size
  sol.dtype = dtype
  sol.layers = layers
  sol.sol_scatter = sol_scatter
  sol.sDtN = sDtN

  sol.BC = BC
  sol.Beta = Beta

  sol.X = torch.reshape(sol.X_uv.clone().transpose(-2,-3), [batch_size,-1, sol.X_uv.shape[-1]]) 
  sol.sol_blockwise = sol_blockwise


  return sol


def uniform_lap_patch(p, q):

  import pyamg
  import time
  import math

  sten = np.array([[ 0., -1., -0.], [-1.,  4., -1.], [-0., -1.,  0.]])

  print(f'sten:{sten}')

  A = pyamg.gallery.stencil_grid(sten, (p, q))

  d = A.shape[0]

  DA = A.toarray()

  np.fill_diagonal(DA, 0)

  diag = np.sum(DA, axis=1)

  np.fill_diagonal(DA, -diag)

  print(np.linalg.norm(DA @ np.ones(d)))

  return DA


def dense_patchwise_laplacian_rw(images):

  assert(False)


  batch_dims = len(images.shape) - 3

  batch_sizes = list(images.shape[0:batch_dims])

  c = images.shape[-1]
  h = images.shape[-2]
  w = images.shape[-3]

  dL_unflatten = torch.zeros(batch_sizes + [w,h,w,h])

  dL_unflatten 

  images[...,:,:,:]


def beta_topdown(beta_abuv, ab_wh, value_divide=True):


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

  mRHS = flatten_and_permute(RHS_bdr, (w, h))

  return mRHS


def beta_from_numpy(numpyRHS, ab_wh, dtype=None):

  if dtype is None:
    dtype = torch.get_default_dtype()
  
  a,b,w,h,W,H = unpack_dims(ab_wh)

  assert(len(numpyRHS.shape)==2)

  RHS_f = torch.tensor(numpyRHS, dtype=dtype)[None,...] 

  RHS_abuv = image_unflatten(RHS_f, W, H)[:,None,None,...]

  beta = beta_topdown(RHS_abuv, ab_wh=ab_wh)


  return beta


def torch_batch_cartesian_prod(A, B):
  assert A.shape[-1]==1
  assert B.shape[-1]==1
  dims = len(A.shape[:-2])
  assert dims == len(B.shape[:-2])
  a = A.shape[-2]
  b = B.shape[-2]
  BT = B.transpose(-1, -2)
  return A.expand([-1]*dims + [-1] + [b]), BT.expand([-1]*dims + [a] + [-1])


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
  

  FX = grid_x[None,None,None,...,None]
  FY = grid_y[None,None,None,...,None]


  FXY = torch.cat([FX,FY], dim=-1)

  FXY = top_down(FXY, ab=(a,b), value_divide=False)
  FXY = FXY[0]
  if flattened:
    FXY = torch.reshape(FXY.transpose(-2,-3), FXY.shape[:-3] + torch.Size([-1]) + FXY.shape[-1:])

  return FXY


def patch_adjacency_mask(wh, nebr):
  XY = mesh_hier(((1,1), wh))[0,0] * 1.0 
  if nebr=='5pt':
    adj_mask = (torch.cdist(XY, XY, p=2)**2) <= 1.01
  else:
    assert nebr=='9pt'
    adj_mask = (torch.cdist(XY, XY, p=2)**2) <= 2.01
  return adj_mask


def torch_bottom_up_sparse_indices(ab_wh):

  a,b,w,h,W,H = unpack_dims(ab_wh)

  FXY = mesh_hier(ab_wh=ab_wh, flattened=True)

  FX = FXY[...,0:1]
  FY = FXY[...,1:2]


  FD = FX + FY * W

  IX, IY = torch_batch_cartesian_prod(FD, FD)


  return IX, IY


def torch_bottom_up_sparse_VIJ(DA, wh, numpy_out=False, adj='all'):


  w = wh[0]
  h = wh[1]

  a = DA.shape[-4]
  b = DA.shape[-3]

  IV = DA

  IX, IY = torch_bottom_up_sparse_indices(ab_wh=((a,b),(w,h)))

  if adj=='9pt' or adj=='5pt':
    adj_mask = patch_adjacency_mask(wh=(w,h), nebr='9pt')
    ii, jj = torch.where(adj_mask)

    if True:
      ii2, jj2 = torch.where(torch.logical_not(adj_mask))
      removed = IV[..., ii2, jj2]
      assert 0.0==removed.abs().sum()

    IX = IX[..., ii, jj]
    IY = IY[..., ii, jj]
    IV = IV[..., ii, jj]

  else:
    assert adj=='all'

  if numpy_out:
    return IV.detach().cpu().numpy(), IX.detach().cpu().numpy(), IY.detach().cpu().numpy()
  else:
    return IV, IX, IY


def torch_bottom_up_sparse_scipy(DA, ab_wh, shape=None, boundary_first_order=True, adj='all'):
  """ 
  adj {'all', '9pt', '5pt'}
  """
  import scipy

  a,b,w,h,W,H = unpack_dims(ab_wh)

  DA_abxx = torch.reshape(DA, [a,b]+list(DA.shape[-2:]))


  if boundary_first_order:
    

    B, UB = alg.matrix_bdr_interior_split(wh=(w,h))
    invBUB = alg.inverse_permutation(B+UB)


    invd = torch.tensor(invBUB, dtype=torch.int64)

    DA_abxx = DA_abxx[...,invd,:][...,:,invd]

  IV, IX, IY = torch_bottom_up_sparse_VIJ(DA_abxx, wh=(w,h), numpy_out=True, adj=adj)

  N = W * H
  A_scipy = scipy.sparse.csr_matrix((IV.flatten(), (IX.flatten(), IY.flatten())), shape=(N, N))

  return A_scipy


def flattenWH(X):
  return torch.reshape(X.transpose(-2,-3), X.shape[:-3] + torch.Size([-1]) + X.shape[-1:])


def patch_laplacian_adjust_boundary(alpha, ab_wh):

  
  a,b,w,h,W,H = unpack_dims(ab_wh)

  assert w==5
  assert h==5
  bottom = [0,1,2,3,4]
  left = [20,15,10,5,0]
  right = [4,9,14,19,24]
  top = [24,23,22,21,20] 

  left = torch.tensor(left, dtype=torch.int64)
  right = torch.tensor(right, dtype=torch.int64)
  bottom = torch.tensor(bottom, dtype=torch.int64)
  top = torch.tensor(top, dtype=torch.int64)
  

  rr, cc = torch.meshgrid(left, left, indexing='ij')
  alpha[..., 1:, :, rr, cc] *= 0.5
  rr, cc = torch.meshgrid(right, right, indexing='ij')
  alpha[..., :-1, :, rr, cc] *= 0.5
  rr, cc = torch.meshgrid(bottom, bottom, indexing='ij')
  alpha[..., :, 1:, rr, cc] *= 0.5
  rr, cc = torch.meshgrid(top, top, indexing='ij')
  alpha[..., :, :-1, rr, cc] *= 0.5

  return


def convert_to_boundary_first_ordering(DA, wh):
  '''
  spiral ordering for boundary nodes, and [w-2, h-2] array ordering for interior nodes.
  '''

  PB, PUB = alg.matrix_bdr_interior_split(wh=wh)


  ind = torch.tensor(PB + PUB, dtype=torch.int64)


  PDA = DA[...,ind,:][...,:,ind]

  return PDA


def batch_dense_laplacian_whwh(X):
  assert False
  return


def single_patch_uniform_laplacian(wh):

  w, h = wh[0], wh[1]

  batch_size = 1
  n = w*h
  f = (w-1)*(h-1) * 4

  if True: 
    
    import mesh_processing as mgp
    mesh = mgp.MeshGrid(w, h)
    np_V, np_F = mesh.V, mesh.F
    II, JJ, _, _ = mgp.lap_entries(np_V, np_F, multicol=False)
    assert np_F.shape[0]==f
    assert np_V.shape[0]==n

    adjs = torch.tensor(np.stack([II,JJ]), dtype=torch.int64, requires_grad=False)

    au = torch.concat([
          torch.ones([batch_size, f, 1]), 
          torch.zeros([batch_size, f, 1]),
          torch.ones([batch_size, f, 1])
        ],axis=-1)


    np_gn_p = mgp.grad_normalized(np_V, np_F)
    gn_p = torch.tensor(np_gn_p, requires_grad=False)

    print('au', au.shape)

    import torch_geometry_processing as tgp
    values = tgp.assemble_lap_values(au, gn_p)

  spLA = torch.sparse_coo_tensor( 
                indices=adjs, 
                values=values, 
                size=[n, n, batch_size])
                
  dLA = spLA.to_dense().permute(2,0,1)

  return dLA[0]


def batch_diag(XX):
  assert XX.shape[-1]==1
  n = XX.shape[-2]

  RR = torch.zeros(list(XX.shape[:-2]) + [n,n])

  RR[..., np.arange(n), np.arange(n)] = XX[..., 0]

  return RR


def alpha_from_patchwise_features(CX):
  
  _, a, b, w, h, c = list(CX.shape)

  assert c==1

  if True:

    import mesh_processing as mgp
    mesh = mgp.MeshGrid(w, h)
    Gx = torch.tensor(mesh.Gx.todense())[None, None, None]
    Gy = torch.tensor(mesh.Gy.todense())[None, None, None]
    F  = torch.tensor(mesh.F, dtype=torch.int64)

  C = flattenWH(CX)

  CF = (C[...,F[:,0],:] + C[...,F[:,1],:] + C[...,F[:,2],:]) / 6.0

  CF = batch_diag(CF) 


  LA = Gx.transpose(-1,-2) @ CF @ Gx + Gy.transpose(-1,-2) @ CF @ Gy

  alpha = LA

  return alpha
  

def single_patch_mass_diagonal(wh):

  w, h = wh[0], wh[1]

  M = torch.eye(w*h)
  m = 2 * torch.ones([w,h])

  m[..., [0,-1], :] = m[..., [0,-1], :] * 0.5
  m[..., :, [0,-1]] = m[..., :, [0,-1]] * 0.5

  ind = torch.arange(w*h)

  M[...,ind,ind] = m.transpose(-1,-2).flatten()

  return M


def alpha_uniform_laplacian(ab_wh): 

  a,b,w,h,W,H = unpack_dims(ab_wh)
  alpha_shape = [1,a,b,w*h,w*h]
  alpha = torch.zeros(alpha_shape)

  alpha[...,:,:] = single_patch_uniform_laplacian([w,h])

  alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])

  return alpha


def alpha_coeff_laplacian(image_hwc, ab_wh):


  image_whc = image_hwc[None,None,None].transpose(3,4)

  a,b,w,h,W,H = unpack_dims(ab_wh)

  CX = top_down(image_whc, div_level='Auto', value_divide=False, ab=ab_wh[0])


  alpha = alpha_from_patchwise_features(CX)

  alpha_shape = [1,a,b,w*h,w*h]
  assert(list(alpha.shape)==alpha_shape)

  adj_mask = patch_adjacency_mask([w,h], nebr='9pt')

  alpha = alpha * (1.0*adj_mask)

  alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
  
  return alpha


def alpha_uniform_mass_diagonal(ab_wh):

  a,b,w,h,W,H = unpack_dims(ab_wh)
  alpha_shape = [1,a,b,w*h,w*h]
  alpha = torch.zeros(alpha_shape)

  alpha[...,:,:] = single_patch_mass_diagonal([w,h])

  alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])

  return alpha


def batch_dense_affinity_image(X, wh, params=None):


  w, h = wh[0], wh[1]

  shape = X.shape
  c = shape[-1]
  assert shape[-2]==(w*h)
  BS = shape[-2:]

  D = torch.cdist(X, X)


  sigma = 0.0
  if hasattr(params,'sigma'):
    sigma = params.sigma
  sigma = torch.tensor(sigma)
  
  Weights = torch.maximum(torch.exp(-900.0 * D*D), sigma)

  return Weights


def alpha_image_laplacian(
  image, 
  ab_wh, 
  diag_zero_sum=True, 
  adjust_border_patches=False, 
  params=None
  ):
  '''
  Parameters
  ----------
  adjust_border_patches should be 
    Flase if using FEM, or 
    True only for the purpose of being consistent with external code like pymatting. 
  '''


  image_whc = image[None,None,None].transpose(3,4)

  a,b,w,h,W,H = unpack_dims(ab_wh)

  CX = top_down(image_whc, div_level='Auto', value_divide=False, ab=ab_wh[0])


  FCX = flattenWH(CX)

  alpha = batch_dense_affinity_image(FCX, ab_wh[1], params=params)
  alpha_shape = [1,a,b,w*h,w*h]
  assert(list(alpha.shape)==alpha_shape)

  adj_mask = patch_adjacency_mask([w,h], nebr='5pt' if hasattr(params,'5pt') else '9pt')

  if hasattr(params,'random_sparse_patch'):
    alpha = torch.rand(alpha_shape) - 0.5

  alpha = alpha * (1.0*adj_mask)

  if hasattr(params, 'random_dense_patch'):
    alpha = torch.rand(alpha_shape) - 0.5

  if hasattr(params, 'eye'):
    alpha = torch.zeros(alpha_shape)
    ind = torch.arange(w*h)
    alpha[...,ind,ind] = 1

    diag_zero_sum = False
    assert adjust_border_patches==True



  if adjust_border_patches:
    patch_laplacian_adjust_boundary(alpha, ab_wh)

  if diag_zero_sum:
    alpha = - alpha
    ind = torch.arange(w*h)
    alpha[...,ind,ind] = 0
    alpha[...,ind,ind] = - alpha.sum([-1])


  alpha = convert_to_boundary_first_ordering(alpha, wh=ab_wh[1])
  
  return alpha


def test_alpha_image_laplacian(image_numpy, ab_wh=((64,64),(5,5))):

  import image_processing as ip
  

  a,b,w,h,W,H = unpack_dims(ab_wh)

  image = torch.tensor(image_numpy, dtype=torch.get_default_dtype())

  alpha = alpha_image_laplacian(image, ab_wh, diag_zero_sum=False, adjust_border_patches=True)


  A_scipy = torch_bottom_up_sparse_scipy(alpha[0], ab_wh=ab_wh)



  ind = torch.tensor([1,2,3,6,7,8], dtype=torch.int64)

  alpha[...,ind][...,ind,:].shape

  print(A_scipy.shape)

  A_scipy_ref = ip.random_walk_affinity(image_numpy) 

  import math
  div_level = int((math.log2(a)+math.log2(b))/2)
  _, alpha_ref = scipy_sparse_top_down(A_scipy_ref, init_shape=(W,H), div_level=div_level, 
                                          wh_if_permute_bounary_first_order=(w,h))
  alpha_ref = alpha_ref[None,...]

  A_scipy_ref2 = torch_bottom_up_sparse_scipy(alpha_ref[0], ab_wh=ab_wh)

  print('errors:',\
    np.abs(A_scipy_ref - A_scipy).max(),\
    np.abs(A_scipy_ref - A_scipy).max(),\
    np.abs(A_scipy_ref - A_scipy_ref2).max()
  )

  print('errors:',\
    (alpha - alpha_ref).abs().max(),\
    (alpha - alpha_ref).norm()\
  )

  return


def torch_bottom_up_scipy_sparse_slow(Cs, wh, shape=None):


  import scipy
  pwh = Cs[0][0].shape[0]
  assert pwh==Cs[0][0].shape[1]

  w = wh[0]
  h = wh[1]

  assert pwh == (w*h)

  is_torch = False
  if torch.is_tensor(Cs):
    m = Cs.shape[0]
    n = Cs.shape[1]
    is_torch = True
    Cs = Cs.detach().cpu().numpy()
  else:
    m = len(Cs)
    n = len(Cs[0])

  if shape is None:
    N = (m*(w-1)+1) * (n*(h-1)+1)
    shape = (N, N)

  B, UB = alg.matrix_bdr_interior_split(wh=(w,h))
  invBUB = alg.inverse_permutation(B+UB)

  print(invBUB)


  if True:
    A = scipy.sparse.csr_matrix(([],([],[])), shape=shape)
    for i in range(m):
      for j in range(n):
        indices = alg.indices_matrix_2d(w, h, m, n, i, j, x_first=True)
        indices = indices.transpose().flatten()
        if i==0 and j==0:
          print('indices:', indices)
        R = Cs[i][j][np.ix_(invBUB,invBUB)]
        A[np.ix_(indices,indices)] += R
  else:
    assert False

  return A

