
import torch
import torch.nn as nn

# do not load mesh_processing.py in this file. 

def cross2d(a, b):
  
  # vectorize: aa(0)*bb(1) - aa(1)*bb(0) # but keep the last dimension of size 1. 
  assert a.shape[-1]==2
  assert b.shape==a.shape

  return a[...,0:1] * b[...,1:2] - a[...,1:2] * b[...,0:1] 


def doublearea(V, F):

  # https://github.com/alecjacobson/gptoolbox/blob/master/mesh/doublearea.m

  assert F.shape[-1]==3
  assert len(F.shape)==2

  assert V.shape[-1]==2 # only pure 2D case has been implemented. 
  r = V[F[:,0],:] - V[F[:,2],:]
  s = V[F[:,1],:] - V[F[:,2],:]
  dblA = cross2d(r, s)

  # consistent with libigl. 
  return dblA[..., 0]


########### keep functions in this block same as mesh_processing.py ###########
# see also mesh_processing.py

def inner_prod(x, y):
  # note it keeps the last dimension of size 1. 
  return torch.sum(x*y, dim=-1, keepdim=True)


def grad_core_2d(V0, V1, V2):
  
  # implement the matlab function: 
  # r = bsxfun(@times,v10, sum(v20.*v20, 2)./ns12) ...
  #   + bsxfun(@times,v20, sum(v10.*v10, 2)./ns12) ...
  #   - bsxfun(@times,v10 + v20, dot12./ns12);

  v20 = V2 - V0
  v10 = V1 - V0
          
  dot12 = inner_prod(v20, v10) # note np.inner does not do this. 

  # aa(0)*bb(1) - aa(1)*bb(0)
  # cross(v20, v10)

  c21 = cross2d(v20, v10)

  ns12 = c21 * c21

  r = (v10 * inner_prod(v20, v20) + v20 * inner_prod(v10, v10)) / ns12 - (v10+v20) * dot12 / ns12

  return r


def grad_area_vectorized(V, F):

  V0 = V[F[:,0], :]
  V1 = V[F[:,1], :]
  V2 = V[F[:,2], :]

  g0 = grad_core_2d(V0,V1,V2)
  g1 = grad_core_2d(V1,V2,V0)
  g2 = grad_core_2d(V2,V0,V1)

  # [3 x f x 2], and [f]
  return torch.stack([g0,g1,g2], dim=0), doublearea(V,F)/2


def grad_normalized(V, F):

  g_all, area = grad_area_vectorized(V, F)
  area = area[None,:,None]

  assert area.min()>0

  return g_all * torch.sqrt(area)


def lap_entries(V, F, multicol=True):

  # this gives p.s.d. laplacian's entries. 

  g = grad_normalized(V, F)

  # remove the last single dimension: 

  v01 = inner_prod(g[0], g[1])[:,0]
  v12 = inner_prod(g[1], g[2])[:,0]
  v20 = inner_prod(g[2], g[0])[:,0]

  v00 = - (v01 + v20)
  v11 = - (v12 + v01)
  v22 = - (v20 + v12)

  II = [F[:,0],F[:,1],F[:,2],F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0]]
  JJ = [F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0],F[:,0],F[:,1],F[:,2]]
  VV = [v00,v11,v22,v01,v12,v20,v01,v12,v20]

  if multicol:
    II = torch.stack(II)
    JJ = torch.stack(JJ)
    VV = torch.stack(VV)
  else:
    II = torch.cat(II)
    JJ = torch.cat(JJ)
    VV = torch.cat(VV)

  n = V.shape[0]

  return II, JJ, VV, n


def grid_to_double_covered_mesh(W,H):
  '''
  triangles in F are pointing outward following the right-hand rule---counter-clock-wise.  
  '''

  x = torch.linspace(0, 1.0*(W-1), W)
  y = torch.linspace(0, 1.0*(H-1), H)
  xv, yv = torch.meshgrid(x, y, indexing='xy')

  V = torch.stack([xv.flatten(), yv.flatten()], dim=-1)
  # print(V)

  '''
  F = []
  H0 = []
  H1 = []

  for j in range(H-1):
    for i in range(W-1):
      ind00 = i + j * W
      ind10 = (i+1) + j * W
      ind01 = i + (j+1) * W
      ind11 = (i+1) + (j+1) * W
      F.append([ind00, ind10, ind01])
      F.append([ind11, ind01, ind10])
      F.append([ind10, ind11, ind00])
      F.append([ind01, ind00, ind11])

      H0.append([ind00, ind10, ind01])
      H0.append([ind11, ind01, ind10])
      H1.append([ind10, ind11, ind00])
      H1.append([ind01, ind00, ind11])

  F = np.array(F, dtype=np.int64)
  H0 = np.array(H0, dtype=np.int64)
  H1 = np.array(H1, dtype=np.int64)
  '''

  I = torch.arange(W-1, dtype=torch.int64)
  J = torch.arange(H-1, dtype=torch.int64)

  ind00 = I[None, :] + J[:, None]  * W
  ind10 = (I+1)[None, :] + J[:, None]  * W
  ind01 = I[None, :] + (J+1)[:, None]  * W
  ind11 = (I+1)[None, :] + (J+1)[:, None]  * W

  ind00 = ind00.flatten()
  ind10 = ind10.flatten()
  ind01 = ind01.flatten()
  ind11 = ind11.flatten()

  F = torch.cat([
        torch.stack([ind00, ind10, ind01], dim=-1), 
        torch.stack([ind11, ind01, ind10], dim=-1), 
        torch.stack([ind10, ind11, ind00], dim=-1), 
        torch.stack([ind01, ind00, ind11], dim=-1), 
    ], axis=0)

  return V, F, None # [H0, H1] is not implemented.


############# end of keeping block ###########


def vertex2face(F, uv):

  batch_size = uv.shape[0]
  f = F.shape[0]

  uf = torch.zeros(torch.Size([batch_size,f])+uv.shape[2:])

  for i in range(batch_size):
    uf[i] = ( uv[i,F[:,0],...] + uv[i,F[:,1],...] + uv[i,F[:,2],...] )/3
  
  return uf


def edges(F):
  '''
  F shape is [..., f, 3]
  '''

  assert F.shape[-1]==3

  E = [ 
      F[...,0:2], \
      F[...,1:3], \
      torch.stack([F[...,2], F[...,0]], dim=-1)
  ]

  E = torch.concat(E, dim=-2)

  return E


def sparse_dirac_operator_2D(F, E=None, n=None):

  '''Circular boundary gradient, or discrete area form
  [Wang, Guo, Solomon 2023]
  '''

  if E==None:
    E = edges(F)
  # print(E.shape) # [..., #edges, 2]
  

  if n==None:
    n = F.max() + 1

  assert len(E.shape)==2 # not implement otherwise
  batch_size = 1 if len(E.shape)==2 else E.shape[0]

  dtype = torch.get_default_dtype()

  indices = torch.stack([E.flatten(), (E[..., [1,0]]).flatten()])
  values = 0.5 * torch.concat([-torch.ones_like(E[...,0], dtype=dtype).flatten(), torch.ones_like(E[...,0], dtype=dtype).flatten()])

  indices = torch.tensor(indices, requires_grad=False)
  values  = torch.tensor(values, requires_grad=False)

  print(indices.shape, values.shape, indices.dtype, values.dtype)


  if True: 

    spD = torch.sparse_coo_tensor( 
                  indices=indices, 
                  values=values, 
                  size=[n, n], 
                  dtype=dtype)
  else:
    # did not do this since it requires anyway compiling PyTorch with CUDA cuDSS 
    # >>> torch.linalg.solve( spD[...,0].to_sparse_csr(),  torch.ones([spD.shape[0]],dtype=torch.float32) )

    # adding a batch_size at the end
    spD = torch.sparse_coo_tensor( 
                  indices=indices, 
                  values=values[...,None], 
                  size=[n, n, 1], 
                  dtype=dtype)

  return spD


def edge_lengths_from_edge_list(V, E):


  # this does not work somehow. 
  #P0 = torch.select(V, 0, E[:,0])
  #P1 = torch.select(V, 0, E[:,1])

  P0 = V[E[:,0], :]
  P1 = V[E[:,1], :]

  l = torch.sqrt(torch.sum(torch.square(P1-P0), dim=-1))

  return l


def value_vertex2edge(E, VV):

  return (VV[:, E[:,0]] + VV[:, E[:,1]]) / 2.0


def assemble_lap_values(au, g):
  # g is same for all meshes. 

  # au: [48, 3844, 3] --> [3,3844,48]
  # g:  [3, 3844, 2]

  au = au.permute([2,1,0])

  # [au[i,:,0], au[i,:,1];
  #  au[i,:,1], au[i,:,2];]

  def inner_prod(p, q):
    # p, q: [3844, 2]
    return p[:,0,None]*au[0]*q[:,0,None] + p[:,0,None]*au[1]*q[:,1,None] + p[:,1,None]*au[1]*q[:,0,None] + p[:,1,None]*au[2]*q[:,1,None]

  v01 = inner_prod(g[0], g[1])
  v12 = inner_prod(g[1], g[2])
  v20 = inner_prod(g[2], g[0])

  v00 = - (v01 + v20)
  v11 = - (v12 + v01)
  v22 = - (v20 + v12)

  ## II = [F[:,0],F[:,1],F[:,2],F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0]]
  ## JJ = [F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0],F[:,0],F[:,1],F[:,2]]
  
  VV = [v00,v11,v22,v01,v12,v20,v01,v12,v20]

  values = torch.concatenate(VV)

  return values

