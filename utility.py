import numpy as np
import torch


def max_unused_folder_index(path):
  '''
  example usage: 
  ==============

    fpath = '/workspace/projects/diff-linear-solve/data/temp/'
    imax = util.max_unused_folder_index(fpath)
    folder = fpath + str(imax) + '/'
    os.mkdir(folder)

  '''
  import os
  assert os.path.exists(path)

  maxi = 0
  for _ in range(1000000):
    if os.path.exists(path + f'/{maxi}/'):
      maxi += 1
      continue
    else:
      break

  return maxi


def lap_from_stencil(sten, grid_size_x, grid_size_y): 
  
  import numpy as np
  import pyamg
  
  A = pyamg.gallery.stencil_grid(sten, (grid_size_x, grid_size_y))

  A.setdiag(0)

  A.setdiag(- A @ np.ones(A.shape[0])) # A.setdiag(- A * np.ones(A.shape[0]))

  return A


def error_measure_using_scipy(
  A_scipy, b_scipy, x_scipy, 
  U_scipy=None, V_scipy=None, Omega_scipy=None, sigma_scipy=None, lam_scipy=None, # only for augmented systems
  name=None
  ):

  tol = tsi.error_measure_using_scipy(A=A_scipy, b=b_scipy, x=x_scipy, 
    U=U_scipy, V=V_scipy, Omega=Omega_scipy, sigma=sigma_scipy, lam=lam_scipy)

  if not (name is None):
    print(f"\t{name}\terror={tol}")

  return tol


def scipy_solver_dirichlet(mdata, A_scipy):
  '''
  input: 
    note both b,bc are nx1 vector
  '''

  class SolverDirichlet():
    name = 'scipy_solver_dirichlet'

  solver = SolverDirichlet()

  B = mdata.B
  UB = mdata.UB

  # right way of indexing: https://stackoverflow.com/questions/28574041/modifying-block-matrices-in-python
  Auu = A_scipy[np.ix_(UB,UB)]
  Aub = A_scipy[np.ix_(UB,B)]

  lhs = Auu
  def get_rhs(b, bc):
    assert len(b.shape)==2
    assert len(bc.shape)==2
    rhs = b[UB,:] - Aub*bc
    return rhs

  def fetchX(x):
    return x[UB]

  def fill_in(xx, bc):
    x_ref = np.zeros([A_scipy.shape[1], bc.shape[-1]])
    # print(f'size of bc={bc.shape}, rhs={rhs.shape}, L={L.shape}, test={(Lub*bc).shape}, x_ref={x_ref.shape}')
    x_ref[B,:] = bc
    x_ref[UB,:] = xx
    return x_ref

  def solve(b, bc):
    import scipy
    rhs = get_rhs(b, bc)
    xx = scipy.sparse.linalg.spsolve(lhs,  rhs)
    xx = xx[:, np.newaxis]
    x_ref = fill_in(xx, bc)
    return x_ref
  
  def eval_sol(x, b, bc, name=''):
    rhs = get_rhs(b, bc)
    xx = fetchX(x)
    return error_measure_using_scipy(lhs, rhs, xx, name)

  solver.fetchX = fetchX
  solver.solve = solve
  solver.get_rhs = get_rhs
  solver.fill_in = fill_in
  solver.eval_sol = eval_sol
  solver.lhs = lhs

  return solver


'''
make_compiled_solver:
def mysolver(*args, **kwargs):
  return tip.Schwarz_Schur_involution(*args, **kwargs)
#with torch.no_grad():
#  return tip.condense_general_solve(*args, **kwargs)

'''

def make_summarize(problem, ab_wh, eval_sol, A_scipy):

  def summarize(dur=None, name=None, x=None, extra=None):
    # implicit input: problem, ab_wh, eval_sol, A_scipy
    # print(problem)
    info = {'problem': problem,
            'ab_wh': ab_wh,
            'dtype': str(torch.get_default_dtype()),
            'sym': 1 if np.abs((A_scipy-A_scipy.T).data).sum() < 1e-8 else 0,
            'avgnnz': A_scipy.nnz / A_scipy.shape[0]}
    if x is None:
      return info
    for k,v in info.items():
      print(f'{k}={v}\t', end="")
    print(f'\n\t{name}\ttime={dur*1000} ms')
    error = eval_sol(x, name)
    info[name] = {'error': error, 'time': dur*1000}
    if not (extra is None):
      print('\t\t\t info:\t', extra)
    return info
  
  return summarize


def save_problem(folder, info_dict, A_scipy, B_scipy, print_back=True):

  import scipy

  scipy.io.mmwrite(folder + f'/A.mtx', A_scipy)
  scipy.io.mmwrite(folder + f'/B.mtx', B_scipy)

  import json

  with open(folder+'/problem.json', "w") as outfile:
    json.dump(info_dict, outfile, indent=4, sort_keys=False)

  if print_back:
    jfile = folder+'/problem.json'

    with open(jfile) as f:
        d = json.load(f)
        print(d)

  return


def print_julia_record(
  d, 
  order=['cudss', 'lsqr', 'bicgstab', 'cgs', 'diom', 'fom', 'qmr', 'bilq', 'gmres', 'dqgmres', 'fgmres']
  ):
  # cudss_backsub, cg, lsmr2, lsqr2, lsmr

  # d.keys(), np.array(d['cudss']['x']).shape
  strformat = "{name:<9s}\t {time:<20}\t {error:<20}\t {time2:<20}"
  #strformat = "{name}\t {time}\t"
  print(strformat.format(name="name",time= "time (ms)",error="error",time2="time2"))

  for k in order:
  # for k,_ in d.items():
    # print(f'{k:<8}\t\t\
    #   {(np.Inf if (d[k]["time"] is None) else d[k]["time"]):<20}\t\
    #   {(np.Inf if (d[k]["error"] is None) else d[k]["error"]):<20}\t\
    #   {d[k]["time2"] if "time2" in d[k].keys() else None}')
    
    print(strformat.format(
        name=k,\
        time= np.Inf if d[k]["time"]  is None else d[k]["time"],\
        error=np.Inf if d[k]["error"] is None else d[k]["error"],\
        time2=d[k]["time2"] if "time2" in d[k].keys() else np.Inf,\
        ))

      #   {():<20}\t\
      # {():<20}\t\
      # {}'
  return


def seed_everything(seed=42):

    import os
    import random
    import numpy as np

    import torch
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy library
    np.random.seed(seed)
    
    # 4. PyTorch global CPU and CUDA seed
    torch.manual_seed(seed)
    
    # 5. CUDA-specific seeds (if using multiple GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 6. Configure CuDNN to be deterministic (can reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def simple_image():
  # import torch
  import numpy as np
  #import cv2

  import PIL
  import PIL.Image
  filename = "./data/images/lemur.png"


  if True:
      # 
      image_hwc = PIL.Image.open(filename)
      image_hwc = np.array(image_hwc) / 255.0
  else:
      # gray scale image.
      image_hwc = PIL.Image.open(filename).convert('L')
      #image_hwc = cv2.imread(filename)
      image_hwc = np.array(image_hwc)[...,None] / 255.0

  image_hwc = torch.tensor(image_hwc, dtype=torch.get_default_dtype())
  image_hwc = image_hwc.to(torch.get_default_device())


  # image_hwc.max()
  # out = image_hwc.detach().cpu().numpy()

  # import matplotlib.pyplot as plt
  # plt.imshow(out[...])
  # plt.show()

  #import PIL
  #PIL.Image.fromarray(out.astype(np.uint8))
  return image_hwc


def simple_matting_laplacian(
  ab_wh=((128,128), (5,5)),
  image_hwc=None,
  lam = 0.01,  # 0.0001 or 0.01 in the paper.
  vis=False
  ):

  class Info:
    name = 'system_info'

  info = Info()

  problem = 'matting_lap_'

  a,b,w,h,W,H = tsi.unpack_dims(ab_wh)

  if image_hwc is None:
    fimg = "./data/images/lemur.png" # "cameraman"
    a,b,w,h,W,H = tsi.unpack_dims(ab_wh)
    image_hwc = image_HWC_numpy(WH=(W,H), name=fimg) # demo
    image_hwc = torch.tensor(image_hwc, dtype=torch.get_default_dtype())

  # lam = 0.0001 # 0.0001 # 0.01
  problem += str(lam)
  alpha = tsi.alpha_image_laplacian(image_hwc, ab_wh, diag_zero_sum=True)
  alpha += tsi.alpha_uniform_mass_diagonal(ab_wh) * lam / 2
  # alpha += torch.eye(25) * lam

  A_scipy = tsi.torch_bottom_up_sparse_scipy(alpha[0], ab_wh=ab_wh, adj='9pt') # the sparse matrix A.

  B_scipy = A_scipy @ image_hwc.flatten(start_dim=0, end_dim=1).cpu().numpy() # since the image is [H,W,C]

  beta = tsi.beta_from_numpy(B_scipy, ab_wh)

  BC = 'Neumann'
  # BC = 'Neumann-full'

  if vis:
    import matplotlib.pyplot as plt
    plt.imshow(image_hwc.detach().cpu().numpy())
    plt.show()

  return problem, alpha, beta, BC, A_scipy, B_scipy, info


def simple_helmholtz_equation(
  ab_wh=((128,128), (5,5)), 
  kc = 0.01, # 0.01 or 1.0 in the paper.
  ):
  
  info = None

  problem = 'helmholtz'

  a,b,w,h,W,H = tsi.unpack_dims(ab_wh)

  # kc = 1 # 2 #10.0 # 0.01 # 0.1 # 0.01 # 1.0 # 1 # 0.1
  kasq = kc / ((a/64)*(b/64))

  problem += "_kc=" + str(kc)

  alpha_lap  = tsi.alpha_uniform_laplacian(ab_wh)
  alpha_mass = tsi.alpha_uniform_mass_diagonal(ab_wh)

  alpha = alpha_lap - kasq * alpha_mass

  A_scipy = tsi.torch_bottom_up_sparse_scipy(alpha[0], ab_wh=ab_wh, adj='9pt')

  B_scipy_unfl = np.zeros([W, H, 1]) # W,H,C
  B_scipy_unfl[W//4, H//2] = 100
  B_scipy_unfl[3*W//4, H//2] = -100

  B_scipy = B_scipy_unfl.swapaxes(0,1).flatten()[...,None]

  beta = tsi.beta_from_numpy(B_scipy, ab_wh)

  BC = 'Neumann'

  return problem, alpha, beta, BC, A_scipy, B_scipy, info


# from utility import make_summarize
import sinv
from sinv import torch_sparse_involution as tsi

#def sol2numpy(sol):
#  return tsi.sol2numpy(sol)


# def sparse_solver(
#   image_hwc, 
#   ab_wh=((128,128), (5,5)),
#   vis=True,
#   example=None, # matting_laplacian,
#   transposed_solve=False,
#   test_solvers='all', # [demo.run_cupy, demo.run_scipy, demo.run_amgx], [], 'all',
#   custom_experiments=lambda alpha, beta, BC, A_scipy, B_scipy, BC_scipy, info : None,
# ):

#   import time

#   import matplotlib.pyplot as plt

#   import torch

#   # torch.set_default_dtype(torch.float64)
#   # torch.set_default_dtype(torch.float32)

#   # if True:
#   with torch.no_grad():

#     a,b,w,h,W,H = tsi.unpack_dims(ab_wh)

#     problem, alpha, beta, BC, A_scipy, B_scipy, info = \
#       example(ab_wh, image_hwc) # demo.

#     #BC = 'Neumann-full'
#     #BC = 'Neumann-scatter'

#     if transposed_solve:
#       A_scipy = A_scipy.transpose().conjugate()
#       alpha = tsi.hermitian(alpha)

#     ### set up evaluation: ###

#     n_chans = image_hwc.shape[2]

#     def unflatten(x):
#       return x.reshape(W,H, n_chans)

#     if torch.is_tensor(BC):

#       solver_scipy = util.scipy_solver_dirichlet(mdata, A_scipy)
#       # x_scipy = solver_scipy.solve(B_scipy, BC_scipy)

#       def eval_sol_dirichlet(x, name=''):
#         return solver_scipy.eval_sol(x, B_scipy, BC_scipy, name=name)
#       def fill_in(xx):
#         return solver_scipy.fill_in(xx, BC_scipy)
#       def get_rhs(b):
#         return solver_scipy.get_rhs(b, BC_scipy)

#       def eval_sol(x, name=''):
#         # tol = error_measure_using_scipy(A_scipy, B_scipy, x, name=None)
#         tol = eval_sol_dirichlet(x, name=None)
#         print(f'\t{name}\terror={tol}')
#         if vis:
#           plt.imshow(unflatten(x))#, cmap='gray')
#           plt.show()
#         return tol

#     else:

#       solver_scipy = None # not used.

#       def eval_sol(x, name=''):
#         tol = error_measure_using_scipy(A_scipy, B_scipy, x, name=None)
#         print(f'\t{name}\terror={tol}')
#         if vis:
#           plt.imshow(unflatten(x))#, cmap='gray')
#           plt.show()
#         return tol

#       BC_scipy = BC

#     summarize = make_summarize(problem, ab_wh, eval_sol, A_scipy)

#     if False:

#       import os

#       fpath = '/workspace/projects/diff-linear-solve/data/temp/'

#       imax = util.max_unused_folder_index(fpath)
#       folder = fpath + str(imax) + '/'
#       print('saved in ', folder)
#       os.mkdir(folder)

#       util.save_problem(folder, summarize(), A_scipy, B_scipy)

#     if True:
#       print("test our method with the forward solve")

#       N = 10 #10

#       # warm up
#       if N>5:
#         for i in range(5):
#           sol = tsi.Schwarz_Schur_involution(alpha, beta, BC=BC, wh=ab_wh[1], transposed_solve=transposed_solve)
#           torch.cuda.synchronize()

#       start = time.time()

#       for i in range(N):
#         sol = tsi.Schwarz_Schur_involution(alpha, beta, BC=BC, wh=ab_wh[1], transposed_solve=transposed_solve)
#         torch.cuda.synchronize()

#       x_ours = sol2numpy(sol)

#       return summarize((time.time()-start)/N, "ours", x_ours)

def normalize(xx):
    return (xx - xx.min()) / (xx.max() - xx.min() + 1e-16) 


def sparse_solver(
  ab_wh=((128,128), (5,5)),
  vis=True,
  example=None, #matting_laplacian,
  transposed_solve=False,
  test_solvers='all', # [demo.run_cupy, demo.run_scipy, demo.run_amgx], [], 'all',
  custom_experiments=lambda alpha, beta, BC, A_scipy, B_scipy, BC_scipy, info : None,
  **kwargs
):

  import time

  import matplotlib.pyplot as plt

  import torch

  # torch.set_default_dtype(torch.float64)
  # torch.set_default_dtype(torch.float32)

  # if True:
  with torch.no_grad():

    a,b,w,h,W,H = tsi.unpack_dims(ab_wh)

    problem, alpha, beta, BC, A_scipy, B_scipy, info = \
      example(ab_wh=ab_wh, **kwargs) # demo.

    #BC = 'Neumann-full'
    #BC = 'Neumann-scatter'

    if transposed_solve:
      A_scipy = A_scipy.transpose().conjugate()
      # no need to do this, since transposed_solve=True will be passed: alpha = tsi.hermitian(alpha)
      # still need to make sure that B_scipy is compatible in the data generation process.

    ### set up evaluation: ###

    # n_chans = image_hwc.shape[2]
    n_chans = B_scipy.shape[-1]

    def unflatten(x):
      return x.reshape(W,H, n_chans)

    cmap = 'gray' if n_chans==1 else None

    if torch.is_tensor(BC):

      solver_scipy = util.scipy_solver_dirichlet(mdata, A_scipy)

      def eval_sol_dirichlet(x, name=''):
        return solver_scipy.eval_sol(x, B_scipy, BC_scipy, name=name)
      def fill_in(xx):
        return solver_scipy.fill_in(xx, BC_scipy)
      def get_rhs(b):
        return solver_scipy.get_rhs(b, BC_scipy)

      def eval_sol(x, name=''):
        # tol = error_measure_using_scipy(A_scipy, B_scipy, x, name=None)
        tol = eval_sol_dirichlet(x, name=None)
        print(f'\t{name}\terror={tol}')
        if vis:
          plt.imshow(unflatten(normalize(B_scipy)), cmap=cmap)
          plt.show()
          plt.imshow(unflatten(x), cmap=cmap)
          plt.show()
        return tol

    else:

      solver_scipy = None # not used.

      def eval_sol(x, name=''):
        tol = tsi.error_measure_using_scipy(A_scipy, B_scipy, x, name=None)
        print(f'\t{name}\terror={tol}')
        if vis:
          plt.imshow(unflatten(normalize(B_scipy)), cmap=cmap)
          plt.show()
          plt.imshow(unflatten(x), cmap=cmap)
          plt.show()
        return tol

      BC_scipy = BC

    summarize = make_summarize(problem, ab_wh, eval_sol, A_scipy)

    if True:
      # print("test our method with the forward solve")

      N = 10 #10

      # warm up
      if N>5:
        for i in range(5):
          sol = tsi.Schwarz_Schur_involution(alpha, beta, BC=BC, wh=ab_wh[1], transposed_solve=transposed_solve)
          torch.cuda.synchronize()

      start = time.time()

      for i in range(N):
        sol = tsi.Schwarz_Schur_involution(alpha, beta, BC=BC, wh=ab_wh[1], transposed_solve=transposed_solve)
        torch.cuda.synchronize()

      x_ours = tsi.sol2numpy(sol)

      summarize((time.time()-start)/N, "ours", x_ours)
  
  return custom_experiments(alpha, beta, BC, A_scipy, B_scipy, BC_scipy, info)
