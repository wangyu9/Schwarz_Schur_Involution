import numpy as np

# https://www.geeksforgeeks.org/print-a-given-matrix-in-spiral-form/

def spiral_order(matrix):

  # Number of rows in the matrix
  m = len(matrix)

  # Number of columns in the matrix
  n = len(matrix[0])

  # List to store the spiral order elements
  result = []

  # Edge case: if matrix is empty
  if m == 0:
    return result

  # List to keep track of visited cells
  seen = [[False] * n for _ in range(m)]

  # Arrays to represent the four possible movement
  # directions: right, down, left, up
  # Change in row index for each direction
  dr = [0, 1, 0, -1]

  # Change in column index for each direction
  dc = [1, 0, -1, 0]

  # Initial position in the matrix
  r, c = 0, 0

  # Initial direction index (0 corresponds to 'right')
  di = 0

  # Iterate through all elements in the matrix
  for i in range(m * n):

    # Add current element to result list
    result.append(matrix[r][c])

    # Mark current cell as visited
    seen[r][c] = True

    # Calculate the next cell coordinates based on
    # current direction
    newR, newC = r + dr[di], c + dc[di]

    # Check if the next cell is within bounds and not
    # visited
    if 0 <= newR < m and 0 <= newC < n and not seen[newR][newC]:

        # Move to the next row
        r, c = newR, newC

    else:

        # Change direction (turn clockwise)
        di = (di + 1) % 4

        # Move to the next row according to new
        # direction
        r += dr[di]

        # Move to the next column according to new
        # direction
        c += dc[di]

  # Return the list containing spiral order elements
  return result


def indices_matrix_single_2d(p, q, x_first=True):

  xx, yy = np.meshgrid(np.arange(p),np.arange(q), indexing='ij')

  if x_first:
    indices = xx + yy * p
  else:
    indices = xx * q + yy

  assert indices.dtype == np.int64

  return indices


def indices_matrix_2d(w, h, a, b, i, j, x_first=True):

  xx, yy = np.meshgrid(np.arange(w),np.arange(h), indexing='ij')

  if x_first:
    indices = (xx+i*(w-1)) + (yy+j*(h-1)) * ((w-1)*a+1)
  else:
    indices = (xx+i*(w-1)) * ((h-1)*b+1) + (yy+j*(h-1))

  assert indices.dtype == np.int64

  return indices


def matrix_spiral_order(m, n): 
    
  matrix = [[i+j*m for i in range(m)] for j in range(n)]

  result = spiral_order(matrix)

  return result


def matrix_spiral_bdr_list(p, q):
  # this is equivalent to:
  ##  alg.matrix_spiral_order(p, q)[0:2*(p+q)-4]

  l = list(range(0,p-1,1)) \
    + list(range(p-1,p*q-1, p)) \
    + list(range(p*q-1, p*(q-1),-1)) \
    + list(range(p*(q-1),0,-p))

  return l


import itertools

def matrix_interior_list(m, n, ring=1):

  r = [[i+j*m for i in range(ring,m-ring)] for j in range(ring,n-ring)]

  return list(itertools.chain.from_iterable(r))


def matrix_bdr_interior_split(wh):

  B = matrix_spiral_bdr_list(wh[0], wh[1])
  UB = matrix_interior_list(wh[0], wh[1])

  return B, UB


from numpy import copy

def array_substitute(theArray, table):
  # https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array

  n = len(table)

  d = {table[i]:i for i in range(n)}

  if False: # this is too slow
    newArray = copy(theArray)
    for k, v in d.items(): newArray[theArray==k] = v
  else:
    newArray = copy(theArray)
    for index, x in np.ndenumerate(newArray):
      # print(index, x)
      newArray[index] = d[x]

  return newArray


def array_row_permute(theArray, table):

  newArray = theArray[table,:]

  return newArray


def inverse_permutation(permute):

  n = len(permute)
  assert min(permute)==0
  assert max(permute)==(n-1)

  return array_substitute([i for i in range(n)], permute).tolist()


def test_inverse_permutation(wh=[5,5]):
  B = matrix_spiral_bdr_list(wh[0], wh[1])
  UB = matrix_interior_list(wh[0], wh[1])
  # ind = torch.tensor(B + UB, dtype=torch.int64)

  print(B+UB) # list
  ordreing = inverse_permutation(B + UB)
  print(ordreing) # 
  tmp = inverse_permutation(ordreing)
  print(tmp) # numpy.ndarray

  # numpy.ndarray with list do: 
  assert (np.array(tmp) == (B+UB)).all()
  
  # list with list do: assert tmp == (B+UB)
  assert tmp == (B+UB)


def test_array_substitute(p, q):

  matrix_spiral_order(p, q)[0:2*(p+q)-4]
  B = matrix_spiral_bdr_list(p,q)
  np.array(B)

  UB = matrix_interior_list(p,q)
  np.array(UB)

  M = [[(i+j*p) for i in range(p)] for j in range(q)]
  M = np.array(M)
  print(M)

  SM = array_substitute(M, B+UB)

  print(SM)

  return


def test_spiral_order():

  # Example matrix initialization
  matrix = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16]
  ]

  # Function call to get the spiral order traversal
  result = spiral_order(matrix)

  # Print the result elements
  print(result)
  
  return matrix, result


# Main function
def test_spiral_order2(m=7, n=5):

  # Example matrix initialization
  # matrix = [[(i,j,i+j*m) for j in range(n)] for i in range(m)]
  matrix = [[(i,j,i+j*m) for i in range(m)] for j in range(n)]

  # Function call to get the spiral order traversal
  result = spiral_order(matrix)

  # Print the result elements
  print(result)
  
  return matrix, result
