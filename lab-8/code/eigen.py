#! /usr/bin/env python3
#
def eigen_matrix ( p ):

## eigen_matrix() returns one of several test matrices for eigen problems.
#
#  Input:
#    integer p, the problem number between 0 and 7.
#
#  Output:
#    real A(N,N), a test matrix.
#
  import numpy as np
#
#  p = 0
#  Eigenvalues are 2, 1, -1.5.
#
  if ( p == 0 ):

    A = np.array ( [ \
      [ 0, -1.0,  1 ], \
      [ 7,  5.5, -7 ], \
      [ 5,  2.5, -4 ] ], dtype = np.double )
#
#  p = 1
#  Eigenvalues are 4.73, 3.0, 1.27
#  (See Atkinson, page 624).
#
  elif ( p == 1 ):

    A = np.array ( [ \
      [ 2, 1, 0 ], \
      [ 1, 3, 1 ], \
      [ 0, 1, 4 ] ], dtype = np.double )
#
#  p = 2
#  Eigenvalues are 6.2749, -1.2749, -2
#  eigenvalue of smallest magnitude is negative
#
  elif ( p == 2 ):

    A = np.array ( [ \
      [ 1,  2,  3 ], \
      [ 2,  1,  4 ], \
      [ 3,  2,  1 ] ], dtype = np.double )
#
#  p = 3
#  Eigenvalues are 0.20, 0.75, 1.55, 2.45, 3.24, 3.80.
#  (See Atkinson, page 620).
#
  elif ( p == 3 ):

    A = np.array ( [ \
      [ 2, 1, 0, 0, 0, 0 ], \
      [ 1, 2, 1, 0, 0, 0 ], \
      [ 0, 1, 2, 1, 0, 0 ], \
      [ 0, 0, 1, 2, 1, 0 ], \
      [ 0, 0, 0, 1, 2, 1 ], \
      [ 0, 0, 0, 0, 1, 2 ] ], dtype = np.double )
#
#  p = 4
#  Eigenvalues are 2, -2, 1+sqrt(3)i, 1-sqrt(3)i
#
  elif ( p == 4 ):

    A = np.array ( [ \
      [ 2.4, -4.4,  2.0,  1.6 ], \
      [ 2.8, -4.8,  2.0,  3.2 ], \
      [ 3.2, -5.2,  4.0,  0.8 ], \
      [ 3.6, -3.6,  2.0,  0.4 ] ], dtype = np.double )
#
#  p = 5
#  Eigenvalues are approximately 22.4, 6, 3, 2, 1.61
#  -6*inv(dif2(5))
#
  elif ( p == 5 ):

    A = np.array ( [ \
      [ 5,  4,  3,  2,  1 ], \
      [ 4,  8,  6,  4,  2 ], \
      [ 3,  6,  9,  6,  3 ], \
      [ 2,  4,  6,  8,  4 ], \
      [ 1,  2,  3,  4,  5 ] ], dtype = np.double )
#
#  p = 6
#  dif2 matrix, size 50: slowly converging for power method.
#  LAMBDA(I) = 2 + 2 * COS(I*PI/(N+1)) = 4 SIN^2(I*PI/(2*N+2))
#
  elif ( p == 6 ):
    n = 50
    d = 2.0 * np.ones ( n )
    s = -1.0 * np.ones ( n - 1 )
    A = np.diag ( s, k = -1 ) + np.diag ( d ) + np.diag ( s, k = +1 )
#
#  p = 7
#  Eigenvalues are -4.1496, -2.1747, 3.3243
#
  elif ( p == 7 ):

    A = np.array ( [ \
      [ -2, 1,  0 ], \
      [  1, 3,  1 ], \
      [  0, 1, -4 ] ], dtype = np.double )
#
#  p out of bounds.
#
  else:
    raise Exception ( 'eigen_matrix(): No matrix for index p = ', p );

  return A

def eigen_values ( p ):

## eigen_values() returns eigenvalues for one of several eigen problems.
#
#  Input:
#    integer p, the problem number between 0 and 7.
#
#  Output:
#    real W[N], the eigenvalues for the test matrix.
#
  import numpy as np

  if ( p == 0 ):

    w = np.array ( [ -1.5, 1.0, 2.0 ], dtype = np.double )

  elif ( p == 1 ):

    w = np.array ( [ 1.26794919, 3.0, 4.73205081 ], dtype = np.double )

  elif ( p == 2 ):

    w = np.array ( [ 6.27491722, -1.27491722, -2.0 ], dtype = np.double )

  elif ( p == 3 ):

    w = np.array ( [ 3.80193774, 3.2469796, 2.44504187, 0.19806226, \
      1.55495813, 0.7530204 ], dtype = np.double )

  elif ( p == 4 ):

    w = np.array ( [-2.+0.j, 1.+1.73205081j, 1.-1.73205081j, 2.+0.j ], \
      dtype = np.cdouble )

  elif ( p == 5 ):

    w = np.array ( [ 22.39230485, 6.0, 3.0, 2.0, 1.60769515 ], dtype = np.double )

  elif ( p == 6 ):

    w = np.array ( [ \
        3.99620666e+00, 3.98484102e+00, 3.96594620e+00, 3.93959387e+00, \
        3.90588400e+00, 3.86494446e+00, 3.81693054e+00, 3.76202439e+00, \
        3.70043427e+00, 3.63239382e+00, 3.55816115e+00, 3.47801783e+00, \
        3.39226789e+00, 3.30123660e+00, 3.20526927e+00, 3.10472995e+00, \
        3.00000000e+00, 2.89147671e+00, 2.77957175e+00, 2.66470960e+00, \
        2.54732598e+00, 2.42786617e+00, 2.30678331e+00, 2.18453672e+00, \
        2.06159012e+00, 1.93840988e+00, 1.81546328e+00, 1.69321669e+00, \
        1.57213383e+00, 1.45267402e+00, 1.33529040e+00, 1.22042825e+00, \
        1.10852329e+00, 3.79334253e-03, 1.51589807e-02, 3.40538006e-02, \
        6.04061279e-02, 9.41159991e-02, 1.35055541e-01, 1.83069456e-01, \
        2.37975611e-01, 2.99565729e-01, 3.67606175e-01, 4.41838851e-01, \
        5.21982166e-01, 6.07732108e-01, 6.98763400e-01, 7.94730727e-01, \
        8.95270054e-01, 1.00000000e+00 ], dtype = np.double )

  elif ( p == 7 ):

    w = np.array ( [ 3.32434738, -2.17474532, -4.14960207 ], dtype = np.double )
#
#  p out of bounds.
#
  else:
    raise Exception ( 'eigen_values(): No eigenvalues for index p = ', p );

  return w

def eigen_vectors ( p ):

## eigen_vectors() returns eigenvectors for one of several eigen problems.
#
#  Input:
#    integer p, the problem number between 0 and 7.
#
#  Output:
#    real V[N,N], the eigenvectors for the test matrix.  
#    Each eigenvector is a COLUMN of V.
#
  import numpy as np

  if ( p == 0 ):

    V = np.array ( [ \
      [-1.83462559e-16,  7.07106781e-01,  4.47213595e-01], \
      [ 7.07106781e-01, -4.57966998e-16, -8.94427191e-01], \
      [ 7.07106781e-01,  7.07106781e-01,  1.65502277e-15]], \
      dtype = np.double )

  elif ( p == 1 ):

    V = np.array ( [ \
      [-0.78867513, -0.57735027,  0.21132487], \
      [ 0.57735027, -0.57735027,  0.57735027], \
      [-0.21132487,  0.57735027,  0.78867513] ], dtype = np.double)

  elif ( p == 2 ):

    V = np.array ( [ \
      [-0.55099872, -0.33414764, -0.12700013], \
      [-0.62673824,  0.88130058, -0.76200076], \
      [-0.55099872, -0.33414764,  0.63500064]] , dtype = np.double )

  elif ( p == 3 ):

    V = np.array ( [ \
      [ 0.23192061, -0.41790651, -0.52112089,  0.23192061,  0.52112089,  0.41790651], \
      [ 0.41790651, -0.52112089, -0.23192061, -0.41790651, -0.23192061, -0.52112089], \
      [ 0.52112089, -0.23192061,  0.41790651,  0.52112089, -0.41790651,  0.23192061], \
      [ 0.52112089,  0.23192061,  0.41790651, -0.52112089,  0.41790651,  0.23192061], \
      [ 0.41790651,  0.52112089, -0.23192061,  0.41790651,  0.23192061, -0.52112089], \
      [ 0.23192061,  0.41790651, -0.52112089, -0.23192061, -0.52112089,  0.41790651]], \
      dtype = np.double )

  elif ( p == 4 ):

    V = np.array ( [ \
      [-0.61558701+0.j, -0.32598807-0.20912144j, -0.32598807+0.20912144j, -0.24806947+0.j ], \
      [-0.71818485+0.j, -0.59160798+0.j        , -0.59160798-0.j        , -0.49613894+0.j ], \
      [-0.30779351+0.j, -0.525203  -0.05228036j, -0.525203  +0.05228036j, -0.74420841+0.j ], \
      [ 0.10259784+0.j, -0.45879802-0.10456072j, -0.45879802+0.10456072j, -0.3721042 +0.j ]], \
      dtype = np.cdouble )

  elif ( p == 5 ):

    V = np.array ( [ \
      [-2.88675135e-01, -5.00000000e-01, -5.77350269e-01, -5.00000000e-01,  2.88675135e-01], \
      [-5.00000000e-01, -5.00000000e-01,  7.55807087e-16,  5.00000000e-01, -5.00000000e-01], \
      [-5.77350269e-01,  5.57424527e-16,  5.77350269e-01, -1.80715269e-15,  5.77350269e-01], \
      [-5.00000000e-01,  5.00000000e-01, -1.65791836e-16, -5.00000000e-01, -5.00000000e-01], \
      [-2.88675135e-01,  5.00000000e-01, -5.77350269e-01,  5.00000000e-01,  2.88675135e-01]], \
      dtype = np.double )

  elif ( p == 6 ):

    n = 50
    V = np.zeros ( ( n, n ) )

    for i in range ( 0, n ):
      for j in range ( 0, n ):
        angle = float ( ( i + 1 ) *  ( j + 1 ) ) * np.pi / float ( n + 1 )
        V[i,j] = np.sqrt ( 2.0 / float ( n + 1 ) ) * np.sin ( angle )

  elif ( p == 7 ):

    V = np.array ( [ \
      [ 0.18294927, -0.98072138,  0.06866681],\
      [ 0.97408546,  0.17137647, -0.14760632],\
      [ 0.1329928 ,  0.09389181,  0.98665964]], dtype = np.double )
#
#  p out of bounds.
#
  else:
    raise Exception ( 'eigen_values(): No eigenvalues for index p = ', p );

  return V

def eigen_test ( ):

  import numpy as np

  print ( "eigen_test:" )

  for p in range ( 0, 8 ):
    A = eigen_matrix ( p )
    w = eigen_values ( p )
    print ( "" )
    print ( "  p = ", p )
    print ( w )

  return

if ( __name__ == '__main__' ):
  eigen_test ( )

