#! /usr/bin/env python3
#
def dif2_matrix ( n ):

#*****************************************************************************80
#
## dif2_matrix() returns the second difference matrix.
#
#  Modified:
#    23 October 2021
#  Author:
#    John Burkardt
#  Input:
#    integer N: the number of rows and columns of A.
#  Output:
#    real A[N,N]: the matrix.
#
  import numpy as np

  d = 2.0 * np.ones ( n )
  s = -1.0 * np.ones ( n - 1 )

  A = np.diag ( s, k = -1 ) + np.diag ( d ) + np.diag ( s, k = +1 )

  return A

def dif2_sparse ( n ):

#*****************************************************************************80
#
## dif2_sparse() returns a sparse second difference matrix.
#
#  Discussion:
#    The matrix will be in the "csc" format, which allows the use of the
#    x = spsolve ( A, b ) solver for linear systems.
#  Modified:
#    26 October 2021
#  Author:
#    John Burkardt
#  Input:
#    integer N: the number of rows and columns of A.
#  Output:
#    real A[N,N]: the matrix.
#
  from scipy.sparse import spdiags
  from scipy.sparse import coo_matrix
  import numpy as np

  dm1 = -1.0 * np.ones ( n )
  dia = 2.0 * np.ones ( n )
  dp1 = -1.0 * np.ones ( n )
  data = np.array ( [ dm1, dia, dp1 ] )
  diags = np.array ( [ -1, 0, +1 ] )
  A = spdiags ( data, diags, n, n )
#
#  Convert to csc format.
#
  A = A.tocsc ( )

  return A

def frank_matrix ( n ):

#*****************************************************************************80
#
## frank_matrix() returns the Frank matrix.
#
#  Modified:
#    25 October 2021
#  Author:
#    John Burkardt
#  Input:
#    integer N: the number of rows and columns of A.
#  Output:
#    real A[N,N]: the matrix.
#
  import numpy as np

  A = np.zeros ( [ n, n ], dtype = np.float )

  for i in range ( 0, n ):
    for j in range ( 0, n ):
      if ( j == i - 1 ):
        A[i,j] = float ( n - i )
      elif ( i <= j ):
        A[i,j] = float ( n - j )

  return A

def helmert_matrix ( n ):

#*****************************************************************************80
#
## helmert_matrix() returns the Helmert matrix.
#
#  Formula:
#
#    If I == 0 then
#      A(I,J) = 1 / sqrt ( N )
#    else if J < I then
#      A(I,J) = 1 / sqrt ( I * ( I + 1 ) )
#    else if J == I then
#      A(I,J) = -I / sqrt ( I * ( I + 1 ) )
#    else
#      A(I,J) = 0
#
#  Discussion:
#
#    The matrix given above by Helmert is the classic example of
#    a family of matrices which are now called Helmertian or
#    Helmert matrices.
#
#    A matrix is a (standard) Helmert matrix if it is orthogonal,
#    and the elements which are above the diagonal and below the
#    first row are zero.
#
#    If the elements of the first row are all strictly positive,
#    then the matrix is a strictly Helmertian matrix.
#
#    It is possible to require in addition that all elements below
#    the diagonal be strictly positive, but in the reference, this
#    condition is discarded as being cumbersome and not useful.
#
#    A Helmert matrix can be regarded as a change of basis matrix
#    between a pair of orthonormal coordinate bases.  The first row
#    gives the coordinates of the first new basis vector in the old
#    basis.  Each later row describes combinations of (an increasingly
#    extensive set of) old basis vectors that constitute the new
#    basis vectors.
#
#    Helmert matrices have important applications in statistics.
#
#  Example:
#
#    N = 5
#
#    0.4472    0.4472    0.4472    0.4472    0.4472
#    0.7071   -0.7071         0         0         0
#    0.4082    0.4082   -0.8165         0         0
#    0.2887    0.2887    0.2887   -0.8660         0
#    0.2236    0.2236    0.2236    0.2236   -0.8944
#
#  Properties:
#
#    A is generally not symmetric: A' ~= A.
#
#    A is orthogonal: A' * A = A * A' = I.
#
#    Because A is orthogonal, it is normal: A' * A = A * A'.
#
#    det ( A ) = (-1)^(N+1)
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    11 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    HO Lancaster,
#    The Helmert Matrices,
#    American Mathematical Monthly,
#    Volume 72, 1965, pages 4-12.
#
#  Input:
#
#    integer N, the order of A.
#
#  Output:
#
#    real A(N,N), the matrix.
#
  import numpy as np

  A = np.zeros ( ( n, n ), dtype = np.float )
#
#  A begins with the first row, diagonal, and lower triangle set to 1.
#
  for i in range ( 0, n ):
    for j in range ( 0, n ):

      if ( i == 0 ):
        A[i,j] = 1.0 / np.sqrt ( n )
      elif ( j < i ):
        A[i,j] = 1.0 / np.sqrt ( float ( i * ( i + 1 ) ) )
      elif ( i == j ):
        A[i,j] = float ( - i ) / np.sqrt ( float ( i * ( i + 1 ) ) )

  return A

def hilbert_inverse ( n ):

#*****************************************************************************80
#
## hilbert_inverse() returns the inverse of the Hilbert matrix.
#
#  Formula:
#
#    A(I,J) =  (-1)^(I+J) * (N+I-1)! * (N+J-1)! /
#           [ (I+J-1) * ((I-1)!*(J-1)!)^2 * (N-I)! * (N-J)! ]
#
#  Example:
#
#    N = 5
#
#       25    -300     1050    -1400     630
#     -300    4800   -18900    26880  -12600
#     1050  -18900    79380  -117600   56700
#    -1400   26880  -117600   179200  -88200
#      630  -12600    56700   -88200   44100
#
#  Properties:
#
#    A is symmetric: A' = A.
#
#    Because A is symmetric, it is normal.
#
#    Because A is normal, it is diagonalizable.
#
#    A is almost impossible to compute accurately by general routines
#    that compute the inverse.
#
#    A is the exact inverse of the Hilbert matrix; however, if the
#    Hilbert matrix is stored on a finite precision computer, and
#    hence rounded, A is actually a poor approximation
#    to the inverse of that rounded matrix.  Even though Gauss elimination
#    has difficulty with the Hilbert matrix, it can compute an approximate
#    inverse matrix whose residual is lower than that of the
#    "exact" inverse.
#
#    All entries of A are integers.
#
#    The sum of the entries of A is N^2.
#
#    The family of matrices is nested as a function of N.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    26 March 2015
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer N, the order of A.
#
#  Output:
#
#    real A(N,N), the matrix.
#
  import numpy as np

  A = np.zeros ( ( n, n ), dtype = np.float )
#
#  Set the (1,1) entry.
#
  A[0,0] = float ( n * n )
#
#  Define Row 1, Column J by recursion on Row 1 Column J-1
#
  i = 0
  for j in range ( 1, n ):
    A[i,j] = - A[i,j-1] * float ( ( n + j ) * ( i + j ) * ( n - j ) ) \
      / float ( ( i + j + 1 ) * j * j )
#
#  Define Row I by recursion on row I-1
#
  for i in range ( 1, n ):
    for j in range ( 0, n ):

      A[i,j] = - A[i-1,j] * float ( ( n + i ) * ( i + j ) * ( n- i ) ) \
        / float ( ( i + j + 1 ) * i * i )

  return A

def hilbert_matrix ( n ):

#*****************************************************************************80
#
## hilbert_matrix() returns the Hilbert matrix.
#
#  Formula:
#
#    A(I,J) = 1 / ( I + J - 1 )
#
#  Example:
#
#    N = 5
#
#    1/1 1/2 1/3 1/4 1/5
#    1/2 1/3 1/4 1/5 1/6
#    1/3 1/4 1/5 1/6 1/7
#    1/4 1/5 1/6 1/7 1/8
#    1/5 1/6 1/7 1/8 1/9
#
#  Rectangular Properties:
#
#    A is a Hankel matrix: constant along anti-diagonals.
#
#  Square Properties:
#
#    A is positive definite.
#
#    A is symmetric: A' = A.
#
#    Because A is symmetric, it is normal.
#
#    Because A is normal, it is diagonalizable.
#
#    A is totally positive.
#
#    A is a Cauchy matrix.
#
#    A is nonsingular.
#
#    A is very ill-conditioned.
#
#    The entries of the inverse of A are all integers.
#
#    The sum of the entries of the inverse of A is N*N.
#
#    The ratio of the absolute values of the maximum and minimum
#    eigenvalues is roughly EXP(3.5*N).
#
#    The determinant of the Hilbert matrix of order 10 is
#    2.16417... * 10^(-53).
#
#    If the (1,1) entry of the 5 by 5 Hilbert matrix is changed from
#    1 to 24/25, the matrix is exactly singular.  And there
#    is a similar rule for larger Hilbert matrices.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    13 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    MD Choi,
#    Tricks or treats with the Hilbert matrix,
#    American Mathematical Monthly,
#    Volume 90, 1983, pages 301-312.
#
#    Robert Gregory, David Karney,
#    Example 3.8,
#    A Collection of Matrices for Testing Computational Algorithms,
#    Wiley, New York, 1969, page 33,
#    LC: QA263.G68.
#
#    Nicholas Higham,
#    Accuracy and Stability of Numerical Algorithms,
#    Society for Industrial and Applied Mathematics, Philadelphia, PA,
#    USA, 1996; section 26.1.
#
#    Donald Knuth,
#    The Art of Computer Programming,
#    Volume 1, Fundamental Algorithms, Second Edition
#    Addison-Wesley, Reading, Massachusetts, 1973, page 37.
#
#    Morris Newman and John Todd,
#    Example A13,
#    The evaluation of matrix inversion programs,
#    Journal of the Society for Industrial and Applied Mathematics,
#    Volume 6, 1958, pages 466-476.
#
#    Joan Westlake,
#    Test Matrix A12,
#    A Handbook of Numerical Matrix Inversion and Solution of Linear Equations,
#    John Wiley, 1968.
#
#  Input:
#
#    integer N, the number of rows and columns of A.
#
#  Output:
#
#    real A(N,N), the matrix.
#
  import numpy as np

  A = np.zeros ( ( n, n ), dtype = np.float )

  for i in range ( 0, n ):
    for j in range ( 0 , n ):
      A[i,j] = 1.0 / float ( i + j + 1 )

  return A

def jordan_block ( n, alpha ):

#*****************************************************************************80
#
## jordan_block() returns a Jordan block matrix.
#
#  Formula:
#
#    if ( I == J )
#      A(I,J) = ALPHA
#    else if ( I = J-1 )
#      A(I,J) = 1
#    else
#      A(I,J) = 0
#
#  Example:
#
#    ALPHA = 2, N = 5
#
#    2  1  0  0  0
#    0  2  1  0  0
#    0  0  2  1  0
#    0  0  0  2  1
#    0  0  0  0  2
#
#  Properties:
#
#    A is upper triangular.
#
#    A is lower Hessenberg.
#
#    A is bidiagonal.
#
#    Because A is bidiagonal, it has property A (bipartite).
#
#    A is banded, with bandwidth 2.
#
#    A is generally not symmetric: A' /= A.
#
#    A is persymmetric: A(I,J) = A(N+1-J,N+1-I).
#
#    A is nonsingular if and only if ALPHA is nonzero.
#
#    det ( A ) = ALPHA^N.
#
#    LAMBDA(I) = ALPHA.
#
#    A is defective, having only one eigenvector, namely (1,0,0,...,0).
#
#    The family of matrices is nested as a function of N.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    16 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer N, the order of A.
#
#    real ALPHA, the eigenvalue of the Jordan matrix.
#
#  Output:
#
#    real A(N,N), the matrix.
#
  import numpy as np

  a = np.zeros ( ( n, n ) )

  for i in range ( 0, n ):
    for j in range ( 0, n ):

      if ( i == j ):
        a[i,j] = alpha
      elif ( j == i + 1 ):
        a[i,j] = 1.0

  return a

def lulu_matrix ( n = 5 ):

#*****************************************************************************80
#
## lulu_matrix() returns the 5x5 "lulu" matrix.
#
#  Discussion:
#    This matrix cannot be LU factored.  It requires pivoting.
#  Modified:
#    27 October 2021
#  Author:
#    John Burkardt
#  Input:
#    integer N: ignored
#  Output:
#    real A[5,5]: the matrix.
#
  import numpy as np

  A = np.array ( [ \
    [ -2,  1,  0,  0,  0 ], \
    [  1,  0,  1, -2,  0 ], \
    [  0,  0,  0,  1, -2 ], \
    [  1, -2,  1,  0,  0 ], \
    [  0,  1, -2,  1,  0 ] ], dtype = np.float )

  return A

def magic_matrix ( n ):

#*****************************************************************************80
#
## magic_matrix() returns the magic matrix.
#
#  Example:
#
#    N = 5
#
#    17    24     1     8    15
#    23     5     7    14    16
#     4     6    13    20    22
#    10    12    19    21     3
#    11    18    25     2     9
#
#  Properties:
#
#    A is not symmetric.
#
#    The row sums and column sums and diagonal sums of A are n*(n^2+1)/2.
#
#    A has an eigenvalue of n*(n^2+1)/2, with left and right eigenvectors
#    of ( 1, 1, 1, ..., 1 ).
#
#    A is not diagonally dominant.
#
#    A is singular if N is even.
#
#  Modified:
#    25 October 2021
#  Input:
#    integer N: the number of rows and columns of A.  3 <= N.
#  Output:
#    real A[N,N]: the matrix.
#
  import numpy as np

  n = int ( n )

  if ( n < 3 ):
    raise Exception ( "magic_matrix(): Size n must be at least 3!" )

  if ( n % 2 == 1 ):
    p = np.arange ( 1, n + 1 )
    A = n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
  elif ( n % 4 == 0 ):
    J = np.mod(np.arange(1, n+1), 4) // 2
    K = J[:, None] == J
    A = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
    A[K] = n*n + 1 - A[K]
  else:
    p = n // 2
    A = magic_matrix ( p )
    A = np.block([[A, A+2*p*p], [A+3*p*p, A+p*p]])
    i = np.arange(p)
    k = (n-2)//4
    j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
    A[np.ix_(np.concatenate((i, i+p)), j)] = A[np.ix_(np.concatenate((i+p, i)), j)]
    A[np.ix_([k, k+p], [0, k])] = A[np.ix_([k+p, k], [0, k])]

  A = A.astype ( float )

  return A 

def moler3_matrix ( n ):

#*****************************************************************************80
#
## moler3_matrix() returns the Moler3 matrix.
#
#  Formula:
#
#    if ( I == J )
#      A(I,J) = I
#    else
#      A(I,J) = min(I,J) - 2
#
#  Example:
#
#    N = 5
#
#     1 -1 -1 -1 -1
#    -1  2  0  0  0
#    -1  0  3  1  1
#    -1  0  1  4  2
#    -1  0  1  2  5
#
#  Properties:
#
#    A is integral, therefore det ( A ) is integral, and 
#    det ( A ) * inverse ( A ) is integral.
#
#    A is positive definite.
#
#    A is symmetric: A' = A.
#
#    Because A is symmetric, it is normal.
#
#    Because A is normal, it is diagonalizable.
#
#    A has a simple Cholesky factorization.
#
#    A has one small eigenvalue.
#
#    The family of matrices is nested as a function of N.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    19 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer N, the number of rows and columns of A.
#
#  Output:
#
#    real A(N), the matrix.
#
  import numpy as np

  A = np.zeros ( ( n, n ), dtype = np.float )

  for i in range ( 0, n ):
    for j in range ( 0, n ):
      if ( i == j ):
        A[i,j] = float ( i + 1 )
      else:
        A[i,j] = float ( min ( i, j ) - 1 )

  return A

def pascal_matrix ( n ):

#*****************************************************************************80
#
## pascal_matrix() returns the Pascal matrix.
#
#  Formula:
#
#    If ( I == 1 or J == 1 )
#      A(I,J) = 1
#    else
#      A(I,J) = A(I-1,J) + A(I,J-1)
#
#  Example:
#
#    N = 5
#
#    1 1  1  1  1
#    1 2  3  4  5
#    1 3  6 10 15
#    1 4 10 20 35
#    1 5 15 35 70
#
#  Properties:
#
#    A is a "chunk" of the Pascal binomial combinatorial triangle.
#
#    A is positive definite.
#
#    A is symmetric: A' = A.
#
#    Because A is symmetric, it is normal.
#
#    Because A is normal, it is diagonalizable.
#
#    A is integral, therefore det ( A ) is integral, and 
#    det ( A ) * inverse ( A ) is integral.
#
#    A is nonsingular.
#
#    det ( A ) = 1.
#
#    A is unimodular.
#
#    Eigenvalues of A occur in reciprocal pairs.
#
#    The condition number of A is approximately 16^N / ( N*PI ).
#
#    The elements of the inverse of A are integers.
#
#    A(I,J) = (I+J-2)! / ( (I-1)! * (J-1)! )
#
#    The Cholesky factor of A is a lower triangular matrix R,
#    such that A = R * R'. 
#
#    If the (N,N) entry of A is decreased by 1, the matrix is singular.
#
#    Gregory and Karney consider a generalization of this matrix as
#    their test matrix 3.7, in which every element is multiplied by a
#    nonzero constant K.  They point out that if K is the reciprocal of
#    an integer, then the inverse matrix has all integer entries.
#
#    The family of matrices is nested as a function of N.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Robert Brawer, Magnus Pirovino,
#    The linear algebra of the Pascal matrix,
#    Linear Algebra and Applications,
#    Volume 174, 1992, pages 13-23.
#
#    Robert Gregory, David Karney,
#    Example 3.7,
#    A Collection of Matrices for Testing Computational Algorithms,
#    Wiley, 1969, page 32, 
#    LC: QA263.G68.
#
#    Nicholas Higham,
#    Accuracy and Stability of Numerical Algorithms,
#    Society for Industrial and Applied Mathematics,
#    Philadelphia, PA, USA, 1996; section 26.4.
#
#    Sam Karlin,
#    Total Positivity, Volume 1,
#    Stanford University Press, 1968.
#
#    Morris Newman, John Todd,
#    The evaluation of matrix inversion programs,
#    Journal of the Society for Industrial and Applied Mathematics,
#    Volume 6, Number 4, pages 466-476, 1958.
#
#    Heinz Rutishauser,
#    On test matrices,
#    Programmation en Mathematiques Numeriques,
#    Centre National de la Recherche Scientifique,
#    1966, pages 349-365.
#
#    John Todd,
#    Basic Numerical Mathematics, Vol. 2: Numerical Algebra,
#    Academic Press, 1977, page 172.
#
#    HW Turnbull,
#    The Theory of Determinants, Matrices, and Invariants,
#    Blackie, 1929.
#
#  Input:
#
#    integer N, the order of A.
#
#  Output:
#
#    real A(N,N), the matrix.
#
  import numpy as np

  A = np.zeros ( ( n, n ), dtype = np.float )

  for i in range ( 0, n ):
    for j in range ( 0, n ):

      if ( i == 0 ):
        A[i,j] = 1.0
      elif ( j == 0 ):
        A[i,j] = 1.0
      else:
        A[i,j] = A[i,j-1] + A[i-1,j]

  return A

def perm_matrix ( rows ):

#*****************************************************************************80
#
## perm_matrix() returns a permutation matrix.
#
#  Modified:
#    28 October 2021
#  Author:
#    John Burkardt
#  Input:
#    integer ROWS[N]: for each row, lists the column where a 1 appears.
#    Each value from 0 to N-1 should appear once.
#  Output:
#    real A[N,N]: the matrix.
#
  import numpy as np

  n = len ( rows )
  r1 = np.sort ( rows )
  r2 = np.arange ( n )
  if ( any ( r1 != r2 ) ):
    Exception ( 'perm_matrix(): input rows is not a permutation!' )
  A = np.zeros ( [ n, n ], dtype = np.float )
  for i in range ( 0, n ):
    A[i,rows[i]] = 1.0

  return A

def random_matrix ( n, seed = 123456789 ):

#*****************************************************************************80
#
## random_matrix() returns a random matrix.
#
#  Modified:
#    24 October 2021
#  Author:
#    John Burkardt
#  Input:
#    integer N: the number of rows and columns of A.
#    integer SEED: the seed for the random number generator, default=123456789.
#  Output:
#    real A[N,N]: the matrix.
#
  import numpy as np

  np.random.seed ( seed )

  A = np.random.normal ( size = ( n, n ) )

  return A

if ( __name__ == '__main__' ):

  print ( "gallery:" )

  A = dif2_matrix ( 5 )
  print ( "" )
  print ( "dif2 matrix:" )
  print ( A )

  A = dif2_sparse ( 5 )
  print ( "" )
  print ( "dif2 sparse:" )
  print ( A )

  A = frank_matrix ( 5 )
  print ( "" )
  print ( "frank matrix:" )
  print ( A )

  A = helmert_matrix ( 5 )
  print ( "" )
  print ( "helmert matrix:" )
  print ( A )

  A = hilbert_matrix ( 5 )
  print ( "" )
  print ( "hilbert matrix:" )
  print ( A )

  A = hilbert_inverse ( 5 )
  print ( "" )
  print ( "hilbert inverse:" )
  print ( A )

  A = jordan_block ( 7, 0.5 )
  print ( "" )
  print ( "Jordan block:" )
  print ( A )

  A = magic_matrix ( 5 )
  print ( "" )
  print ( "magic matrix:" )
  print ( A )

  A = moler3_matrix ( 5 )
  print ( "" )
  print ( "moler3 matrix:" )
  print ( A )

  A = pascal_matrix ( 5 )
  print ( "" )
  print ( "pascal matrix:" )
  print ( A )

  A = perm_matrix ( [ 1, 3, 2, 0 ] )
  print ( "" )
  print ( "perm matrix:" )
  print ( A )

  A = random_matrix ( 5 )
  print ( "" )
  print ( "random matrix, default seed:" )
  print ( A )

  A = random_matrix ( 5, 1776 )
  print ( "" )
  print ( "random matrix, seed = 1776:" )
  print ( A )

  A = random_matrix ( 5 )
  print ( "" )
  print ( "random matrix, default seed:" )
  print ( A )

