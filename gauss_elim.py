import numpy as np


def timer(f):
    def wrapper(*args, **kwargs):
        t0 = time()
        ret = f(*args, **kwargs)
        dt = time() - t0
        return ret, dt
    return wrapper


def print_mat(round=4, **kwargs):
    for k, v in kwargs.items():
        print(f'{k} =\n{np.round(v, round)}')


def ge_solve(A, b):
    '''
    Solve a linear system Ax = b for x
    using Gaussian elimination.

    Args:
        A: M x N matrix of coefficents.
        b: N vector of righthand side.
    Returns:
        x: N vector solution.
    '''
    M, N = A.shape
    assert A.shape == (N, N)
    assert b.shape == (N, 1)

    # create augmented matrix
    W = np.append(A, b, axis=1)

    # iterate over pivots
    for i in range(N-1):

        # TODO check if pivot is zero
        pivot = W[i,i]
        assert not np.isclose(pivot, 0), 'pivot is zero'

        # iterate over rows below
        for j in range(i+1, N):

            # compute and store multiplier
            m = W[j,i] / pivot

            # perform row elimination
            for k in range(i, N+1):
                W[j,k] -= m * W[i,k]

    # back substitution
    x = np.zeros((N, 1))
    for i in reversed(range(N)):
        x[i] = W[i,N]
        for j in range(i+1, N):
            x[i] -= W[i,j] * x[j]
        x[i] /= W[i,i]

    return x


def lu_factor(A):
    
    n = A.shape[0]
    assert A.shape == (n, n)
    
    L = np.eye(n)
    U = A.copy()
    
    # iterate over pivot rows
    for i in range(n-1):

        # check if pivot is zero
        assert not np.isclose(U[i,i], 0), f'pivot {i} is zero'
        
        # compute and store multipliers in L
        L[i+1:,i] = U[i+1:,i] / U[i,i]
            
        # perform row reductions on U below pivot
        U[i+1:,i:] = U[i+1:,i:] - L[i+1:,i:i+1] * U[i:i+1,i:]

    return L, U


def plu_factor(A):
    '''
    Factorize the matrix A into a permutation
    matrix P, a lower triangular matrix L, and
    an upper triangular matrix U.

    Args:
        A: N x N matrix.
    Returns
        P: N x N permutation matrix.
        L: N x N lower triangular matrix.
        U: N x N upper triangular matrix.
    '''
    # check input shape
    n = A.shape[0]
    assert A.shape == (n, n)
    
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()
    
    # iterate over pivot rows/columns
    for i in range(n-1):

        # find largest entry below pivot
        p = i + np.argmax(np.abs(U[i:,i]))
        
        if p > i: # swap rows i and p
            U[[i,p],:]  = U[[p,i],:]
            L[[i,p],:i] = L[[p,i],:i]
            P[:,[i,p]]  = P[:,[p,i]]

        # compute and store multipliers in L
        L[i+1:,i] = U[i+1:,i] / U[i,i]
            
        # perform row reductions on U below pivot
        U[i+1:,i:] -= L[i+1:,i:i+1] * U[i:i+1,i:]

    return P, L, U


def p_solve(P, b):
    '''
    Solve Pz = b, where P is
    a permutation matrix.
    '''
    # check input shapes
    n = P.shape[0]
    assert P.shape == (n, n)
    assert b.shape == (n,)
    
    return P.T @ b


def l_solve(L, z):
    '''
    Solve Ly = z, where L is
    a lower triangular matrix.
    '''
    # check input shapes
    n = L.shape[0]
    assert L.shape == (n, n)
    assert z.shape == (n,)

    y = z.copy()
    for i in range(n):
        y[i] -= L[i,:i] @ y[:i]

    return y


def u_solve(U, y):
    '''
    Solve Ux = y, where U is
    an upper triangular matrix.
    '''
    # check input shapes
    n = U.shape[0]
    assert U.shape == (n, n)
    assert y.shape == (n,)

    x = y.copy()
    for i in reversed(range(n)):
        x[i] -= U[i,i+1:] @ x[i+1:]
        x[i] /= U[i,i]

    return x


def plu_solve(A, b):
    '''
    Solve Ax = b using
    PLU decomposition.
    '''
    P, L, U = plu_factor(A)
    z = p_solve(P, b)
    y = l_solve(L, z)
    x = u_solve(U, y)
    return x


def gs(X):
    '''
    Gram Schmidt orthogonalization method.

    Args:
        X: M x N matrix.
    Returns:
        M x N_Q matrix with orthonormal columns
            that span the same space as X.
    '''
    m, n = X.shape
    
    # accrue orthonormal vectors in Q
    Q = []
    
    # for each column,
    for j in range(n):
        
        X_j = X[:,j]
    
        # for each previous orthogonal column,
        for i in range(len(Q)):

            Q_i = Q[i]
    
            # compute the dot product
            r_ij = Q_i @ X_j
            
            # subtract the projection
            #   this makes them orthogonal
            X_j = X_j - Q_i * r_ij

        # normalize the vector
        r_jj = norm(X_j)

        if r_jj > 0:
            Q.append(X_j / r_jj)
             
    return np.stack(Q, axis=1)


def gs_factor(A):
    '''
    Factor an M x N matrix A into a N x N
    orthogonal matrix Q and a M x N upper
    triangular matrix R such that A = QR,
    using the Gram Schmidt method.

    Args:
        A: M x N matrix.
    Returns:
        M x N orthogonal matrix Q.
        N x N upper triangular matrix R.
    '''
    m, n = A.shape
    
    # initialize return values
    Q = A.astype(float)
    R = np.zeros((n, n))
    
    # for each column,
    for j in range(n):

        # for each previous column,
        for i in range(j):

            # compute the dot product
            R[i,j] = Q[:,i] @ Q[:,j]
            
            # subtract the projection
            #   this makes them orthogonal
            Q[:,j] -= Q[:,i] * R[i,j]

        # normalize the vector
        R[j,j] = norm(Q[:,j])
        assert R[j,j] > 0, 'zero column'
        Q[:,j] /= R[j,j]

    return Q, R


def householder(b, k):
    '''
    Return an orthogonal matrix H such 
    that all entries of H b below index
    k are zero.
    
    Args:
        b: Vector of length N.
        k: Scalar index.
    Returns:
        N x N Householder matrix.
    '''
    n, = b.shape
    d = b[k:n]
    alpha = norm(d)
    
    if alpha == 0:
        return np.eye(n)
    
    # choose sign to minimize roundoff errors
    if d[0] > 0:
        alpha = -alpha

    v = np.zeros_like(d, dtype=float)
    v[0] = np.sqrt(1/2*(1 - d[0]/alpha))
    p = -alpha*v[0]
    v[1:] = d[1:]/(2*p)

    # prepend zeros to get w
    w = np.r_[np.zeros(k), v]
    
    # construct householder matrix
    return np.eye(n) - 2*np.outer(w, w)


def h_factor(A):
    '''
    Factor an M x N matrix A into a M x M
    orthogonal matrix Q and a M x N upper
    triangular matrix R such that A = QR,
    using Householder transformations.

    Args:
        A: M x N matrix.
    Returns:
        M x M orthogonal matrix Q.
        M x N upper triangular matrix R.
    '''
    m, n = A.shape
    
    # initialize return values
    Q = np.eye(m)
    R = A.copy()
    
    # for each column,
    for k in range(n):

        # get Householder matrix that zeros out
        #   the current column below the diagonal
        H = householder(R[:,k], k)
        Q = Q @ H.T
        R = H @ R

    return Q, R
