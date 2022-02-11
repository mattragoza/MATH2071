import numpy as np


def plu_factor(A):
    '''
    Factorize the matrix A into a permutation
    matrix, a lower triangular matrix, and an
    upper triangular matrix.

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


def solve(A, b):
    '''
    Solve a linear system Ax = b
    for x by Gaussian elimination.
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


if __name__ == '__main__':
    import sys
    import pandas as pd
    sys.path.append('lab-5/code')
    import gallery

    def test_plu_factor(n, matrix):
        A = matrix(n)
        P, L, U = plu_factor(A)
        PLU = P@L@U
        e = np.linalg.norm(A - PLU)
        return e

    data = []
    for m in ['dif2', 'pascal', 'lulu']:
        matrix = getattr(gallery, f'{m}_matrix')
        for n in [5, 10, 20]:
            e = test_plu_factor(n, matrix)
            data.append((m, n, e))

    df = pd.DataFrame(data, columns=['matrix', 'n', 'error'])
    df = df.set_index(['matrix', 'n'])
    print(df)
