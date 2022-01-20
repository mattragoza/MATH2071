import numpy as np


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


A = np.array([
	[2, 1, 1],
	[4, 4, 3],
	[6, 7, 4]
])
x = np.array([[1, 2, 3]]).T
b = np.array([[7, 21, 32]]).T

A = np.array([
	[0.0003, 1.566],
	[0.3454, 2.436],
])
x = np.array([[10, 1]]).T
b = np.array([[1.569, 1.01]]).T

assert np.allclose(A @ x, b), A @ x
assert np.allclose(solve(A, b), x)
