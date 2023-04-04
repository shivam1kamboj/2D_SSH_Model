import numpy as np

# Fast diagonalization algorithm


def H_matrix_eigens(h, N):

    """This function takes input matrix h in the form of Block tridiagonal matrix
    and is specific to only this type of matrix"""

    row, column = h.shape
    L = int(column // N)  # L is size of block and N is number of rows

    A, B = h[:L, :L], h[:L, L:2 * L]  # Diagonal block A and sub-diagonal block B

    H = np.empty(h.shape, complex)  # H matrix to store diagonal blocks (H will be block-diagonal)
    for i in range(N):
        H[L * i:L + L * i, L * i:L + L * i] = A + 2 * B * np.cos(i * np.pi / (N + 1))

    Eigen_values = np.empty((row,), complex)  # 1-D array to store eigenvalues
    for i in range(row):
        Eigen_values[L * i:L + L * i] = np.linalg.eigvals(
            H[L * i:L + L * i, L * i:L + L * i])  # I used numpy to diagonalise blocks
    return np.sort(Eigen_values)
