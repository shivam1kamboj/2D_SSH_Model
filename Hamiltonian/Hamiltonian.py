import numpy as np

# Hamiltonian is defined as written in equation (4) in 2D STO model notes in the Dr. Hurst research group google drive


def Hamiltonian_2DSSH(t, J, J2, m, n, w, a, b, l):  # n_cell depicts no. of unit cells
    T = w + 1j * (a - l * w)
    T2 = w + 1j * (b - l * w)
    n_cell = int(m * n)

    Dd1 = np.tile(np.array([T, T2]), n_cell)  # Diagonal terms
    D = np.diag(Dd1, k=0)  # Diagonal matrix

    t1 = np.tile(np.array([J, t]), n_cell)  # J and J tilde terms

    h = np.diag(t1, k=-1) + np.diag(t1, k=1)
    H0 = h[0:-1, 0:-1]  # 1D coupling matrix containing J and J tilde terms

    ax = int(n * 2)  # axis of diagonailzation for vertical coupling matrix

    j2 = J2 * np.ones(2 * n_cell)  # vertical coupling
    r1 = np.diag(j2, k=-ax) + np.diag(j2, k=ax)
    R1 = r1[0:-ax, 0:-ax]  # vertical coupling matrix
    H = H0 + D + R1
    for i in range(1, m):
        H[2 * i * n - 1, 2 * i * n] = 0
        H[2 * i * n, 2 * i * n - 1] = 0
    return H
