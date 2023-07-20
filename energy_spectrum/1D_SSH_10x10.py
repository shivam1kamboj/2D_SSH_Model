from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')
def Hamiltonian_1DSSH(t, J, J2, m, n, k_x, a, b, l):  # n_cell depicts no. of unit cells
    w = 2*J2*np.cos(k_x)
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


# Calculating energy eigen values without vertical coupling (J2); 2D will show spectrum of 1D SSH
m, n, t, J2, J = 10, 10, 1, 0.1, 0.6
j0 = np.linspace(-np.pi, np.pi, 50)  # linear space of v values

# diagonalizing Hamiltonian

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_1DSSH(t, J, J2, m, n, k_x, a=0.2, b=0, l=0.1))) for k_x in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy = np.sort(Eigenvalues.imag)

J2 = 0.01

Eigenvalues_01 = np.array([(np.linalg.eigvals(Hamiltonian_1DSSH(t, J, J2, m, n, k_x, a=0.2, b=0, l=0.1))) for k_x in j0])
Real_energy_01 = np.sort(Eigenvalues_01.real)
Imaginary_energy_01 = np.sort(Eigenvalues_01.imag)

fig = plt.figure(figsize=(3.2, (2 * 3.3)/3))
plt.subplot(2, 2, 1)
plt.ylabel('Re(E) [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False)
plt.plot(j0, Real_energy_01)

plt.subplot(2, 2, 2)
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
plt.plot(j0, Real_energy)

plt.subplot(2, 2, 3)
plt.xlabel('$k_y$')
plt.ylabel('Im(E) [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.plot(j0, Imaginary_energy_01)

plt.subplot(2, 2, 4)
plt.xlabel('$k_y$')
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)
plt.plot(j0, Imaginary_energy)

plt.savefig('1D_SSH.pdf', bbox_inches='tight')
