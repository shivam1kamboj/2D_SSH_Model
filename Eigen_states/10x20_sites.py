from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')

j0 = np.linspace(-2, 2, 50)
t = 1
m, n, w = 10, 10, 1  # with a STO array of size 50 X50, H-matrix would be 5000 X 5000 dimensions
a, b, l = 0.2, 0, 0.1
J2, J = 0.1, 0.2
HH = Hamiltonian_2DSSH(t, J, J2, m, n, w, a, b, l)
EI, VI = np.linalg.eig(HH)

igens = []  # igens are indices of Eigen values with imaginary part >=0.09
for i in range(len(EI)):
    if abs(EI[i].imag) >= 0.09:
        igens.append(i)
Vec = VI[:, igens]

# Eigenvectors on the left edge

plt.figure()
for i, vector in enumerate(Vec.T[:4]):
    plt.subplot(2, 2, i + 1)
    plt.imshow((abs(vector).reshape(m, 2 * m)) ** 2)

plt.savefig('10x20_left_edge.pdf')

plt.figure()
plt.xlabel(" Lattice index ")
plt.ylabel('$|\psi|^2$')
plt.title('Left Edge Eigenstates of 10x20 STO Sites With non-zero Imaginary Eigenvalue')
for j, vector in enumerate(Vec.T[:10]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], label='eigenvalue=' + str(np.round(EI[igens][j], decimals=2)))
    plt.legend(loc='best', fontsize=7)

plt.savefig('10x20_left_edge_line_plot.pdf')

# Eigenvectors on the right edge

plt.figure()
for i, vector in enumerate(Vec.T[10:14]):
    plt.subplot(3, 2, i + 1)
    plt.imshow((abs(vector).reshape(m, 2 * m)) ** 2)

plt.savefig('10x20_right_edge.pdf')

plt.figure()
plt.xlabel(" Lattice index ")
plt.ylabel('$|\psi|^2$')
plt.title('Right Edge Eigenstates of 10x20 STO Sites With negavtive imaginary Eigenvalue')
for j, vector in enumerate(Vec.T[10:]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1], label='eigenvalue=' + str(np.round(EI[igens][j], decimals=2)))
    plt.legend(loc='best', fontsize=7)

plt.savefig('10x20_right_edge_line_plot.pdf')
