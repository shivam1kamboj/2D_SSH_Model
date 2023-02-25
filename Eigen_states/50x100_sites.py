from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')

j0 = np.linspace(-2, 2, 50)
t = 1
m, n, w = 50, 50, 1  # with a STO array of size 50 X100, H-matrix would be 5000 X 5000 dimensions
a, b, l = 0.2, 0, 0.1
J2, J = 0.1, 0.2
HH = Hamiltonian_2DSSH(t, J, J2, m, n, w, a, b, l)
EI, VI = np.linalg.eig(HH)

igens = []  # igens are indices of Eigen values with imaginary part >=0.09
Nigens = []  # Nigens are indices of Eigen values with imaginary part <=0.09
for i in range(len(EI)):
    if EI[i].imag >= 0.09:
        igens.append(i)
    elif EI[i].imag <= -0.09:
        Nigens.append(i)
Vec = VI[:, igens]
NVec = VI[:, Nigens]

# Eigenvectors on the left edge

plt.figure()
for i, vector in enumerate(Vec.T[:4]):
    plt.subplot(3, 2, i + 1)
    plt.imshow((abs(vector).reshape(m, 2 * m)) ** 2)

plt.savefig('50x100_left_edge.pdf')

fig = plt.figure()
plt.xlabel(" Lattice index ")
plt.ylabel('$Amplitude$')
plt.title('Left Edge Eigenstates of 50x100 STO Sites With non-zero Imaginary Eigenvalue')
for i, vector in enumerate(Vec.T[:5]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0])

plt.savefig('50x100_left_edge_line_plot.pdf')

# Eigenvectors on the right edge
plt.figure()
for i, vector in enumerate(NVec.T[:4]):
    plt.subplot(3, 2, i + 1)
    plt.imshow((abs(vector).reshape(m, 2 * m)) ** 2)

plt.savefig('50x100_right_edge.pdf')

fig = plt.figure()
plt.xlabel(" Lattice index ")
plt.ylabel('$Amplitude$')
plt.title('Right Edge Eigenstates of 50x100 STO Sites With non-zero Imaginary Eigenvalue')
for i, vector in enumerate(NVec.T[:5]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1])

plt.savefig('50x100_right_edge_line_plot.pdf')

