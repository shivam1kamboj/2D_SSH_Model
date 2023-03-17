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
    ax = plt.subplot(3, 2, i + 1)
    im = ax.imshow((abs(vector).reshape(m, 2 * m)) ** 2)
    plt.colorbar(im)
    axins = ax.inset_axes([0.5, 0.4, 0.5, 0.6])
    axins.set_xlim(0.01, 0.19)
    axins.set_ylim(0, 9.5)
    ax.indicate_inset_zoom(axins, edgecolor="white")
    axins.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], np.arange(10))
plt.savefig('10x20_left_edge.pdf', bbox_inches='tight')

plt.figure()
plt.xlabel(" Lattice index ")
plt.ylabel('$|\psi|^2$')
plt.title('Left Edge Eigenstates of 10x20 STO Sites With non-zero Imaginary Eigenvalue')
for j, vector in enumerate(Vec.T):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0],
             label=f"State with Re(E)={np.around(EI[igens][i].real, decimals=2)}")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, fancybox=True, shadow=True)

plt.savefig('10x20_left_edge_line_plot.pdf', bbox_inches='tight')

# Eigenvectors on the right edge

plt.figure()
for i, vector in enumerate(NVec.T):
    ax = plt.subplot(5, 2, i + 1)
    im = ax.imshow((abs(vector).reshape(m, 2 * m)) ** 2)
    plt.colorbar(im)
    axins = ax.inset_axes([0.2, 0.3, 0.5, 0.6])
    axins.set_xlim(0.01, 0.19)
    axins.set_ylim(0,10)
    # ax.indicate_inset_zoom(axin2, edgecolor="white")
    axins.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1], np.arange(10))
    axins.tick_params(left=False,
                    bottom=False,
                    labelleft=True,
                    labelbottom=True, labelsize=6)

plt.savefig('10x20_right_edge.pdf', bbox_inches='tight')

plt.figure()
plt.xlabel(" Lattice index ")
plt.ylabel('$|\psi|^2$')
plt.title('Right Edge Eigenstates of 10x20 STO Sites With negavtive imaginary Eigenvalue')
for j, vector in enumerate(NVec.T):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1],
             label='eigenvalue=' + str(np.round(EI[Nigens][j], decimals=2)))
plt.legend(bbox_to_anchor=(0.65, 1.25))

plt.savefig('10x20_right_edge_line_plot.pdf')

plt.figure(figsize=(8, 5))
Real_E_edge = np.zeros(4)
plt.ylim(1, 1.3)
plt.xlabel('Edge State number')
plt.ylabel('Re(E)[arbitrary units]')
for i in range(4):
    Real_E_edge[i] = EI[igens[:4]][i].real
    plt.scatter(np.arange(4), Real_E_edge, label=f"Re[E] = {np.around(EI[igens][i].real, decimals=2)}")
    plt.legend(loc='best')
    Real_E_edge = np.zeros(4)
plt.savefig('10x20_E[Re]_first_4_left.pdf')
