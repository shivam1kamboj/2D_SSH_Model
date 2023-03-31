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

fig = plt.figure(figsize=(3.3, (2 * 3.3) / 3))
for i, vector in enumerate(Vec.T[:4]):
    ax = plt.subplot(3, 2, i + 1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i == 1 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 3 else 0
    ax.set_xticks(np.arange(0, 50, 20))
    ax.set_yticks(np.arange(0, 0.04, 0.01))
    ax.set_ylabel("$|\psi|^2$") if i % 2 == 0 else 0
    ax.set_xlabel("Site no.") if i == 2 else 0
    ax.set_xlabel("Site no.") if i == 3 else 0
    ax.plot(np.arange(50), ((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0])

plt.savefig('50x100_left_edge.pdf', bbox_inches='tight')

# Eigenvectors on the right edge
fig = plt.figure(figsize=(3.3, (2 * 3.3) / 3))
for i, vector in enumerate(NVec.T[:4]):
    ax = plt.subplot(3, 2, i + 1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i == 1 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 3 else 0
    ax.set_xticks(np.arange(0, 50, 20))
    ax.set_yticks(np.arange(0, 0.04, 0.01))
    ax.set_ylabel("$|\psi|^2$") if i % 2 == 0 else 0
    ax.set_xlabel("Site no.") if i == 2 else 0
    ax.set_xlabel("Site no.") if i == 3 else 0
    ax.plot(np.arange(50), ((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1])

plt.savefig('50x100_right_edge.pdf', bbox_inches='tight')

# Eigenvectors on the left edge first 10
fig = plt.figure(figsize=(3.3, (5*3.3)/3))
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vector in enumerate(Vec.T[:10]):
    ax = plt.subplot(5, 2, i + 1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i in list(
        np.arange(8)[::2]) else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i in list(
        np.arange(8)[1::2]) else 0
    ax.set_ylabel("$|\psi|^2$") if i in list(np.arange(10)[::2]) else 0
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True) if i == 8 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 9 else 0
    ax.set_xticks(np.arange(0, 50, 20))
    ax.set_yticks(np.arange(0, 0.04, 0.01))
    ax.set_xlabel("Site no.") if i == 8 else 0
    ax.set_xlabel("Site no.") if i == 9 else 0
    ax.plot(np.arange(50), ((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], c=colors[i],
            label=f"{np.around(EI[igens][i].real, decimals=2)}")
fig.legend(title=f"Re(E)", bbox_to_anchor=(0.98, 1.08), ncol = 5, fontsize = 6)
fig.tight_layout()
plt.savefig('50x100_left_edge_10.pdf', bbox_inches='tight')


# Eigenvectors on the right edge first 10
fig = plt.figure(figsize=(3.3, (5*3.3)/3))
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vector in enumerate(NVec.T[:10]):
    ax = plt.subplot(5, 2, i + 1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i in list(
        np.arange(8)[::2]) else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i in list(
        np.arange(8)[1::2]) else 0
    ax.set_ylabel("$|\psi|^2$") if i in list(np.arange(10)[::2]) else 0
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True) if i == 8 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 9 else 0
    ax.set_xticks(np.arange(0, 50, 20))
    ax.set_yticks(np.arange(0, 0.04, 0.01))
    ax.set_xlabel("Site no.") if i == 8 else 0
    ax.set_xlabel("Site no.") if i == 9 else 0
    ax.plot(np.arange(50), ((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1], c=colors[i],
            label=f"{np.around(EI[Nigens][i].real, decimals=2)}")
fig.legend(title=f"Re(E)", bbox_to_anchor=(0.98, 1.08), ncol = 5, fontsize = 6)
fig.tight_layout()
plt.savefig('50x100_right_edge_10.pdf', bbox_inches='tight')


fig = plt.figure(figsize=(3.3, (3 * 3.3) / 3))
plt.xlabel(" Lattice index ")
plt.ylabel('$|\psi|^2$')
for i, vector in enumerate(Vec.T[:10]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0])

plt.savefig('50x100_Left_edge_line_plot.pdf')

fig = plt.figure(figsize=(3.3, (2 * 3.3) / 3))
plt.xlabel(" Lattice index ")
plt.ylabel('$|\psi|^2$')
for i, vector in enumerate(NVec.T[:10]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1])

plt.savefig('50x100_right_edge_line_plot.pdf')

