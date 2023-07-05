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

args = np.argsort(EI)

sorted_vectors = VI[:, args]
sorted_EI = EI[args]

igens = []  # igens are indices of Eigen values with imaginary part >=0.09
Nigens = []  # Nigens are indices of Eigen values with imaginary part <=0.09
for i in range(len(EI)):
    if sorted_EI[i].imag >= 0.09:
        igens.append(i)
    elif sorted_EI[i].imag <= -0.09:
        Nigens.append(i)


Vec = sorted_vectors[:, igens]
NVec = sorted_vectors[:, Nigens]


# Eigenvectors on the left edge first 10 plotted in two rows
fig = plt.figure(figsize=(6.4, (2 * 3.3) / 3))
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vector in enumerate(Vec.T[:10]):
    ax = plt.subplot(2, 5, i + 1)
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i in list(np.arange(1, 5)) else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i in list(np.arange(6, 10)) else 0
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
    ax.set_yticks(np.arange(0, 10, 4))
    ax.set_ylabel("Site no.") if i in [0, 5] else 0
    ax.set_xlabel("Site no.") if i in [5, 6, 7, 8, 9] else 0

    ax.plot(np.arange(50), ((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], c=colors[i],
            label=f"{np.around(sorted_EI[igens][i].real, decimals=2)}")

fig.legend(title=f"Re(E)", bbox_to_anchor=(0.96, 1.08), ncol=10, fontsize=6)
fig.tight_layout()
plt.savefig('left_edge_50_01.pdf', bbox_inches='tight')

# Plots in 5 rows:

# Eigenvectors on the left edge first 10
fig = plt.figure(figsize=(3.3, (5 * 3.3) / 3))
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
            label=f"{np.around(10 * (sorted_EI[igens][:10][i].real - 1), decimals=2)}")
fig.legend(title="$Re(E -\omega)$ $x$ $10^{-1}$", bbox_to_anchor=(0.96, 1.08), ncol=5, fontsize=6)
fig.tight_layout()
plt.savefig('50x100_left_10.pdf', bbox_inches='tight')

# Eigenvectors on the left edge 10:20
fig = plt.figure(figsize=(3.3, (5 * 3.3) / 3))
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vector in enumerate(Vec.T[10:20]):
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
            label=f"{np.around(10 * (sorted_EI[igens][10:20][i].real - 1), decimals=2)}")
fig.legend(title="$Re(E -\omega)$ $x$ $10^{-1}$", bbox_to_anchor=(0.96, 1.08), ncol=5, fontsize=6)
fig.tight_layout()
plt.savefig('50x100_left_20.pdf', bbox_inches='tight')

# Eigenvectors on the left edge 20:30
fig = plt.figure(figsize=(3.3, (5 * 3.3) / 3))
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vector in enumerate(Vec.T[20:30]):
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
            label=f"{np.around(10 * (sorted_EI[igens][20:30][i].real - 1), decimals=2)}")
fig.legend(title="$Re(E -\omega)$ $x$ $10^{-1}$", bbox_to_anchor=(0.96, 1.08), ncol=5, fontsize=6)
fig.tight_layout()
plt.savefig('50x100_left_30.pdf', bbox_inches='tight')


# Eigenvectors on the left edge 25:35
fig = plt.figure(figsize=(3.3, (5 * 3.3) / 3))
colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, vector in enumerate(Vec.T[25:35]):
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
            label=f"{np.around(10 * (sorted_EI[igens][25:35][i].real - 1), decimals=2)}")
fig.legend(title="$Re(E -\omega)$ $x$ $10^{-1}$", bbox_to_anchor=(0.96, 1.08), ncol=5, fontsize=6)
fig.tight_layout()
plt.savefig('50x100_left_35.pdf', bbox_inches='tight')