from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')

j0 = np.linspace(-2, 2, 50)
t = 1
m, n, w = 10, 10, 1  # with a STO array of size 50 X50, H-matrix would be 5000 X 5000 dimensions
a, b, l = 0.2, 0, 0.1 # a is J_SA , b is J_SB and l is alpha
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

# Eigenvectors on the left edge

# fig = plt.figure(figsize=(3.3, (4*3.3)/3))
fig, ax = plt.subplots(3, 2, figsize=(3.4, (3.6*3.3)/3))
for i, vector in enumerate(Vec.T[:4]):
    ax = plt.subplot(5, 2, i+1)
    plt.subplots_adjust(hspace=0.1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
    plt.text(1.2, 2, r'(a)') if i == 0 else 0
    plt.text(1.2, 2, r'(b)') if i == 1 else 0
    plt.text(1.2, 2, r'(c)') if i == 2 else 0
    plt.text(1.2, 2, r'(d)') if i == 3 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i == 1 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 3 else 0
    ax.set_yticks(np.arange(0, 10, 4))
    ax.set_xticks([0, 8, 16])
    ax.set_ylabel("Site no.") if i % 2 == 0 else 0
    ax.set_xlabel("Site no.") if i == 2 else 0
    ax.set_xlabel("Site no.") if i == 3 else 0
    im = ax.imshow((abs(vector).reshape(m,2*m))**2, cmap = 'Greys')
#     plt.colorbar(im) if i%2!=0 else 0
    axins = ax.inset_axes([0.5,0.4,0.5,0.6])
    axins.set_xlim(0.01,0.19)
    axins.set_ylim(0,9.5)
    # axins.set_ylabel('Site No.', fontsize = 6)
    axins.set_xlabel("$|\psi|^2$", fontsize = 6)
#     ax.indicate_inset_zoom(axins, edgecolor="black",lw =3)
    axins.tick_params(labelbottom=False, labelsize=6)
    axins.plot(((abs(vector).reshape(m, 2*m))**2)[:, 0], np.arange(10))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.585, 0.02, 0.29])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_title('$|\psi|^2$')

# plt.savefig('10x20_left_edge.pdf', bbox_inches='tight')

# plt.figure(figsize=(3.4, (2*3.3)/3))
ax4 = plt.subplot2grid((4, 2), (2, 0), rowspan=2, colspan=2)
ax4.set_xlabel(" Site no. ")
ax4.set_ylabel('$|\psi|^2$')
# plt.tick_params(labelsize=9)
plt.text(0.16, 0.16, r'(e)')
# plt.title('Left Edge Eigenstates of 10x20 STO Sites With non-zero Imaginary Eigenvalue')
for j, vector in enumerate(Vec.T[:5]):
    ax4.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0],
             label=f"{np.around(sorted_EI[igens][j].real-1, decimals=2)}")

for j, vector in enumerate(Vec.T[5:]):
    ax4.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], "--",
             label=f"{np.around(sorted_EI[igens][5:][j].real-1, decimals=2)}")

ax4.legend(title = "Re($E$) -$\omega$", bbox_to_anchor=(1, 1.06), labelspacing=0.3)

plt.savefig('10x20_left_edge_line_plot.pdf', bbox_inches='tight')