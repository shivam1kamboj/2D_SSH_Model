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

fig = plt.figure(figsize=(3.3, (2*3.3)/3))
for i,vector in enumerate(Vec.T[:4]):
    ax = plt.subplot(3,2,i+1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
    plt.text(1.2, 2, r'(a)') if i == 0 else 0
    plt.text(1.2, 2, r'(b)') if i == 1 else 0
    plt.text(1.2, 2, r'(c)') if i == 2 else 0
    plt.text(1.2, 2, r'(d)') if i == 3 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i == 1 else 0
    ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 3 else 0
    ax.set_yticks(np.arange(0, 10, 4))
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
cbar_ax = fig.add_axes([0.83, 0.38, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_title('$|\psi|^2$')

plt.savefig('10x20_left_edge.pdf', bbox_inches='tight')

plt.figure(figsize=(3.4, (2*3.3)/3))
plt.xlabel(" Site no. ", fontsize = 11)
plt.ylabel('$|\psi|^2$', fontsize = 10)
plt.tick_params(labelsize=9)
plt.text(0, 0.19, r'(e)')
# plt.title('Left Edge Eigenstates of 10x20 STO Sites With non-zero Imaginary Eigenvalue')
for j, vector in enumerate(Vec.T[:5]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0],
             label=f"{np.around(sorted_EI[igens][j].real-1, decimals=2)}")

for j, vector in enumerate(Vec.T[5:]):
    plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], "--",
             label=f"{np.around(sorted_EI[igens][5:][j].real-1, decimals=2)}")

plt.legend(title = "$Re(E -\omega)$ $x$ $10^{-1}$", bbox_to_anchor=(1, 1.02), fontsize = 7.5)
plt.savefig('10x20_left_edge_line_plot.pdf', bbox_inches='tight')

print(EI[igens].real)

# # Eigenvectors on the right edge
#
# fig = plt.figure(figsize=(3.3, (2*3.3)/3))
# for i, vector in enumerate(NVec.T):
#     ax = plt.subplot(5, 2, i + 1)
#     im = ax.imshow((abs(vector).reshape(m, 2 * m)) ** 2)
#     plt.colorbar(im)
#     axins = ax.inset_axes([0.2, 0.3, 0.5, 0.6])
#     axins.set_xlim(0.01, 0.19)
#     axins.set_ylim(0,10)
#     # ax.indicate_inset_zoom(axin2, edgecolor="white")
#     axins.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1], np.arange(10))
#     axins.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True, labelsize=6)
#
# plt.savefig('10x20_right_edge.pdf', bbox_inches='tight')
#
# plt.figure()
# plt.xlabel(" Lattice index ")
# plt.ylabel('$|\psi|^2$')
# plt.title('Right Edge Eigenstates of 10x20 STO Sites With negavtive imaginary Eigenvalue')
# for j, vector in enumerate(NVec.T):
#     plt.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1],
#              label='eigenvalue=' + str(np.round(EI[Nigens][j], decimals=2)))
# plt.legend(bbox_to_anchor=(0.65, 1.25))
#
# plt.savefig('10x20_right_edge_line_plot.pdf')
#
# plt.figure(figsize=(8, 5))
# Real_E_edge = np.zeros(4)
# plt.ylim(1, 1.3)
# plt.xlabel('Edge State number')
# plt.ylabel('Re(E)[arbitrary units]')
# for i in range(4):
#     Real_E_edge[i] = EI[igens[:4]][i].real
#     plt.scatter(np.arange(4), Real_E_edge, label=f"Re[E] = {np.around(EI[igens][i].real, decimals=2)}")
#     plt.legend(loc='best')
#     Real_E_edge = np.zeros(4)
# plt.savefig('10x20_E[Re]_first_4_left.pdf')
#
# plt.rcParams["font.size"] = "7"
# fig = plt.figure(figsize=(3.3, (5*3.3)/3))
# colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# for i, vector in enumerate(Vec.T):
#     ax = plt.subplot(5, 2, i + 1)
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i in list(
#         np.arange(8)[::2]) else 0
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i in list(
#         np.arange(8)[1::2]) else 0
#     ax.set_ylabel("$|\psi|^2$") if i in list(np.arange(10)[::2]) else 0
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True) if i == 8 else 0
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 9 else 0
#     ax.set_xlabel("Site no.") if i == 8 else 0
#     ax.set_xlabel("Site no.") if i == 9 else 0
#     ax.plot(np.arange(10), ((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], c=colors[i],
#             label=f"{np.around(EI[igens][i].real, decimals=2)}")
# fig.legend(title=f"Re(E)", bbox_to_anchor=(0.98, 1.08), ncol = 5, fontsize = 6)
# fig.tight_layout()
# plt.savefig('10x10_left_edge_10.pdf', bbox_inches='tight')
#
#
# labels = []
# fig = plt.figure(figsize=(6.4, (2*3.3)/3))
# colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# for i, vector in enumerate(Vec.T):
#     ax = plt.subplot(3, 5, i + 1)
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i in list(np.arange(1, 5)) else 0
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i in list(np.arange(6, 10)) else 0
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
#     ax.set_yticks(np.arange(0, 10, 4))
#     ax.set_ylabel("Site no.") if i in [0, 5] else 0
#     ax.set_xlabel("Site no.") if i in [5, 6, 7, 8,  9] else 0
#     im = plt.imshow((abs(vector).reshape(m, 2 * m)) ** 2, cmap='Greys')
#     # clb = plt.colorbar(im) if i == 9 else 0
#     # clb.ax.set_title('$|\psi|^2$') if i ==9 else 0
#     axins = ax.inset_axes([0.4, 0.4, 0.4, 0.6])
#     axins.set_xlim(0.01, 0.19)
#     axins.set_ylim(0, 9.5)
#     # axins.set_ylabel('Site No.', fontsize = 6)
#     axins.set_xlabel("$|\psi|^2$", fontsize=6)
#     #     ax.indicate_inset_zoom(axins, edgecolor="black",lw =3)
#     axins.tick_params(labelbottom=False, labelsize=6)
#     label, = axins.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, 0], np.arange(10), color = colors[i])
#     labels.append(label)
# fig.legend(labels, list(np.around(EI[igens], decimals = 2).real), title=f"Re(E)", bbox_to_anchor=(0.96, 1.08),
#            ncol = 10, fontsize = 6.5)
#
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.83, 0.38, 0.02, 0.5])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_title('$|\psi|^2$')
#
# # fig.tight_layout()
# plt.savefig('left_edge_10_01.pdf', bbox_inches='tight')
#
#
# fig, ax = plt.subplots(2, 2, figsize=(3.3, (2*3.3)/3))
# for i, vector in enumerate(Vec.T[:4]):
#     ax = plt.subplot(3, 2, i+1)
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i == 1 else 0
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i == 3 else 0
#     ax.set_yticks(np.arange(0, 10, 4))
#     ax.set_ylabel("Site no.") if i % 2 == 0 else 0
#     ax.set_xlabel("Site no.") if i == 2 else 0
#     ax.set_xlabel("Site no.") if i == 3 else 0
#     im = ax.imshow((abs(vector).reshape(m, 2*m))**2 + (abs(NVec.T[i]).reshape(m, 2*m))**2, cmap='Greys')
#
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.83, 0.38, 0.02, 0.5])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_title('$|\psi|^2$')
#
# plt.savefig('Both_edges_10.pdf', bbox_inches='tight')
#
#
# labels = []
# fig = plt.figure(figsize=(6.4, (2*3.3)/3))
# colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# for i, vector in enumerate(NVec.T):
#     ax = plt.subplot(2, 5, i + 1)
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False) if i in list(np.arange(1, 5)) else 0
#     ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True) if i in list(np.arange(6, 10)) else 0
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False) if i == 0 else 0
#     ax.set_yticks(np.arange(0, 10, 4))
#     ax.set_ylabel("Site no.") if i in [0, 5] else 0
#     ax.set_xlabel("Site no.") if i in [5, 6, 7, 8,  9] else 0
#     im = plt.imshow((abs(vector).reshape(m, 2 * m)) ** 2, cmap='Greys')
#     clb = plt.colorbar(im) if i == 9 else 0
#     clb.ax.set_title('$|\psi|^2$') if i ==9 else 0
#     axins = ax.inset_axes([0.4, 0.4, 0.4, 0.6])
#     axins.set_xlim(0.01, 0.19)
#     axins.set_ylim(0, 9.5)
#     # axins.set_ylabel('Site No.', fontsize = 6)
#     axins.set_xlabel("$|\psi|^2$", fontsize=6)
#     #     ax.indicate_inset_zoom(axins, edgecolor="black",lw =3)
#     axins.tick_params(labelbottom=False, labelsize=6)
#     label, = axins.plot(((abs(vector).reshape(m, 2 * m)) ** 2)[:, -1], np.arange(10), color = colors[i])
#     labels.append(label)
# fig.legend(labels, list(np.around(EI[Nigens], decimals = 2).real), title=f"Re(E)", bbox_to_anchor=(0.96, 1.08),
#            ncol = 10, fontsize = 6.5)
#
# fig.tight_layout()
# plt.savefig('Right_edge_10_01.pdf', bbox_inches='tight')
