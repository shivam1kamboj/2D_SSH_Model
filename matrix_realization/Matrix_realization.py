from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import matplotlib.pyplot as plt

h1 = Hamiltonian_2DSSH(t=1, J=0.6, J2=0.2, m=4, n=4, w=0.8, a=0.4, b=0.2, l=0.1).real
plt.pcolormesh(h1, edgecolors='k', linewidth=1)
ax = plt.gca()
ax.set_aspect('equal')
ax.invert_yaxis()
plt.savefig('Matrix_representation.pdf')

# I have taken 10 rows and 10 columns, we can do for arbitrary number of rows and colums: m and n need not be the same

h2 = Hamiltonian_2DSSH(t=1, J=0.6, J2=0.2, m=10, n=10, w=0.8, a=0.4, b=0.2, l=0.1).real
h3 = Hamiltonian_2DSSH(t=1, J=0.6, J2=0, m=10, n=10, w=0.8, a=0.4, b=0.2, l=0.1).real

plt.pcolormesh(h2, edgecolors='k', linewidth=1)
plt.savefig('Matrix_representation_10x10.pdf')

plt.pcolormesh(h3, edgecolors='k', linewidth=1)
plt.savefig('1D_Matrix_representation_10x10.pdf')
