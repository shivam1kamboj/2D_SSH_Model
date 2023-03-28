from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
from Fast_algorithm import H_matrix_eigens
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')

j0 = np.linspace(-2, 2, 50)

t, m, n, w = 1, 50, 50, 1  # with a STO array of size 50 X100, H-matrix would be 5000 X 5000 dimensions
a, b, l = 0.2, 0, 0.1
J2 = 0.01

EIGENS_01 = np.empty((50, m * n * 2), complex)
for i, j in enumerate(j0):
    hh = Hamiltonian_2DSSH(t, j, J2, m, n, w, a, b, l)
    EIGENS_01[i] = H_matrix_eigens(hh, m)

fig = plt.figure(figsize=(3.3, (2 * 3.3)/3))
plt.subplot(2, 2, 1)
plt.ylabel('Re(E) [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False)
plt.plot(j0, EIGENS_01.real)

J2 = 0.1 #10% vertical coupling

EIGENS_10 = np.empty((50, m * n * 2), complex)
for i, j in enumerate(j0):
    hh = Hamiltonian_2DSSH(t, j, J2, m, n, w, a, b, l)
    EIGENS_10[i] = H_matrix_eigens(hh, m)

plt.subplot(2, 2, 2)
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
plt.plot(j0, EIGENS_10.real)


plt.subplot(2, 2, 3)
plt.xlabel(r"""$J/\~J$""")
plt.ylabel('Im(E) [arb. units]')
plt.plot(j0, np.sort(EIGENS_01.imag))

plt.subplot(2, 2, 4)
plt.xlabel(r"""$J/\~J$""")

plt.plot(j0, np.sort(EIGENS_10.imag))
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)

plt.savefig('2D_SSH_50x100.pdf', bbox_inches='tight')
