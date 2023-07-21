from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
from Fast_algorithm import H_matrix_eigens
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')

fig = plt.figure(figsize=(6.4, (2 * 3.3)/3))

plt.subplot(2, 4, 1)
m, n, t, J2 = 10, 10, 1, 0.01
j0 = np.linspace(-2, 2, 200)  # linear space of v values

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_2DSSH(t, J, J2, m, n, w=1, a=0.2, b=0, l=0.1))) for J in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy_01 = np.sort(Eigenvalues.imag)  # Im (E) with 1% vertical coupling

plt.ylabel('Re($E$) - $\omega$\n [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False)
plt.plot(j0, Real_energy - 1)
plt.text(-1.8, 3.9, r'(a)')


plt.subplot(2, 4, 2)
J2 = 0.1
# diagonalizing Hamiltonian

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_2DSSH(t, J, J2, m, n, w=1, a=0.2, b=0, l=0.1))) for J in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy_10 = np.sort(Eigenvalues.imag)  # Im (E) with 10% vertical coupling
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
plt.plot(j0, Real_energy - 1)
plt.text(-1.8, 4, r'(b)')

# Larger system 50x100
j0 = np.linspace(-2, 2, 200)

t, m, n, w = 1, 50, 50, 1  # with a STO array of size 50 X100, H-matrix would be 5000 X 5000 dimensions
a, b, l = 0.2, 0, 0.1
J2 = 0.01

EIGENS_01 = np.empty((200, m * n * 2), complex)
for i, j in enumerate(j0):
    hh = Hamiltonian_2DSSH(t, j, J2, m, n, w, a, b, l)
    EIGENS_01[i] = H_matrix_eigens(hh, m)

plt.subplot(2, 4, 3)
# plt.ylabel('Re(E) [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
plt.plot(j0, EIGENS_01.real -1)
plt.text(-1.8, 4, r'(c)')

J2 = 0.1 #10% vertical coupling

EIGENS_10 = np.empty((200, m * n * 2), complex)
for i, j in enumerate(j0):
    hh = Hamiltonian_2DSSH(t, j, J2, m, n, w, a, b, l)
    EIGENS_10[i] = H_matrix_eigens(hh, m)

plt.subplot(2, 4, 4)
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
plt.plot(j0, EIGENS_10.real -1)
plt.text(-1.8, 4, r'(d)')


plt.subplot(2, 4, 5)

plt.xlabel(r"""$J/|\~J|$""")
plt.ylabel('Im($E$) [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.xticks(np.arange(-2, 3, 1.0))
plt.plot(j0, Imaginary_energy_01)
plt.text(-1.8, 0.06, r'(e)')

plt.subplot(2, 4, 6)

plt.xlabel(r"""$J/|\~J|$""")

plt.plot(j0, Imaginary_energy_10)
plt.text(-1.8, 0.06, r'(f)')
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)
plt.xticks(np.arange(-2, 3, 1.0))

plt.subplot(2, 4, 7)
plt.xlabel(r"""$J/|\~J|$""")
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)
plt.xticks(np.arange(-2, 3, 1.0))
# plt.ylabel('Im(E)-$\omega$ \n[arb. units]')
plt.plot(j0, np.sort(EIGENS_01.imag))
plt.text(-1.8, 0.06, r'(g)')

plt.subplot(2, 4, 8)
plt.xlabel(r"""$J/|\~J|$""")
plt.text(-1.8, 0.06, r'(h)')
plt.plot(j0, np.sort(EIGENS_10.imag))
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)
plt.xticks(np.arange(-2, 3, 1.0))

plt.savefig('50x100.pdf', bbox_inches='tight')
