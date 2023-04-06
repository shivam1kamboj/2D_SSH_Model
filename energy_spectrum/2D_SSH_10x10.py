from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('/Users/shivamkamboj/Documents/UC_Merced/Research/2D_SSH_full_project/matplotlibrc')

fig = plt.figure(figsize=(3.3, (2*3.3)/3))

plt.subplot(2, 2, 1)
m, n, t, J2 = 10, 10, 1, 0.01
j0 = np.linspace(-2, 2, 50)  # linear space of v values

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_2DSSH(t, J, J2, m, n, w=1, a=0.2, b=0, l=0.1))) for J in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy_01 = np.sort(Eigenvalues.imag)  # Im (E) with 1% vertical coupling

plt.ylabel('Re(E) [arb. units]')
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False)
plt.plot(j0, Real_energy)
plt.text(0.012, 0.02, r'(a)')


plt.subplot(2, 2, 2)
J2 = 0.1
# diagonalizing Hamiltonian

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_2DSSH(t, J, J2, m, n, w=1, a=0.2, b=0, l=0.1))) for J in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy_10 = np.sort(Eigenvalues.imag)  # Im (E) with 10% vertical coupling
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
plt.plot(j0, Real_energy)
plt.text(0.012, 0.04, r'(b)')


plt.subplot(2, 2, 3)

plt.xlabel(r"""$J/\~J$""")
plt.ylabel('Im(E) [arb. units]')

plt.plot(j0, Imaginary_energy_01)
plt.text(0.012, 0.02, r'(c)')


plt.subplot(2, 2, 4)

plt.xlabel(r"""$J/\~J$""")

plt.plot(j0, Imaginary_energy_10)
plt.text(0.012, 0.02, r'(d)')
plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)

fig.tight_layout()
plt.savefig('2D_SSH_10x10.pdf', bbox_inches='tight')
