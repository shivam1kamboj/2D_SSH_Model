from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
import numpy as np
import matplotlib.pyplot as plt

# Calculating energy eigen values without vertical coupling (J2); 2D will show spectrum of 1D SSH
m, n, t, J2 = 10, 10, 1, 0.1
j0 = np.linspace(-2, 2, 50)  # linear space of v values

# diagonalizing Hamiltonian

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_2DSSH(t, J, J2, m, n, w=1, a=0.2, b=0, l=0.1))) for J in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy = np.sort(Eigenvalues.imag)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlabel('Intercell coupling(J/t)')
plt.ylabel('Re(E) [arb. units]')
plt.title('J2=0.1 i.e. 10% vertical coupling strength,t=1')
plt.plot(j0, Real_energy)

plt.subplot(1, 2, 2)
plt.xlabel('Intercell coupling(J/t)')
plt.ylabel('Im(E) [arb. units]')
plt.title('J2=0.1 i.e. 10% vertical coupling strength,t=1')
plt.plot(j0, Imaginary_energy)

plt.savefig('2D_SSH_10x10.pdf')

J2 = 0.01

Eigenvalues = np.array([(np.linalg.eigvals(Hamiltonian_2DSSH(t, J, J2, m, n, w=1, a=0.2, b=0, l=0.1))) for J in j0])
Real_energy = np.sort(Eigenvalues.real)
Imaginary_energy = np.sort(Eigenvalues.imag)

fig2 = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlabel('Intercell coupling(J/t)')
plt.ylabel('Re(E) [arb. units]')
plt.title('J2=0.01 i.e. 1% vertical coupling strength,t=1')
plt.plot(j0, Real_energy)

plt.subplot(1, 2, 2)
plt.xlabel('Intercell coupling(J/t)')
plt.ylabel('Im(E) [arb. units]')
plt.title('J2=0.01 i.e. 1% vertical coupling strength,t=1')
plt.plot(j0, Imaginary_energy)

plt.savefig('2D_SSH_10x10_1%_vertical.pdf')
