from Hamiltonian.Hamiltonian import Hamiltonian_2DSSH
from Fast_algorithm import H_matrix_eigens
import numpy as np
import matplotlib.pyplot as plt

j0 = np.linspace(-2, 2, 50)

t, m, n, w = 1, 50, 50, 1  # with a STO array of size 50 X100, H-matrix would be 5000 X 5000 dimensions
a, b, l = 0.2, 0, 0.1
J2 = 0.1

EIGENS = np.empty((50, m * n * 2), complex)
for i, j in enumerate(j0):
    hh = Hamiltonian_2DSSH(t, j, J2, m, n, w, a, b, l)
    EIGENS[i] = H_matrix_eigens(hh, m)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlabel('Intercell coupling(J/t)')
plt.ylabel('Re(E) [arb. units]')
plt.title('J2=0.1 i.e. 10% vertical coupling strength,t=1')
plt.plot(j0, EIGENS.real)

plt.subplot(1, 2, 2)
plt.xlabel('Intercell coupling(J/t)')
plt.ylabel('Im(E) [arb. units]')
plt.title('J2=0.1 i.e. 10% vertical coupling strength,t=1')
plt.plot(j0, np.sort(EIGENS.imag))

plt.savefig('2D_SSH_50x100_10%_vertical.pdf')
