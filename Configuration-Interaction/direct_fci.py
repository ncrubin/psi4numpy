"""
A Psi4 input script to compute Full Configuration Interaction from a SCF reference

Requirements:
SciPy 0.13.0+, NumPy 1.7.2+

References:
Equations from [Szabo:1996]
"""

__authors__ = "Tianyuan Zhang"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-26"

import sys
import time
import numpy as np
from scipy.sparse.linalg import eigsh
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Check energy against psi4?
compare_psi4 = True

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

# mol = psi4.geometry("""
# O
# H 1 1.1
# H 1 1.1 2 104
# symmetry c1
# """)

mol = psi4.geometry("""
O
C 1 1.128
symmetry c1
""")



psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from wavfunction class
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()

# Compute size of Hamiltonian in GB
from scipy.special import comb
nDet = comb(nmo, ndocc)**2
H_Size = nDet**2 * 8e-9
print('\nSize of the Hamiltonian Matrix will be %4.2f GB.' % H_Size)
if H_Size > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (H_Size, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

#Make spin-orbital MO
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

from direct_helper_ci import Determinant, HamiltonianGenerator
from itertools import combinations

print('Generating %d Full CI Determinants...' % (nDet))
t = time.time()
detList = []
for alpha in combinations(range(nmo), ndocc):
    for beta in combinations(range(nmo), ndocc):
        detList.append(Determinant(alphaObtList=alpha, betaObtList=beta))

print('..finished generating determinants in %.3f seconds.\n' % (time.time() - t))

print('Generating Hamiltonian Matrix...')

t = time.time()
Hamiltonian_generator = HamiltonianGenerator(H, MO)
Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)

print('..finished generating Matrix in %.3f seconds.\n' % (time.time() - t))

print('Diagonalizing Hamiltonian Matrix...')

t = time.time()

# e_fci, wavefunctions = np.linalg.eigh(Hamiltonian_matrix)
e_fci, wavefunctions = eigsh(Hamiltonian_matrix)
print('..finished diagonalization in %.3f seconds.\n' % (time.time() - t))

fci_mol_e = e_fci[0] + mol.nuclear_repulsion_energy()

print('# Determinants:     % 16d' % (len(detList)))

print('SCF energy:         % 16.10f' % (scf_e))
print('FCI correlation:    % 16.10f' % (fci_mol_e - scf_e))
print('Total FCI energy:   % 16.10f' % (fci_mol_e))

gs_wf = wavefunctions[:, [0]]
with open('fci_vec_co.dat', 'w') as fid:
    for idx, det in enumerate(detList):
        bits = np.zeros(nmo * 2, dtype=int)
        alpha_bits, beta_bits = det.getOrbitalIndexLists()
        print(alpha_bits, str(alpha_bits), beta_bits, str(beta_bits))
        for a, b in zip(alpha_bits, beta_bits):
            bits[2 * a] = 1
            bits[2 * b + 1] = 1
        print(idx, det, det.alphaObtBits, det.betaObtBits, bits, gs_wf[idx, 0])
        if abs(gs_wf[idx, 0]) > 1.0E-14:
            fid.write("{} {:5.15f} {:5.15f}\n".format("".join([str(int(x)) for x in bits]),
                                                      gs_wf[idx, 0].real, gs_wf[idx, 0].imag))


if compare_psi4:
    psi4.compare_values(psi4.energy('FCI'), fci_mol_e, 6, 'FCI Energy')
