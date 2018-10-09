import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Check energy against psi4?
compare_psi4 = True

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# mol = psi4.geometry("""
# O
# C 1 1.128
# symmetry c1
# """)



psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

print('\nStarting FCI execution...')
t = time.time()

# First compute FCI energy using Psi4
fci_e, wfn = psi4.energy('FCI', return_wfn=True)

print(np.array(wfn.S()))
print(wfn.new_civector())
print(wfn.nmo())
print(wfn.ndet())
# print(wfn.variables())
# print(wfn.__attributes__)

# dvec = wfn.new_civector(1, 53, True, True)
# dvec.set_nvec(1)
# dvec.init_io_files(True)
# print(dvec.read(0, 0))
# C0 = np.array(dvec)
# print(C0)