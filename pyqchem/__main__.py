"""PyQChem -- command line interface.

Usage:
    python -m pyqchem input.inp
"""

import sys
import time
from .io.parser import InputParser
from .basis.basis_set import BasisSet
from .scf.hf import RHF
from .grad.rhf_gradient import RHFGradient
from .optimizer.optimizer import GeometryOptimizer
from .io.xyz import write_xyz


BANNER = """
 +==================================================+
 |               P y Q C h e m  v0.1                |
 |                                                  |
 |   A minimal quantum chemistry package            |
 |   Hartree-Fock / STO-3G                          |
 |                                                  |
 |   Inspired by ORCA                               |
 +==================================================+
"""


def main():
    print(BANNER)

    if len(sys.argv) < 2:
        print("Usage: python -m pyqchem <input_file>")
        print("\nExample input file (ORCA-style):")
        print("  ! HF STO-3G Opt")
        print("  * xyz 0 1")
        print("  H  0.0  0.0  0.0")
        print("  H  0.0  0.0  0.74")
        print("  *")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"  Input file: {input_file}")

    parser = InputParser()
    config = parser.parse(input_file)

    if config.molecule is None:
        print("ERROR: No molecular geometry found in input file!")
        sys.exit(1)

    print(f"  Method:     {config.method.upper()}")
    print(f"  Basis:      {config.basis_name.upper()}")
    print(f"  Job type:   {config.job_type.upper()}")
    print(f"  Charge:     {config.molecule.charge}")
    print(f"  Mult:       {config.molecule.multiplicity}")
    print(f"  Atoms:      {config.molecule.n_atoms}")
    print(f"  Electrons:  {config.molecule.n_electrons}")

    basis_set = BasisSet(config.basis_name)
    t_start = time.time()

    if config.job_type == 'sp':
        basis = basis_set.build(config.molecule)
        rhf = RHF(config.molecule, basis)
        result = rhf.compute(
            max_iter=config.scf_maxiter,
            e_conv=config.scf_e_conv,
            d_conv=config.scf_d_conv,
        )

    elif config.job_type == 'engrad':
        basis = basis_set.build(config.molecule)
        rhf = RHF(config.molecule, basis)
        result = rhf.compute(
            max_iter=config.scf_maxiter,
            e_conv=config.scf_e_conv,
            d_conv=config.scf_d_conv,
        )
        grad_comp = RHFGradient()
        gradient = grad_comp.compute(config.molecule, basis_set, RHF)

    elif config.job_type == 'opt':
        optimizer = GeometryOptimizer(
            max_steps=config.opt_maxsteps,
            grad_tol=config.opt_grad_tol,
        )
        opt_result = optimizer.optimize(config.molecule, basis_set, RHF, RHFGradient())

        # Write optimized geometry
        xyz_file = input_file.rsplit('.', 1)[0] + '_opt.xyz'
        write_xyz(config.molecule, xyz_file)
        print(f"  Optimized geometry written to: {xyz_file}")

    else:
        print(f"ERROR: Unknown job type '{config.job_type}'")
        sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\n  Wall time: {elapsed:.1f} seconds")
    print(f"\n  *** PyQChem terminated normally ***")


if __name__ == '__main__':
    main()
