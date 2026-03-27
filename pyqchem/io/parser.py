"""ORCA-like input file parser."""

from ..mol.molecule import Atom, Molecule


class JobConfig:
    """Configuration parsed from an input file."""
    def __init__(self):
        self.method = 'hf'
        self.basis_name = 'sto-3g'
        self.job_type = 'sp'   # 'sp', 'opt', 'engrad'
        self.molecule = None
        self.scf_maxiter = 128
        self.scf_e_conv = 1e-10
        self.scf_d_conv = 1e-8
        self.opt_maxsteps = 100
        self.opt_grad_tol = 3e-4


class InputParser:
    """Parse ORCA-style input files.

    Example input:
        ! HF STO-3G Opt
        * xyz 0 1
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74
        *
    """

    def parse(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        return self.parse_lines(lines)

    def parse_lines(self, lines):
        config = JobConfig()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue

            # Keyword line
            if line.startswith('!'):
                self._parse_keywords(line[1:], config)
                i += 1
                continue

            # Coordinate block
            if line.startswith('*'):
                parts = line.split()
                if len(parts) >= 4 and parts[1].lower() == 'xyz':
                    charge = int(parts[2])
                    mult = int(parts[3])
                    atoms = []
                    i += 1
                    while i < len(lines):
                        aline = lines[i].strip()
                        if aline.startswith('*') or not aline:
                            if aline.startswith('*'):
                                i += 1
                            break
                        parts_a = aline.split()
                        if len(parts_a) >= 4:
                            sym = parts_a[0]
                            x, y, z = float(parts_a[1]), float(parts_a[2]), float(parts_a[3])
                            atoms.append(Atom(sym, [x, y, z]))
                        i += 1
                    config.molecule = Molecule(atoms, charge, mult)
                    continue
                i += 1
                continue

            # %scf block
            if line.lower().startswith('%scf'):
                i += 1
                while i < len(lines):
                    sline = lines[i].strip().lower()
                    if sline == 'end':
                        i += 1
                        break
                    if 'maxiter' in sline:
                        config.scf_maxiter = int(sline.split()[-1])
                    i += 1
                continue

            # %geom block
            if line.lower().startswith('%geom'):
                i += 1
                while i < len(lines):
                    sline = lines[i].strip().lower()
                    if sline == 'end':
                        i += 1
                        break
                    if 'maxstep' in sline or 'maxiter' in sline:
                        config.opt_maxsteps = int(sline.split()[-1])
                    i += 1
                continue

            i += 1

        return config

    def _parse_keywords(self, keyword_str, config):
        tokens = keyword_str.lower().split()
        for t in tokens:
            if t in ('hf', 'rhf'):
                config.method = 'hf'
            elif t in ('sto-3g', 'sto3g'):
                config.basis_name = 'sto-3g'
            elif t in ('sp', 'singlepoint', 'energy'):
                config.job_type = 'sp'
            elif t in ('opt', 'geoopt', 'optimization'):
                config.job_type = 'opt'
            elif t in ('engrad', 'gradient', 'grad'):
                config.job_type = 'engrad'
