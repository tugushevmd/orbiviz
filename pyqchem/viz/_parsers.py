"""Parsers for common quantum chemistry output formats.

Supported formats:
- Gaussian .log / .out  — geometry, charges (Mulliken, NBO, CHELPG, APT), Fukui
- Gaussian .fchk        — geometry, MO coefficients, density, basis info
- Molden .molden        — geometry, MO data
- ORCA .out             — geometry, charges (Mulliken, Löwdin, CHELPG), Mayer bond orders
- Generic XYZ           — geometry (handled in _geometry.py)
- Gaussian cube         — volumetric data (handled in _geometry.py)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ._elements import Z_TO_SYMBOL, ELEMENTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOHR_TO_ANG = 0.52917721067


def _read_text_file(path: Path) -> str:
    """Read a text file while tolerating common Windows shell encodings."""
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Gaussian .log / .out parser
# ---------------------------------------------------------------------------

class GaussianLogParser:
    """Parse a Gaussian log/output file for geometry, charges, and properties."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text = _read_text_file(self.path)
        self._lines = self._text.splitlines()

    def get_geometry(self, which: str = "last") -> list[dict]:
        """Extract geometry from 'Standard orientation' or 'Input orientation'.

        Parameters
        ----------
        which : str
            'last' for the final (optimised) geometry, 'first' for the input.
        """
        # Find all orientation blocks
        pattern = re.compile(
            r"(Standard orientation|Input orientation).*?\n"
            r"\s*-+\n"
            r"\s*Center\s+Atomic\s+Atomic\s+Coordinates.*?\n"
            r"\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\n"
            r"\s*-+\n"
            r"(.*?)\n"
            r"\s*-+",
            re.DOTALL,
        )
        matches = list(pattern.finditer(self._text))
        if not matches:
            raise ValueError(f"No geometry found in {self.path}")

        block = matches[-1] if which == "last" else matches[0]
        atoms = []
        for line in block.group(2).strip().splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            z_num = int(parts[1])
            sym = Z_TO_SYMBOL.get(z_num, str(z_num))
            atoms.append({
                "atom_index": int(parts[0]),
                "element": sym,
                "x": float(parts[3]),
                "y": float(parts[4]),
                "z": float(parts[5]),
            })
        return atoms

    def get_mulliken_charges(self) -> list[dict]:
        """Extract Mulliken atomic charges."""
        return self._parse_charge_block("Mulliken charges:", skip_header=1)

    def get_apt_charges(self) -> list[dict]:
        """Extract APT atomic charges."""
        return self._parse_charge_block("APT charges:", skip_header=1)

    def get_nbo_charges(self) -> list[dict]:
        """Extract NBO natural charges from Natural Population Analysis."""
        pattern = re.compile(
            r"Summary of Natural Population Analysis:.*?\n"
            r"\s*-+\n"
            r"\s*Atom\s+No\s+.*?\n"
            r"\s*-+\n"
            r"(.*?)\n"
            r"\s*[=]+",
            re.DOTALL,
        )
        match = pattern.search(self._text)
        if not match:
            raise ValueError(f"No NBO charges found in {self.path}")
        rows = []
        for line in match.group(1).strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append({
                "atom_index": int(parts[1]),
                "element": parts[0],
                "charge": float(parts[2]),
            })
        return rows

    def get_chelpg_charges(self) -> list[dict]:
        """Extract CHELPG or ESP-fit charges."""
        # Gaussian prints ESP charges after 'Charges from ESP fit'
        return self._parse_charge_block(
            "Charges from ESP fit", skip_header=2,
        )

    def get_energy(self) -> float:
        """Extract the last SCF energy (Hartree)."""
        pattern = re.compile(r"SCF Done:.*?=\s*([-.\d]+)")
        matches = pattern.findall(self._text)
        if not matches:
            raise ValueError(f"No SCF energy found in {self.path}")
        return float(matches[-1])

    def get_dipole_moment(self) -> tuple[float, float, float, float]:
        """Extract dipole moment (X, Y, Z, Total) in Debye."""
        pattern = re.compile(
            r"Dipole moment \(field-independent basis, Debye\):\s*\n"
            r"\s*X=\s*([-.\d]+)\s+Y=\s*([-.\d]+)\s+Z=\s*([-.\d]+)\s+Tot=\s*([-.\d]+)"
        )
        match = pattern.search(self._text)
        if not match:
            raise ValueError(f"No dipole moment found in {self.path}")
        return tuple(float(match.group(i)) for i in range(1, 5))

    def _parse_charge_block(self, header: str, skip_header: int = 1) -> list[dict]:
        """Generic parser for Gaussian charge table blocks."""
        idx = None
        for i, line in enumerate(self._lines):
            if header in line:
                idx = i
        if idx is None:
            raise ValueError(f"Block '{header}' not found in {self.path}")
        rows = []
        start = idx + skip_header + 1  # skip the header line(s)
        for line in self._lines[start:]:
            parts = line.split()
            if len(parts) < 3:
                break
            try:
                atom_idx = int(parts[0])
                charge = float(parts[2])
            except (ValueError, IndexError):
                break
            rows.append({
                "atom_index": atom_idx,
                "element": parts[1],
                "charge": charge,
            })
        return rows


# ---------------------------------------------------------------------------
# Gaussian .fchk parser
# ---------------------------------------------------------------------------

class FchkParser:
    """Parse a Gaussian formatted checkpoint (.fchk) file.

    Extracts geometry, basis info, MO coefficients, energies, etc.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lines = _read_text_file(self.path).splitlines()
        self._index: dict[str, int] = {}
        self._build_index()

    def _build_index(self):
        """Build an index of section headers for fast lookup."""
        for i, line in enumerate(self._lines):
            if len(line) >= 40 and (line[43:44] in ("I", "R", "C")):
                key = line[:40].strip()
                self._index[key] = i

    def _read_scalar(self, key: str) -> int | float:
        """Read a scalar integer or real value."""
        idx = self._index.get(key)
        if idx is None:
            raise KeyError(f"Section '{key}' not found in {self.path}")
        line = self._lines[idx]
        typ = line[43]
        val_str = line[49:].strip()
        return int(val_str) if typ == "I" else float(val_str)

    def _read_array(self, key: str) -> np.ndarray:
        """Read an array of integers or reals following a header line."""
        idx = self._index.get(key)
        if idx is None:
            raise KeyError(f"Section '{key}' not found in {self.path}")
        line = self._lines[idx]
        typ = line[43]
        n = int(line[49:].strip().split()[-1])
        values = []
        i = idx + 1
        while len(values) < n:
            values.extend(self._lines[i].split())
            i += 1
        if typ == "I":
            return np.array([int(x) for x in values[:n]], dtype=int)
        return np.array([float(x) for x in values[:n]], dtype=float)

    def get_geometry(self) -> list[dict]:
        """Extract atomic geometry (Å)."""
        z_nums = self._read_array("Atomic numbers")
        coords = self._read_array("Current cartesian coordinates")
        coords_ang = coords.reshape(-1, 3) * BOHR_TO_ANG
        atoms = []
        for i, (z, xyz) in enumerate(zip(z_nums, coords_ang), start=1):
            sym = Z_TO_SYMBOL.get(int(z), str(z))
            atoms.append({
                "atom_index": i,
                "element": sym,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            })
        return atoms

    def get_n_atoms(self) -> int:
        return int(self._read_scalar("Number of atoms"))

    def get_n_basis(self) -> int:
        return int(self._read_scalar("Number of basis functions"))

    def get_n_electrons(self) -> int:
        return int(self._read_scalar("Number of electrons"))

    def get_alpha_mo_coefficients(self) -> np.ndarray:
        """Return alpha MO coefficient matrix (n_basis x n_basis), column-major."""
        n = self.get_n_basis()
        coeffs = self._read_array("Alpha MO coefficients")
        return coeffs.reshape(n, n)

    def get_alpha_orbital_energies(self) -> np.ndarray:
        """Return alpha orbital energies (Hartree)."""
        return self._read_array("Alpha Orbital Energies")

    def get_total_energy(self) -> float:
        return float(self._read_scalar("Total Energy"))

    def get_total_scf_density(self) -> np.ndarray:
        """Return lower-triangular SCF density matrix."""
        return self._read_array("Total SCF Density")

    def get_shell_types(self) -> np.ndarray:
        return self._read_array("Shell types")

    def get_n_primitives_per_shell(self) -> np.ndarray:
        return self._read_array("Number of primitives per shell")

    def get_shell_to_atom_map(self) -> np.ndarray:
        return self._read_array("Shell to atom map")

    def get_primitive_exponents(self) -> np.ndarray:
        return self._read_array("Primitive exponents")

    def get_contraction_coefficients(self) -> np.ndarray:
        return self._read_array("Contraction coefficients")


# ---------------------------------------------------------------------------
# ORCA output parser
# ---------------------------------------------------------------------------

class OrcaOutputParser:
    """Parse an ORCA output file for geometry, charges, and bond orders."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text = _read_text_file(self.path)
        self._lines = self._text.splitlines()

    def get_geometry(self, which: str = "last") -> list[dict]:
        """Extract Cartesian coordinates from ORCA output.

        Looks for 'CARTESIAN COORDINATES (ANGSTROEM)' blocks.
        """
        header = "CARTESIAN COORDINATES (ANGSTROEM)"
        indices = [i for i, line in enumerate(self._lines) if header in line]
        if not indices:
            raise ValueError(f"No geometry found in {self.path}")
        idx = indices[-1] if which == "last" else indices[0]
        atoms = []
        for j, line in enumerate(self._lines[idx + 2:], start=1):
            parts = line.split()
            if len(parts) < 4:
                break
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                break
            atoms.append({
                "atom_index": j,
                "element": parts[0],
                "x": x, "y": y, "z": z,
            })
        return atoms

    def _parse_orca_charge_block(self, header: str) -> list[dict]:
        """Generic parser for ORCA charge blocks (Mulliken, Löwdin)."""
        idx = None
        for i, line in enumerate(self._lines):
            if header in line:
                idx = i
        if idx is None:
            raise ValueError(f"No '{header}' found in {self.path}")
        rows = []
        charge_re = re.compile(r"\s*(\d+)\s+(\w+)\s*:\s*([-.\d]+)")
        for line in self._lines[idx + 1:]:
            m = charge_re.match(line)
            if m:
                rows.append({
                    "atom_index": int(m.group(1)) + 1,  # ORCA is 0-based
                    "element": m.group(2),
                    "charge": float(m.group(3)),
                })
            elif rows:
                # Already found data, non-matching line = end of block
                break
            # Skip separator lines (---) before data starts
        return rows

    def get_mulliken_charges(self) -> list[dict]:
        """Extract Mulliken atomic charges."""
        return self._parse_orca_charge_block("MULLIKEN ATOMIC CHARGES")

    def get_loewdin_charges(self) -> list[dict]:
        """Extract Loewdin atomic charges."""
        return self._parse_orca_charge_block("LOEWDIN ATOMIC CHARGES")

    def get_chelpg_charges(self) -> list[dict]:
        """Extract CHELPG atomic charges."""
        return self._parse_orca_charge_block("CHELPG Charges")

    def get_mayer_bond_orders(self) -> list[dict]:
        """Extract Mayer bond order matrix.

        Returns list of dicts: {atom_i, atom_j, element_i, element_j, bond_order}.
        """
        idx = None
        for i, line in enumerate(self._lines):
            if "Mayer bond order" in line:
                idx = i
        if idx is None:
            raise ValueError(f"No Mayer bond orders found in {self.path}")

        # Match: B(  0-P ,  1-O ) :   1.1622
        bond_pattern = re.compile(
            r"B\(\s*(\d+)-(\w+)\s*,\s*(\d+)-(\w+)\s*\)\s*:\s*([-.\d]+)"
        )
        rows = []
        for line in self._lines[idx:]:
            for m in bond_pattern.finditer(line):
                bo = float(m.group(5))
                if bo > 0.05:  # skip negligible
                    rows.append({
                        "atom_i": int(m.group(1)) + 1,
                        "element_i": m.group(2),
                        "atom_j": int(m.group(3)) + 1,
                        "element_j": m.group(4),
                        "bond_order": bo,
                    })
        return rows

    def get_energy(self) -> float:
        """Extract final total energy (Hartree)."""
        pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+([-.\d]+)")
        matches = pattern.findall(self._text)
        if not matches:
            raise ValueError(f"No energy found in {self.path}")
        return float(matches[-1])


# ---------------------------------------------------------------------------
# NBO output parser (FILE_*.log from NBO6/NBO7)
# ---------------------------------------------------------------------------

class NboOutputParser:
    """Parse an NBO output file for NPA charges and NAO populations."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text = _read_text_file(self.path)
        self._lines = self._text.splitlines()

    def get_npa_charges(self) -> list[dict]:
        """Extract NPA (Natural Population Analysis) atomic charges.

        Looks for 'Summary of Natural Population Analysis' block.
        """
        header = "Summary of Natural Population Analysis:"
        idx = None
        for i, line in enumerate(self._lines):
            if header in line:
                idx = i
                break
        if idx is None:
            raise ValueError(f"No NPA charges found in {self.path}")

        # Format:
        #  Atom No    Charge    Core    Valence   Rydberg    Total
        # ------------------------------------------------------------
        #    P  1    1.56157   9.99883   3.32740   0.11219  13.43843
        # ============================================================
        rows = []
        dash_count = 0
        for line in self._lines[idx + 1:]:
            stripped = line.strip()
            if stripped.startswith("----"):
                dash_count += 1
                continue
            if stripped.startswith("====") or stripped.startswith("* Total"):
                break
            if dash_count < 1:
                continue  # haven't reached data yet
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                elem = parts[0]
                atom_idx = int(parts[1])
                charge = float(parts[2])
                rows.append({
                    "atom_index": atom_idx,
                    "element": elem,
                    "charge": charge,
                })
            except (ValueError, IndexError):
                continue
        return rows

    def get_job_title(self) -> str:
        """Extract the ORCA Job title from NBO output."""
        for line in self._lines:
            if "Job title:" in line:
                return line.split("Job title:")[1].strip()
        return ""


# ---------------------------------------------------------------------------
# Molden format parser
# ---------------------------------------------------------------------------

class MoldenParser:
    """Parse a Molden format file for geometry and MO data."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text = _read_text_file(self.path)
        self._lines = self._text.splitlines()

    def get_geometry(self) -> list[dict]:
        """Extract geometry from [Atoms] section."""
        in_atoms = False
        is_angs = True
        atoms = []
        for line in self._lines:
            stripped = line.strip()
            if stripped.lower().startswith("[atoms]"):
                in_atoms = True
                is_angs = "angs" in stripped.lower()
                continue
            if stripped.startswith("[") and in_atoms:
                break
            if not in_atoms or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            if not is_angs:
                x *= BOHR_TO_ANG
                y *= BOHR_TO_ANG
                z *= BOHR_TO_ANG
            atoms.append({
                "atom_index": int(parts[1]),
                "element": parts[0],
                "x": x, "y": y, "z": z,
            })
        if not atoms:
            raise ValueError(f"No [Atoms] section found in {self.path}")
        return atoms

    def get_mo_energies(self) -> list[float]:
        """Extract MO energies from [MO] section."""
        energies = []
        in_mo = False
        for line in self._lines:
            stripped = line.strip()
            if stripped.lower() == "[mo]":
                in_mo = True
                continue
            if stripped.startswith("[") and in_mo:
                break
            if not in_mo:
                continue
            if stripped.lower().startswith("ene="):
                energies.append(float(stripped.split("=")[1]))
        return energies

    def get_mo_occupations(self) -> list[float]:
        """Extract MO occupation numbers."""
        occs = []
        in_mo = False
        for line in self._lines:
            stripped = line.strip()
            if stripped.lower() == "[mo]":
                in_mo = True
                continue
            if stripped.startswith("[") and in_mo:
                break
            if not in_mo:
                continue
            if stripped.lower().startswith("occup="):
                occs.append(float(stripped.split("=")[1]))
        return occs


# ---------------------------------------------------------------------------
# Auto-detect and parse
# ---------------------------------------------------------------------------

def auto_parse(path: str | Path) -> dict:
    """Auto-detect file format and return parsed data.

    Returns a dict with keys depending on file type:
    - 'atoms': list[dict] (always present)
    - 'charges': list[dict] (if charge data found)
    - 'energy': float (if energy found)
    - 'bond_orders': list[dict] (if bond order data found)
    - 'format': str (detected format name)
    - 'parser': the parser object for further queries
    """
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()

    result: dict = {"format": "unknown", "atoms": []}

    if suffix == ".fchk" or suffix == ".fch":
        parser = FchkParser(path)
        result["format"] = "gaussian_fchk"
        result["atoms"] = parser.get_geometry()
        result["energy"] = parser.get_total_energy()
        result["parser"] = parser

    elif suffix in (".molden", ".molf"):
        parser = MoldenParser(path)
        result["format"] = "molden"
        result["atoms"] = parser.get_geometry()
        result["parser"] = parser

    elif suffix in (".log", ".out", ".g09", ".g16"):
        text = _read_text_file(path)

        # NBO standalone output (NBO 6.0 / 7.0)
        if "NBO" in text[:500] and "N A T U R A L" in text[:500]:
            parser = NboOutputParser(path)
            result["format"] = "nbo"
            try:
                npa = parser.get_npa_charges()
                if npa:
                    result["atoms"] = [
                        {"atom_index": r["atom_index"], "element": r["element"],
                         "x": 0.0, "y": 0.0, "z": 0.0}
                        for r in npa
                    ]
                    result.setdefault("charges", {})["npa"] = npa
            except ValueError:
                pass
            result["parser"] = parser
            return result

        # Try Gaussian first, fall back to ORCA
        if "Gaussian" in text[:2000] or "Entering Link 1" in text[:500]:
            parser = GaussianLogParser(path)
            result["format"] = "gaussian_log"
            result["atoms"] = parser.get_geometry()
            try:
                result["energy"] = parser.get_energy()
            except ValueError:
                pass
            # Try to get charges
            for method, getter in [
                ("mulliken", parser.get_mulliken_charges),
                ("apt", parser.get_apt_charges),
                ("nbo", parser.get_nbo_charges),
                ("chelpg", parser.get_chelpg_charges),
            ]:
                try:
                    charges = getter()
                    if charges:
                        result.setdefault("charges", {})[method] = charges
                except ValueError:
                    pass
            result["parser"] = parser
        else:
            # Assume ORCA
            parser = OrcaOutputParser(path)
            result["format"] = "orca_out"
            result["atoms"] = parser.get_geometry()
            try:
                result["energy"] = parser.get_energy()
            except ValueError:
                pass
            for method, getter in [
                ("mulliken", parser.get_mulliken_charges),
                ("loewdin", parser.get_loewdin_charges),
                ("chelpg", parser.get_chelpg_charges),
            ]:
                try:
                    charges = getter()
                    if charges:
                        result.setdefault("charges", {})[method] = charges
                except ValueError:
                    pass
            try:
                result["bond_orders"] = parser.get_mayer_bond_orders()
            except ValueError:
                pass
            result["parser"] = parser

    elif suffix in (".xyz",):
        from ._geometry import read_xyz
        result["format"] = "xyz"
        result["atoms"] = read_xyz(path)

    elif suffix == ".cube" or suffix == ".cub":
        from ._geometry import read_cube
        result["format"] = "cube"
        cube_data = read_cube(path)
        result["atoms"] = cube_data["atoms"]
        result["cube"] = cube_data

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Supported: .xyz, .cube, .log, .out, .fchk, .molden"
        )

    return result
