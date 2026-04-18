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
        """Extract Mulliken atomic charges (closed- or open-shell)."""
        try:
            return self._parse_charge_block("Mulliken charges:", skip_header=1)
        except ValueError:
            # Open-shell Gaussian uses "Mulliken charges and spin densities:"
            return self._parse_charge_block(
                "Mulliken charges and spin densities:", skip_header=1
            )

    def get_mulliken_spin_populations(self) -> list[dict]:
        """Extract Mulliken spin populations from an open-shell Gaussian log."""
        idx = None
        for i, line in enumerate(self._lines):
            if "Mulliken charges and spin densities:" in line:
                idx = i
        if idx is None:
            raise ValueError(f"No Mulliken spin densities in {self.path}")
        rows: list[dict] = []
        for line in self._lines[idx + 2:]:
            parts = line.split()
            if len(parts) < 4:
                break
            try:
                atom_idx = int(parts[0])
                spin = float(parts[3])
            except (ValueError, IndexError):
                break
            rows.append({
                "atom_index": atom_idx,
                "element": parts[1],
                "spin": spin,
            })
        if not rows:
            raise ValueError(f"Empty Mulliken spin block in {self.path}")
        return rows

    def get_apt_charges(self) -> list[dict]:
        """Extract APT atomic charges."""
        return self._parse_charge_block("APT charges:", skip_header=1)

    def get_nbo_charges(self) -> list[dict]:
        """Extract NBO natural charges from Natural Population Analysis."""
        # Different Gaussian / NBO versions interleave a variable number of
        # header lines between "Summary of Natural Population Analysis:" and
        # the dashed separator that precedes the data. Locate the header line
        # ("Atom  No    Charge") instead of relying on a strict regex shape.
        idx = None
        for i, line in enumerate(self._lines):
            if "Summary of Natural Population Analysis" in line:
                idx = i
                break
        if idx is None:
            raise ValueError(f"No NBO charges found in {self.path}")
        # Find the column header within the next ~10 lines
        header_idx = None
        for j in range(idx + 1, min(idx + 12, len(self._lines))):
            line = self._lines[j]
            if "Atom" in line and "No" in line and "Charge" in line:
                header_idx = j
                break
        if header_idx is None:
            raise ValueError(f"No NBO charge header found in {self.path}")
        # Skip the dashed separator line right after the header
        start = header_idx + 1
        if start < len(self._lines) and set(self._lines[start].strip()) <= {"-", " "}:
            start += 1
        rows: list[dict] = []
        for line in self._lines[start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("---"):
                if rows:
                    break
                continue
            parts = stripped.split()
            if len(parts) < 3:
                if rows:
                    break
                continue
            try:
                rows.append({
                    "atom_index": int(parts[1]),
                    "element": parts[0],
                    "charge": float(parts[2]),
                })
            except (ValueError, IndexError):
                if rows:
                    break
                continue
        if not rows:
            raise ValueError(f"No NBO rows parsed from {self.path}")
        return rows

    def get_chelpg_charges(self) -> list[dict]:
        """Extract CHELPG or ESP-fit charges."""
        # Gaussian prints ESP charges after 'Charges from ESP fit'
        return self._parse_charge_block(
            "Charges from ESP fit", skip_header=2,
        )

    def get_orbital_energies(self) -> tuple[np.ndarray, int]:
        """Extract alpha orbital energies from 'eigenvalues' print lines.

        Returns
        -------
        (energies_hartree, n_occ) — full orbital energy array in Hartree
        and the number of occupied orbitals.
        """
        occ_vals: list[float] = []
        virt_vals: list[float] = []
        occ_re  = re.compile(r"Alpha\s+occ\.\s+eigenvalues\s+--\s+(.*)")
        virt_re = re.compile(r"Alpha\s+virt\.\s+eigenvalues\s+--\s+(.*)")
        for line in self._lines:
            m = occ_re.search(line)
            if m:
                occ_vals.extend(float(v) for v in m.group(1).split())
                continue
            m = virt_re.search(line)
            if m:
                virt_vals.extend(float(v) for v in m.group(1).split())
        if not occ_vals:
            raise ValueError(f"No orbital eigenvalues found in {self.path}")
        energies = np.array(occ_vals + virt_vals, dtype=float)
        return energies, len(occ_vals)

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

    def get_n_alpha_electrons(self) -> int:
        """Return number of alpha electrons (= n_occ for closed-shell RHF)."""
        try:
            return int(self._read_scalar("Number of alpha electrons"))
        except KeyError:
            return self.get_n_electrons() // 2

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
        """Generic parser for ORCA charge blocks (Mulliken, Löwdin).

        Handles both ``CHARGES`` and ``CHARGES AND SPIN POPULATIONS`` blocks.
        For the latter, both ``charge`` and ``spin`` keys are populated.
        """
        idx = None
        for i, line in enumerate(self._lines):
            if header in line:
                idx = i
        if idx is None:
            raise ValueError(f"No '{header}' found in {self.path}")
        rows = []
        charge_re = re.compile(
            r"\s*(\d+)\s+(\w+)\s*:\s*([-.\d]+)(?:\s+([-.\d]+))?"
        )
        for line in self._lines[idx + 1:]:
            m = charge_re.match(line)
            if m:
                row = {
                    "atom_index": int(m.group(1)) + 1,  # ORCA is 0-based
                    "element": m.group(2),
                    "charge": float(m.group(3)),
                }
                if m.group(4) is not None:
                    row["spin"] = float(m.group(4))
                rows.append(row)
            elif rows:
                # Already found data, non-matching line = end of block
                break
            # Skip separator lines (---) before data starts
        return rows

    def get_mulliken_spin_populations(self) -> list[dict]:
        """Return Mulliken spin populations (open-shell only)."""
        rows = self._parse_orca_charge_block("MULLIKEN ATOMIC CHARGES AND SPIN")
        spin_rows = [
            {"atom_index": r["atom_index"], "element": r["element"],
             "spin": r.get("spin", 0.0)}
            for r in rows
        ]
        if not any("spin" in r for r in rows):
            raise ValueError("No Mulliken spin populations found.")
        return spin_rows

    def get_loewdin_spin_populations(self) -> list[dict]:
        """Return Loewdin spin populations (open-shell only)."""
        rows = self._parse_orca_charge_block("LOEWDIN ATOMIC CHARGES AND SPIN")
        spin_rows = [
            {"atom_index": r["atom_index"], "element": r["element"],
             "spin": r.get("spin", 0.0)}
            for r in rows
        ]
        if not any("spin" in r for r in rows):
            raise ValueError("No Loewdin spin populations found.")
        return spin_rows

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

    def get_orbital_energies(self) -> tuple[np.ndarray, int]:
        """Extract orbital energies from the ORBITAL ENERGIES block.

        Returns
        -------
        (energies_hartree, n_occ) — full array in Hartree and count of
        occupied orbitals (OCC > 0.5).
        """
        # Take the last ORBITAL ENERGIES block (final SCF iteration)
        block_start = None
        for i, line in enumerate(self._lines):
            if "ORBITAL ENERGIES" in line:
                block_start = i
        if block_start is None:
            raise ValueError(f"No ORBITAL ENERGIES section found in {self.path}")

        # Locate the data header "E(Eh)"
        data_start = None
        for j in range(block_start, min(block_start + 12, len(self._lines))):
            if "E(Eh)" in self._lines[j]:
                data_start = j + 1
                break
        if data_start is None:
            raise ValueError(f"Could not locate orbital energy table in {self.path}")

        energies: list[float] = []
        occs: list[float] = []
        for line in self._lines[data_start:]:
            parts = line.split()
            if len(parts) < 3:
                if energies:
                    break
                continue
            try:
                int(parts[0])           # orbital index (0-based)
                occs.append(float(parts[1]))
                energies.append(float(parts[2]))
            except (ValueError, IndexError):
                if energies:
                    break
        if not energies:
            raise ValueError(f"Empty orbital energy table in {self.path}")
        n_occ = sum(1 for o in occs if o > 0.5)
        return np.array(energies, dtype=float), n_occ

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
# NWChem output parser
# ---------------------------------------------------------------------------

class NwchemOutputParser:
    """Parse an NWChem output file for geometry, charges, and energy.

    Recognised blocks:
    - ``Geometry "geometry" -> "geometry"`` (XYZ table in Å)
    - ``Total Mulliken charge``  / ``Mulliken analysis of the total density``
    - ``Total ESP charge``       / ``ESP``
    - ``Total Lowdin charge``
    - ``Total DFT energy`` / ``Total SCF energy``
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text = _read_text_file(self.path)
        self._lines = self._text.splitlines()

    def get_geometry(self, which: str = "last") -> list[dict]:
        # NWChem prints "Output coordinates in angstroms" then a header line
        # then a separator, then "  No.   Tag  Charge      X      Y      Z"
        markers = [
            i for i, line in enumerate(self._lines)
            if "Output coordinates in angstroms" in line
            or "Geometry " in line and "->" in line
        ]
        if not markers:
            raise ValueError(f"No geometry found in {self.path}")
        idx = markers[-1] if which == "last" else markers[0]
        # Skip ahead until the column header line
        start = None
        for j in range(idx, min(idx + 20, len(self._lines))):
            if "Tag" in self._lines[j] and ("Charge" in self._lines[j]
                                            or "X" in self._lines[j]):
                start = j + 2  # skip the header and the dashed separator
                break
        if start is None:
            raise ValueError(f"Could not locate geometry table in {self.path}")
        atoms: list[dict] = []
        idx_counter = 0
        for line in self._lines[start:]:
            parts = line.split()
            if len(parts) < 6:
                if atoms:
                    break
                continue
            try:
                # Columns: No  Tag  Charge  X  Y  Z
                int(parts[0])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except (ValueError, IndexError):
                if atoms:
                    break
                continue
            idx_counter += 1
            atoms.append({
                "atom_index": idx_counter,
                "element": parts[1],
                "x": x, "y": y, "z": z,
            })
        if not atoms:
            raise ValueError(f"Empty geometry block in {self.path}")
        return atoms

    def _parse_population_block(self, header: str) -> list[dict]:
        idx = None
        for i, line in enumerate(self._lines):
            if header in line:
                idx = i
        if idx is None:
            raise ValueError(f"No '{header}' found in {self.path}")
        rows: list[dict] = []
        # NWChem prints something like:
        #     1 C    6     6.080  ->   -0.080
        #     2 H    1     0.940  ->    0.060
        # The columns vary; rely on the trailing float being the charge.
        row_re = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\b.*?([-+]?\d+\.\d+)\s*$")
        for line in self._lines[idx + 1: idx + 4000]:
            m = row_re.match(line)
            if m:
                rows.append({
                    "atom_index": int(m.group(1)),
                    "element": m.group(2),
                    "charge": float(m.group(3)),
                })
            elif rows:
                # End of contiguous block
                if not line.strip():
                    continue
                if rows and not row_re.match(line):
                    # Allow a couple of blank lines, but a non-matching content
                    # line ends the block.
                    if any(c.isalpha() for c in line):
                        break
        return rows

    def get_mulliken_charges(self) -> list[dict]:
        return self._parse_population_block("Mulliken analysis of the total density")

    def get_esp_charges(self) -> list[dict]:
        return self._parse_population_block("ESP")

    def get_lowdin_charges(self) -> list[dict]:
        return self._parse_population_block("Lowdin Population Analysis")

    def get_energy(self) -> float:
        for key in ("Total DFT energy", "Total SCF energy", "Total CCSD(T) energy"):
            pattern = re.compile(re.escape(key) + r"\s*=\s*([-.\d]+)")
            matches = pattern.findall(self._text)
            if matches:
                return float(matches[-1])
        raise ValueError(f"No total energy found in {self.path}")


# ---------------------------------------------------------------------------
# Q-Chem output parser
# ---------------------------------------------------------------------------

class QchemOutputParser:
    """Parse a Q-Chem output file for geometry, charges, and energy.

    Recognised blocks:
    - ``Standard Nuclear Orientation (Angstroms)``
    - ``Ground-State Mulliken Net Atomic Charges``
    - ``Ground-State ChElPG Net Atomic Charges``
    - ``Hirshfeld Atomic Charges``
    - ``Total energy in the final basis set =``
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._text = _read_text_file(self.path)
        self._lines = self._text.splitlines()

    def get_geometry(self, which: str = "last") -> list[dict]:
        markers = [
            i for i, line in enumerate(self._lines)
            if "Standard Nuclear Orientation" in line
        ]
        if not markers:
            raise ValueError(f"No geometry found in {self.path}")
        idx = markers[-1] if which == "last" else markers[0]
        # Header is two lines below, then dashed separator, then atoms
        start = idx + 3
        atoms: list[dict] = []
        for line in self._lines[start:]:
            parts = line.split()
            if len(parts) < 5:
                break
            if parts[0].startswith("-") or "Nuclear" in line:
                break
            try:
                a_idx = int(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except (ValueError, IndexError):
                break
            atoms.append({
                "atom_index": a_idx,
                "element": parts[1],
                "x": x, "y": y, "z": z,
            })
        if not atoms:
            raise ValueError(f"Empty geometry in {self.path}")
        return atoms

    def _parse_qchem_charge_block(self, header: str) -> list[dict]:
        idx = None
        for i, line in enumerate(self._lines):
            if header in line:
                idx = i
        if idx is None:
            raise ValueError(f"No '{header}' found in {self.path}")
        # Q-Chem format:
        #          Atom    Charge (a.u.)
        #     ----------------------------
        #         1 C     -0.123456
        rows: list[dict] = []
        started = False
        for line in self._lines[idx + 1: idx + 4000]:
            if line.strip().startswith("---"):
                started = True
                continue
            if not started:
                continue
            parts = line.split()
            if len(parts) < 3:
                if rows:
                    break
                continue
            try:
                a_idx = int(parts[0])
                charge = float(parts[2])
            except (ValueError, IndexError):
                if rows:
                    break
                continue
            rows.append({
                "atom_index": a_idx,
                "element": parts[1],
                "charge": charge,
            })
        if not rows:
            raise ValueError(f"Empty charge block '{header}' in {self.path}")
        return rows

    def get_mulliken_charges(self) -> list[dict]:
        return self._parse_qchem_charge_block("Ground-State Mulliken Net Atomic Charges")

    def get_chelpg_charges(self) -> list[dict]:
        return self._parse_qchem_charge_block("Ground-State ChElPG Net Atomic Charges")

    def get_hirshfeld_charges(self) -> list[dict]:
        return self._parse_qchem_charge_block("Hirshfeld Atomic Charges")

    def get_energy(self) -> float:
        pattern = re.compile(r"Total energy in the final basis set\s*=\s*([-.\d]+)")
        matches = pattern.findall(self._text)
        if matches:
            return float(matches[-1])
        # Fallback: SCF energy
        pattern = re.compile(r"SCF\s+energy\s*=\s*([-.\d]+)")
        matches = pattern.findall(self._text)
        if matches:
            return float(matches[-1])
        raise ValueError(f"No total energy found in {self.path}")


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
        try:
            energies = parser.get_alpha_orbital_energies()
            n_occ    = parser.get_n_alpha_electrons()
            result["orbital_energies_hartree"] = energies
            result["n_occ"] = n_occ
        except (KeyError, ValueError):
            pass
        result["parser"] = parser

    elif suffix in (".molden", ".molf"):
        parser = MoldenParser(path)
        result["format"] = "molden"
        result["atoms"] = parser.get_geometry()
        try:
            mo_e = parser.get_mo_energies()
            mo_o = parser.get_mo_occupations()
            if mo_e and mo_o:
                result["orbital_energies_hartree"] = np.array(mo_e, dtype=float)
                result["n_occ"] = sum(1 for o in mo_o if o > 0.5)
        except (ValueError, IndexError):
            pass
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
                    # NBO standalone files have no geometry; expose NPA under
                    # *both* the canonical 'npa' key and the legacy 'nbo' alias
                    # so CLI users can switch between Gaussian/ORCA + NBO files
                    # without changing --charge-scheme. Atoms remain empty:
                    # callers must provide geometry via --xyz.
                    charges = result.setdefault("charges", {})
                    charges["npa"] = npa
                    charges["nbo"] = npa
                    result["atoms"] = []
                    result["needs_geometry"] = True
            except ValueError:
                pass
            result["parser"] = parser
            return result

        head = text[:4000]

        # Q-Chem detection
        if "Q-Chem" in head or "Welcome to Q-Chem" in head:
            parser = QchemOutputParser(path)
            result["format"] = "qchem"
            result["atoms"] = parser.get_geometry()
            try:
                result["energy"] = parser.get_energy()
            except ValueError:
                pass
            charges = result.setdefault("charges", {})
            for method, getter in [
                ("mulliken", parser.get_mulliken_charges),
                ("chelpg", parser.get_chelpg_charges),
                ("hirshfeld", parser.get_hirshfeld_charges),
            ]:
                try:
                    rows = getter()
                    if rows:
                        charges[method] = rows
                except ValueError:
                    pass
            result["parser"] = parser
            return result

        # NWChem detection
        if "Northwest Computational Chemistry Package" in head or "NWChem" in head:
            parser = NwchemOutputParser(path)
            result["format"] = "nwchem"
            result["atoms"] = parser.get_geometry()
            try:
                result["energy"] = parser.get_energy()
            except ValueError:
                pass
            charges = result.setdefault("charges", {})
            for method, getter in [
                ("mulliken", parser.get_mulliken_charges),
                ("chelpg", parser.get_esp_charges),
                ("loewdin", parser.get_lowdin_charges),
            ]:
                try:
                    rows = getter()
                    if rows:
                        charges[method] = rows
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
            charges = result.setdefault("charges", {})
            for method, getter in [
                ("mulliken", parser.get_mulliken_charges),
                ("apt", parser.get_apt_charges),
                ("nbo", parser.get_nbo_charges),
                ("chelpg", parser.get_chelpg_charges),
            ]:
                try:
                    rows = getter()
                    if rows:
                        charges[method] = rows
                except ValueError:
                    pass
            # NBO and NPA are the same numbers in Gaussian's NPA block —
            # expose both keys so users can pick either name on the CLI.
            if "nbo" in charges and "npa" not in charges:
                charges["npa"] = charges["nbo"]
            try:
                spin = parser.get_mulliken_spin_populations()
                if spin:
                    result.setdefault("spin_populations", {})["mulliken"] = spin
            except ValueError:
                pass
            try:
                energies, n_occ = parser.get_orbital_energies()
                result["orbital_energies_hartree"] = energies
                result["n_occ"] = n_occ
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
            for method, getter in [
                ("mulliken", parser.get_mulliken_spin_populations),
                ("loewdin", parser.get_loewdin_spin_populations),
            ]:
                try:
                    spin = getter()
                    if spin:
                        result.setdefault("spin_populations", {})[method] = spin
                except ValueError:
                    pass
            try:
                result["bond_orders"] = parser.get_mayer_bond_orders()
            except ValueError:
                pass
            try:
                energies, n_occ = parser.get_orbital_energies()
                result["orbital_energies_hartree"] = energies
                result["n_occ"] = n_occ
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
