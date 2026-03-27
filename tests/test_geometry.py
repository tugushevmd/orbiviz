"""Tests for _geometry.py — XYZ reading, bond detection, 2D projection."""
import pytest
from pathlib import Path

from pyqchem.viz._geometry import read_xyz, build_bonds, project_2d

DATA_DIR = Path(__file__).parent / "data"


class TestReadXyz:
    def test_water(self):
        atoms = read_xyz(DATA_DIR / "water.xyz")
        assert len(atoms) == 3
        assert atoms[0]["element"] == "O"
        assert atoms[1]["element"] == "H"
        assert atoms[0]["atom_index"] == 1

    def test_ethanol(self):
        atoms = read_xyz(DATA_DIR / "ethanol.xyz")
        assert len(atoms) == 9
        elements = [a["element"] for a in atoms]
        assert elements.count("C") == 2
        assert elements.count("O") == 1
        assert elements.count("H") == 6

    def test_coords_are_float(self):
        atoms = read_xyz(DATA_DIR / "water.xyz")
        for a in atoms:
            assert isinstance(a["x"], float)
            assert isinstance(a["y"], float)
            assert isinstance(a["z"], float)

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            read_xyz(DATA_DIR / "nonexistent.xyz")


class TestBuildBonds:
    def test_water_bonds(self):
        atoms = read_xyz(DATA_DIR / "water.xyz")
        bonds = build_bonds(atoms)
        # Water: O-H, O-H (no H-H bond)
        assert len(bonds) == 2
        # Both bonds should involve atom 1 (O)
        for i, j in bonds:
            assert 1 in (i, j)

    def test_ethanol_bonds(self):
        atoms = read_xyz(DATA_DIR / "ethanol.xyz")
        bonds = build_bonds(atoms)
        # Ethanol: C-C, C-O, O-H, 3×C-H, 2×C-H = 8 bonds
        assert len(bonds) == 8

    def test_empty_atoms(self):
        bonds = build_bonds([])
        assert bonds == []


class TestProject2d:
    def test_projection_shape(self):
        atoms = read_xyz(DATA_DIR / "water.xyz")
        proj = project_2d(atoms)
        assert proj.shape == (3, 2)

    def test_atoms_get_px_py(self):
        atoms = read_xyz(DATA_DIR / "water.xyz")
        project_2d(atoms)
        for a in atoms:
            assert "px" in a
            assert "py" in a
