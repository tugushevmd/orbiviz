"""Integration tests — verify renderers produce output files without crashing."""
import pytest
from pathlib import Path

from pyqchem.viz._geometry import read_xyz
from pyqchem.viz._io import read_charge_csv, read_fukui_csv
from pyqchem.viz.charge_map import render_ranked_charge_map
from pyqchem.viz.fukui_map import render_condensed_fukui

DATA_DIR = Path(__file__).parent / "data"


class TestChargeMapRendering:
    def test_produces_png(self, tmp_path):
        charges = read_charge_csv(DATA_DIR / "charges_water.csv", "mulliken_charge")
        # Use coords from CSV
        atoms = []
        for r in charges:
            atoms.append({
                "atom_index": r["atom_index"],
                "element": r["element"],
                "x": r["x"], "y": r["y"], "z": r["z"],
            })
        out = tmp_path / "charge_map.png"
        render_ranked_charge_map(atoms, charges, out, title="Test Water", top_count=2)
        assert out.exists()
        assert out.stat().st_size > 1000  # not empty

    def test_chelpg_charges(self, tmp_path):
        charges = read_charge_csv(DATA_DIR / "charges_water.csv", "chelpg_charge")
        atoms = []
        for r in charges:
            atoms.append({
                "atom_index": r["atom_index"],
                "element": r["element"],
                "x": r["x"], "y": r["y"], "z": r["z"],
            })
        out = tmp_path / "chelpg_map.png"
        render_ranked_charge_map(atoms, charges, out, title="CHELPG")
        assert out.exists()


class TestFukuiMapRendering:
    def test_f_plus(self, tmp_path):
        atoms = read_xyz(DATA_DIR / "ethanol.xyz")
        fukui = read_fukui_csv(DATA_DIR / "fukui_ethanol.csv")
        out = tmp_path / "fukui_fplus.png"
        render_condensed_fukui(atoms, fukui, "f_plus", out, label_count=4)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_dual_descriptor(self, tmp_path):
        atoms = read_xyz(DATA_DIR / "ethanol.xyz")
        fukui = read_fukui_csv(DATA_DIR / "fukui_ethanol.csv")
        out = tmp_path / "fukui_dual.png"
        render_condensed_fukui(atoms, fukui, "dual_descriptor", out)
        assert out.exists()

    def test_invalid_metric_raises(self, tmp_path):
        atoms = read_xyz(DATA_DIR / "ethanol.xyz")
        fukui = read_fukui_csv(DATA_DIR / "fukui_ethanol.csv")
        with pytest.raises(ValueError, match="Unknown metric"):
            render_condensed_fukui(atoms, fukui, "invalid", tmp_path / "x.png")
