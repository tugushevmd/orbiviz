"""Tests for _io.py — CSV readers."""
import pytest
from pathlib import Path

from pyqchem.viz._io import read_charge_csv, read_fukui_csv

DATA_DIR = Path(__file__).parent / "data"


class TestReadChargeCsv:
    def test_mulliken(self):
        rows = read_charge_csv(DATA_DIR / "charges_water.csv", "mulliken_charge")
        assert len(rows) == 3
        assert rows[0]["element"] == "O"
        assert abs(rows[0]["charge"] - (-0.3342)) < 1e-6

    def test_chelpg(self):
        rows = read_charge_csv(DATA_DIR / "charges_water.csv", "chelpg_charge")
        assert len(rows) == 3
        assert abs(rows[0]["charge"] - (-0.7844)) < 1e-6

    def test_has_coords(self):
        rows = read_charge_csv(DATA_DIR / "charges_water.csv", "mulliken_charge")
        assert "x" in rows[0]
        assert "y" in rows[0]
        assert "z" in rows[0]

    def test_wrong_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            read_charge_csv(DATA_DIR / "charges_water.csv", "nbo_charge")


class TestReadFukuiCsv:
    def test_ethanol_fukui(self):
        rows = read_fukui_csv(DATA_DIR / "fukui_ethanol.csv")
        assert len(rows) == 9
        assert rows[0]["element"] == "C"
        assert "f_plus" in rows[0]
        assert "f_minus" in rows[0]
        assert "dual_descriptor" in rows[0]

    def test_values(self):
        rows = read_fukui_csv(DATA_DIR / "fukui_ethanol.csv")
        o_row = [r for r in rows if r["element"] == "O"][0]
        assert abs(o_row["f_plus"] - 0.2114) < 1e-6
        assert abs(o_row["dual_descriptor"] - (-0.2198)) < 1e-6
