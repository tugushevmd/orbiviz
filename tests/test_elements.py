"""Tests for _elements.py — periodic table data."""
import pytest
from pyqchem.viz._elements import ELEMENTS, Z_TO_SYMBOL, get_element


class TestElements:
    def test_element_count(self):
        assert len(ELEMENTS) >= 86

    def test_z_to_symbol_consistency(self):
        for sym, elem in ELEMENTS.items():
            assert Z_TO_SYMBOL[elem.Z] == sym

    def test_common_elements(self):
        h = get_element("H")
        assert h.Z == 1
        assert h.covalent_radius == 0.31
        assert h.cpk_color == "#FFFFFF"

        c = get_element("C")
        assert c.Z == 6
        assert c.covalent_radius == 0.76

        fe = get_element("Fe")
        assert fe.Z == 26
        assert fe.name == "Iron"

    def test_get_element_by_z(self):
        elem = get_element(8)
        assert elem.symbol == "O"
        assert elem.Z == 8

    def test_get_element_case_insensitive(self):
        assert get_element("cl").symbol == "Cl"
        assert get_element("CL").symbol == "Cl"
        assert get_element("Cl").symbol == "Cl"

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="Unknown element"):
            get_element("Xx")

    def test_unknown_z_raises(self):
        with pytest.raises(ValueError, match="Unknown atomic number"):
            get_element(999)

    def test_radii_positive(self):
        for sym, elem in ELEMENTS.items():
            assert elem.covalent_radius > 0, f"{sym} has non-positive covalent radius"
            assert elem.vdw_radius > 0, f"{sym} has non-positive vdW radius"

    def test_transition_metals_present(self):
        for sym in ("Ti", "Cu", "Zn", "Pd", "Pt", "Au"):
            assert sym in ELEMENTS
