from pathlib import Path

from pyqchem.viz._parsers import auto_parse


DATA_DIR = Path(__file__).parent / "data"
ORCA_FIXTURE_DIR = DATA_DIR / "c10h9no2_orca"


class TestOrcaFixtureParsing:
    def test_orca_out_detects_all_supported_charge_schemes(self):
        result = auto_parse(ORCA_FIXTURE_DIR / "10_sp_props.out")
        assert result["format"] == "orca_out"
        assert len(result["atoms"]) == 22
        assert "mulliken" in result["charges"]
        assert "loewdin" in result["charges"]
        assert "chelpg" in result["charges"]
        assert len(result["charges"]["chelpg"]) == 22
        assert len(result["bond_orders"]) > 0

    def test_molden_from_orca_fixture(self):
        result = auto_parse(ORCA_FIXTURE_DIR / "c10h9no2.molden")
        assert result["format"] == "molden"
        assert len(result["atoms"]) == 22

    def test_cube_from_orca_fixture(self):
        result = auto_parse(ORCA_FIXTURE_DIR / "c10h9no2_density.cube")
        assert result["format"] == "cube"
        assert len(result["atoms"]) == 22
        assert result["cube"]["shape"] == (80, 80, 80)
