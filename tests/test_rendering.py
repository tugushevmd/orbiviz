from argparse import Namespace
from pathlib import Path
import warnings

from pyqchem.viz.cli import _cmd_bond_order, _cmd_charge, _cmd_fukui


DATA_DIR = Path(__file__).parent / "data"
ORCA_FIXTURE_DIR = DATA_DIR / "c10h9no2_orca"


class TestDirectOutputRendering:
    def test_charge_from_orca_out_chelpg(self, tmp_path):
        out = tmp_path / "charge_from_out_chelpg.png"
        _cmd_charge(
            Namespace(
                charges_csv="",
                input_file=str(ORCA_FIXTURE_DIR / "10_sp_props.out"),
                xyz="",
                charge_column="adch_charge",
                charge_scheme="chelpg",
                output=str(out),
                title="ORCA CHELPG",
                subtitle="Directly parsed from output",
                top_count=8,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_charge_from_orca_out_loewdin(self, tmp_path):
        out = tmp_path / "charge_from_out_loewdin.png"
        _cmd_charge(
            Namespace(
                charges_csv="",
                input_file=str(ORCA_FIXTURE_DIR / "10_sp_props.out"),
                xyz="",
                charge_column="adch_charge",
                charge_scheme="loewdin",
                output=str(out),
                title="ORCA Loewdin",
                subtitle="Directly parsed from output",
                top_count=8,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_bond_order_from_orca_out(self, tmp_path):
        out = tmp_path / "bond_order_from_out.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _cmd_bond_order(
                Namespace(
                    orca_out=str(ORCA_FIXTURE_DIR / "10_sp_props.out"),
                    output=str(out),
                    title="Bond Order Map",
                    min_bo=0.3,
                    top_count=12,
                )
            )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_fukui_from_output_triplet(self, tmp_path):
        out = tmp_path / "fukui_from_outputs.png"
        _cmd_fukui(
            Namespace(
                xyz="",
                indices_csv="",
                neutral_file=str(ORCA_FIXTURE_DIR / "20_fukui_neutral.out"),
                anion_file=str(ORCA_FIXTURE_DIR / "21_fukui_anion.out"),
                cation_file=str(ORCA_FIXTURE_DIR / "22_fukui_cation.out"),
                charge_scheme="mulliken",
                metric="dual_descriptor",
                output=str(out),
                title="",
                label_count=8,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000
