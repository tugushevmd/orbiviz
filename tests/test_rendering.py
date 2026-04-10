from argparse import Namespace
from pathlib import Path
import warnings

from pyqchem.viz.cli import (
    _cmd_bond_order,
    _cmd_charge,
    _cmd_diff,
    _cmd_fukui,
    _cmd_mo_slice,
    _cmd_spin,
    _cmd_traj,
)


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
                dpi=200,
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
                dpi=200,
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
                    dpi=200,
                )
            )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_spin_density_from_orca_anion(self, tmp_path):
        out = tmp_path / "spin_anion.png"
        _cmd_spin(
            Namespace(
                input_file=str(ORCA_FIXTURE_DIR / "21_fukui_anion.out"),
                xyz="",
                scheme="mulliken",
                output=str(out),
                title="",
                top_count=8,
                dpi=200,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_charge_difference_neutral_anion(self, tmp_path):
        out = tmp_path / "delta_q.png"
        _cmd_diff(
            Namespace(
                file_a=str(ORCA_FIXTURE_DIR / "20_fukui_neutral.out"),
                file_b=str(ORCA_FIXTURE_DIR / "21_fukui_anion.out"),
                charge_scheme="mulliken",
                xyz="",
                output=str(out),
                title="",
                top_count=8,
                dpi=200,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_mo_slice_2d_from_homo_cube(self, tmp_path):
        out = tmp_path / "homo_slice.png"
        _cmd_mo_slice(
            Namespace(
                cube=str(ORCA_FIXTURE_DIR / "c10h9no2_homo.cube"),
                output=str(out),
                plane="xy",
                offset=0.0,
                levels=18,
                title="",
                dpi=200,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_trajectory_animation_from_opt(self, tmp_path):
        out = tmp_path / "opt.gif"
        _cmd_traj(
            Namespace(
                trajectory=str(ORCA_FIXTURE_DIR / "00_opt_r2scan3c_trj.xyz"),
                output=str(out),
                fps=6,
                stride=2,
                title="",
                dpi=120,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_svg_output_format(self, tmp_path):
        out = tmp_path / "charge.svg"
        _cmd_charge(
            Namespace(
                charges_csv="",
                input_file=str(ORCA_FIXTURE_DIR / "10_sp_props.out"),
                xyz="",
                charge_column="adch_charge",
                charge_scheme="mulliken",
                output=str(out),
                title="SVG export",
                subtitle="",
                top_count=8,
                dpi=200,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000
        assert out.read_text(encoding="utf-8", errors="ignore").lstrip().startswith("<?xml")

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
                dpi=200,
            )
        )
        assert out.exists()
        assert out.stat().st_size > 1000
