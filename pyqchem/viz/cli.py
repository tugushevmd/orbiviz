"""Unified CLI entry point: python -m pyqchem.viz <subcommand>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_charge(args: argparse.Namespace) -> None:
    from ._io import read_charge_csv
    from ._geometry import read_xyz
    from .charge_map import render_ranked_charge_map

    charge_rows = read_charge_csv(Path(args.charges_csv), args.charge_column)
    if args.xyz:
        atoms = read_xyz(Path(args.xyz))
    else:
        atoms = []
        for row in charge_rows:
            if not {"x", "y", "z"} <= set(row):
                raise ValueError("CSV must contain x/y/z columns if --xyz is not provided.")
            atoms.append(
                {
                    "atom_index": row["atom_index"],
                    "element": row["element"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                }
            )
    render_ranked_charge_map(
        atoms, charge_rows, Path(args.output),
        title=args.title, subtitle=args.subtitle, top_count=args.top_count,
    )
    print(f"Saved: {args.output}")


def _cmd_fukui(args: argparse.Namespace) -> None:
    from ._io import read_fukui_csv
    from ._geometry import read_xyz
    from .fukui_map import render_condensed_fukui

    atoms = read_xyz(Path(args.xyz))
    fukui_rows = read_fukui_csv(Path(args.indices_csv))
    render_condensed_fukui(
        atoms, fukui_rows, args.metric, Path(args.output),
        title=args.title, label_count=args.label_count,
    )
    print(f"Saved: {args.output}")


def _cmd_esp(args: argparse.Namespace) -> None:
    from .esp_surface import render_esp_surface

    render_esp_surface(
        Path(args.density_cube), Path(args.esp_cube), Path(args.output),
        isovalue=args.isovalue, alpha=args.alpha,
        elev=args.elev, azim=args.azim,
        max_faces=args.max_faces, labels=args.labels, title=args.title,
    )
    print(f"Saved: {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyqchem-viz",
        description="Publication-quality visualizations for computational chemistry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- charge ---
    p_charge = sub.add_parser("charge", help="Render a ranked atomic charge map.")
    p_charge.add_argument("--charges-csv", required=True, help="CSV with charges.")
    p_charge.add_argument("--xyz", default="", help="Optional XYZ geometry file.")
    p_charge.add_argument("--charge-column", default="adch_charge", help="Charge column name.")
    p_charge.add_argument("--output", required=True, help="Output PNG path.")
    p_charge.add_argument("--title", default="Charge Map", help="Figure title.")
    p_charge.add_argument("--subtitle", default="Only the most charged atoms are numbered")
    p_charge.add_argument("--top-count", type=int, default=8)
    p_charge.set_defaults(func=_cmd_charge)

    # --- fukui ---
    p_fukui = sub.add_parser("fukui", help="Render a condensed Fukui index map.")
    p_fukui.add_argument("--xyz", required=True, help="XYZ geometry file.")
    p_fukui.add_argument("--indices-csv", required=True, help="CSV with Fukui indices.")
    p_fukui.add_argument("--metric", required=True, choices=["f_plus", "f_minus", "dual_descriptor"])
    p_fukui.add_argument("--output", required=True, help="Output PNG path.")
    p_fukui.add_argument("--title", default="")
    p_fukui.add_argument("--label-count", type=int, default=8)
    p_fukui.set_defaults(func=_cmd_fukui)

    # --- esp ---
    p_esp = sub.add_parser("esp", help="Render an ESP-colored density isosurface.")
    p_esp.add_argument("--density-cube", required=True, help="Electron density cube file.")
    p_esp.add_argument("--esp-cube", required=True, help="ESP cube file.")
    p_esp.add_argument("--output", required=True, help="Output PNG path.")
    p_esp.add_argument("--isovalue", type=float, default=0.001)
    p_esp.add_argument("--alpha", type=float, default=0.22)
    p_esp.add_argument("--elev", type=float, default=22.0)
    p_esp.add_argument("--azim", type=float, default=58.0)
    p_esp.add_argument("--max-faces", type=int, default=55000)
    p_esp.add_argument("--labels", choices=["none", "elements", "indices"], default="elements")
    p_esp.add_argument("--title", default="")
    p_esp.set_defaults(func=_cmd_esp)

    # --- mo ---
    p_mo = sub.add_parser("mo", help="Render a molecular orbital isosurface.")
    p_mo.add_argument("--cube", required=True, help="MO cube file.")
    p_mo.add_argument("--output", required=True, help="Output PNG path.")
    p_mo.add_argument("--isovalue", type=float, default=0.02)
    p_mo.add_argument("--elev", type=float, default=20.0)
    p_mo.add_argument("--azim", type=float, default=55.0)
    p_mo.add_argument("--max-faces", type=int, default=60000)
    p_mo.add_argument("--labels", choices=["none", "elements", "indices"], default="elements")
    p_mo.add_argument("--title", default="")
    p_mo.set_defaults(func=_cmd_mo)

    # --- nci ---
    p_nci = sub.add_parser("nci", help="NCI analysis (scatter or 3D isosurface).")
    p_nci.add_argument("--density-cube", required=True, help="Electron density cube file.")
    p_nci.add_argument("--output", required=True, help="Output PNG path.")
    p_nci.add_argument("--mode", choices=["scatter", "3d"], default="scatter", help="scatter or 3d.")
    p_nci.add_argument("--rho-cutoff", type=float, default=0.05)
    p_nci.add_argument("--rdg-isovalue", type=float, default=0.5, help="RDG isovalue for 3D mode.")
    p_nci.add_argument("--elev", type=float, default=20.0)
    p_nci.add_argument("--azim", type=float, default=55.0)
    p_nci.add_argument("--labels", choices=["none", "elements", "indices"], default="elements")
    p_nci.add_argument("--title", default="")
    p_nci.set_defaults(func=_cmd_nci)

    # --- bond-order ---
    p_bo = sub.add_parser("bond-order", help="Render a bond order map from ORCA output.")
    p_bo.add_argument("--orca-out", required=True, help="ORCA output file with Mayer bond orders.")
    p_bo.add_argument("--output", required=True, help="Output PNG path.")
    p_bo.add_argument("--title", default="Bond Order Map")
    p_bo.add_argument("--min-bo", type=float, default=0.3, help="Minimum bond order to display.")
    p_bo.add_argument("--top-count", type=int, default=12)
    p_bo.set_defaults(func=_cmd_bond_order)

    # --- auto ---
    p_auto = sub.add_parser("auto", help="Auto-detect file format and show available data.")
    p_auto.add_argument("file", help="Input file (any supported format).")
    p_auto.set_defaults(func=_cmd_auto)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_mo(args: argparse.Namespace) -> None:
    from .mo_surface import render_mo_surface

    render_mo_surface(
        Path(args.cube), Path(args.output),
        isovalue=args.isovalue, elev=args.elev, azim=args.azim,
        max_faces=args.max_faces, labels=args.labels, title=args.title,
    )
    print(f"Saved: {args.output}")


def _cmd_nci(args: argparse.Namespace) -> None:
    from .nci_plot import render_nci_scatter, render_nci_3d

    if args.mode == "scatter":
        render_nci_scatter(
            Path(args.density_cube), Path(args.output),
            rho_cutoff=args.rho_cutoff, title=args.title or "NCI Scatter Plot",
        )
    else:
        render_nci_3d(
            Path(args.density_cube), Path(args.output),
            rho_cutoff=args.rho_cutoff, rdg_isovalue=args.rdg_isovalue,
            elev=args.elev, azim=args.azim, labels=args.labels,
            title=args.title or "NCI Isosurface",
        )
    print(f"Saved: {args.output}")


def _cmd_bond_order(args: argparse.Namespace) -> None:
    from ._parsers import OrcaOutputParser
    from .bond_order_map import render_bond_order_map

    parser = OrcaOutputParser(Path(args.orca_out))
    atoms = parser.get_geometry()
    bond_orders = parser.get_mayer_bond_orders()
    render_bond_order_map(
        atoms, bond_orders, Path(args.output),
        title=args.title, min_bo=args.min_bo, top_count=args.top_count,
    )
    print(f"Saved: {args.output}")


def _cmd_auto(args: argparse.Namespace) -> None:
    from ._parsers import auto_parse

    result = auto_parse(args.file)
    print(f"Format:  {result['format']}")
    print(f"Atoms:   {len(result['atoms'])}")
    if result["atoms"]:
        elements = set(a["element"] for a in result["atoms"])
        formula_parts = []
        for elem in sorted(elements):
            count = sum(1 for a in result["atoms"] if a["element"] == elem)
            formula_parts.append(f"{elem}{count}" if count > 1 else elem)
        print(f"Formula: {''.join(formula_parts)}")
    if "energy" in result:
        print(f"Energy:  {result['energy']:.10f} Hartree")
    if "charges" in result:
        print(f"Charges: {', '.join(result['charges'].keys())}")
    if "bond_orders" in result:
        print(f"Bond orders: {len(result['bond_orders'])} significant bonds")
    if "cube" in result:
        shape = result["cube"]["shape"]
        print(f"Grid:    {shape[0]}×{shape[1]}×{shape[2]}")


if __name__ == "__main__":
    main()
