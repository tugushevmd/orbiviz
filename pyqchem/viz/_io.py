"""Data file readers: CSV charge tables, Fukui indices, etc."""
from __future__ import annotations

import csv
from pathlib import Path


def read_charge_csv(path: Path, charge_column: str) -> list[dict]:
    """Read a CSV with atomic charges.

    Required columns: atom_index, element, <charge_column>.
    Optional columns: x, y, z (coordinates in Å).
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {path}")
        if charge_column not in reader.fieldnames:
            raise ValueError(
                f"Column {charge_column!r} not found in {path}. "
                f"Available: {', '.join(reader.fieldnames)}"
            )
        for required in ("atom_index", "element"):
            if required not in reader.fieldnames:
                raise ValueError(f"Required column {required!r} missing in {path}")

        for row in reader:
            item = {
                "atom_index": int(row["atom_index"]),
                "element": row["element"].strip(),
                "charge": float(row[charge_column]),
            }
            if {"x", "y", "z"} <= set(row):
                item["x"] = float(row["x"])
                item["y"] = float(row["y"])
                item["z"] = float(row["z"])
            rows.append(item)
    return rows


def read_fukui_csv(path: Path) -> list[dict]:
    """Read a CSV with condensed Fukui indices.

    Required columns: atom_index, element, f_plus, f_minus, dual_descriptor.
    """
    required = {"atom_index", "element", "f_plus", "f_minus", "dual_descriptor"}
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {', '.join(sorted(missing))}. "
                f"Available: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            rows.append(
                {
                    "atom_index": int(row["atom_index"]),
                    "element": row["element"].strip(),
                    "f_plus": float(row["f_plus"]),
                    "f_minus": float(row["f_minus"]),
                    "dual_descriptor": float(row["dual_descriptor"]),
                }
            )
    return rows
