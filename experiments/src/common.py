"""Shared serialization and plotting utilities.

The experiment modules return ordinary dictionaries rather than writing files.
Only :mod:`runner` owns artifact paths.  This separation keeps the numerical
definitions testable and makes every committed output traceable to one entry
point.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import numpy as np


METHOD_LABELS = {
    "sc_sdg": "SC--SDG",
    "proximal_residual": "Lagrangian prox residual",
    "kkt_residual": "KKT residual",
    "natural_kkt_residual": "natural KKT residual",
}


def configure_matplotlib() -> None:
    """Apply the report's print-oriented figure style."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.6,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 7.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _csv_value(value: Any) -> Any:
    """Convert arrays and non-finite numbers to stable CSV representations."""

    if isinstance(value, np.ndarray):
        return json.dumps([float(item) for item in value])
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isinf(number):
            return "+inf" if number > 0.0 else "-inf"
        if math.isnan(number):
            return "nan"
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
