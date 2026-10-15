"""Exact proximal-escape diagnostics retained in the report.

Two deterministic illustrations are produced:

* the complete radius classification of the separable box-QP catalog;
* a radial point where both the gradient and Hessian vanish, together with the
  flat global-minimum control ``q_min(x)=||x||^4``.

No direction-search algorithm is tested here.  The directions are analytic and
the figures visualize the value-based statements proved in Section 5.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

# A non-interactive backend keeps figure generation identical on workstations,
# CI runners and headless archival services.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from .box_qp import BoxQP
from .common import configure_matplotlib


Array = np.ndarray


def _box_escape_row(problem: BoxQP, row: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the exact ideal escape radius to one catalog entry."""

    negative_interiors = list(row["negative_interiors"])
    wrong_bounds = list(row["wrong_bounds"])
    if negative_interiors:
        radius = 0.0
        radius_class = "zero"
        witness_index = int(negative_interiors[0])
    elif wrong_bounds:
        thresholds = []
        for index in wrong_bounds:
            a_i = -float(problem.r[index]) / float(problem.q[index])
            thresholds.append((a_i * (2.0 + problem.lam * float(problem.q[index])), index))
        radius, witness_index = min(thresholds)
        radius_class = "positive finite"
    else:
        radius = math.inf
        radius_class = "infinite"
        witness_index = None
    return {
        "oracle_id": row["oracle_id"],
        "state": row["state"],
        "oracle_type": row["type"],
        "objective": row["objective"],
        "escape_radius": radius,
        "radius_class": radius_class,
        "witness_coordinate": "" if witness_index is None else witness_index + 1,
        "x": row["x"],
    }


def box_escape_catalog(
    problem: BoxQP, catalog: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = [_box_escape_row(problem, row) for row in catalog]
    expected = {"zero": 19, "positive finite": 7, "infinite": 1}
    counts = {
        key: sum(row["radius_class"] == key for row in rows) for key in expected
    }
    if counts != expected:
        raise RuntimeError(f"unexpected box radius classes: {counts}")
    return rows


def _representatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def prefix(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return tuple(str(row["state"]).split("|")[:3])  # type: ignore[return-value]

    return {
        "saddle": next(
            row for row in rows if prefix(row) == ("interior", "upper", "upper")
        ),
        "local": next(
            row for row in rows if prefix(row) == ("lower", "upper", "upper")
        ),
        "global": next(
            row for row in rows if prefix(row) == ("upper", "upper", "upper")
        ),
    }


def _box_drop_curve(
    problem: BoxQP, row: Mapping[str, Any], role: str, radii: Array
) -> Array:
    """Evaluate the exact full-ball polished-decrease profile.

    For the IUU/LUU/UUU representatives only the first concave coordinate can
    reduce the objective.  Therefore the Euclidean-ball outer problem is
    exactly one-dimensional, rather than a sampled directional approximation.
    """

    radius = np.asarray(radii, dtype=float)
    if role == "global":
        return np.zeros_like(radius)
    state = str(row["state"]).split("|")
    index = 1 + next(i for i, label in enumerate(state[:3]) if label != "upper")
    q_i = float(problem.q[index])
    h_i = -q_i
    a_i = -float(problem.r[index]) / q_i
    slope = 1.0 - problem.lam * h_i
    if role == "saddle":
        movement = np.minimum(radius / slope, 1.0 - a_i)
        return 0.5 * h_i * movement**2
    if role == "local":
        activation = problem.lam * h_i * a_i
        movement = np.clip((radius - activation) / slope, 0.0, 1.0)
        return np.maximum(0.0, 0.5 * h_i * movement * (movement - 2.0 * a_i))
    raise ValueError(f"unknown representative role {role!r}")


def plot_box_states(problem: BoxQP, output_directory: Path) -> None:
    """Illustrate L/I/U using the actual second coordinate of the box model.

    This explains the 3**3 catalog; it is not an additional benchmark.
    """
    configure_matplotlib()
    q, r = float(problem.q[1]), float(problem.r[1])
    a = -r / q
    t = np.linspace(0.0, 1.0, 401)
    def value(s):
        return 0.5 * q * s**2 + r * s
    figure, axis = plt.subplots(figsize=(4.9, 2.65), constrained_layout=True)
    axis.plot(t, value(t), color="#334155", linewidth=1.8)
    axis.axhline(0, color="#CBD5E1", linewidth=0.7, zorder=0)
    axis.vlines(a, 0, value(a), color="#B15D40", linestyle=":", linewidth=1)
    for s, label, color, offset in (
        (0.0, "$L$", "#BA861B", (9, 8)),
        (a, "$I$", "#B15D40", (0, 9)),
        (1.0, "$U$", "#397866", (-12, 6)),
    ):
        axis.scatter(s, value(s), color=color, s=32, zorder=3)
        axis.annotate(label, (s, value(s)), xytext=offset,
                      textcoords="offset points", ha="center", fontsize=11, color=color)
    axis.set_xticks([0, a, 1], ["$0$", "$a_i$", "$1$"])
    axis.set_yticks([0], ["$0$"])
    axis.set_xlim(-0.035, 1.035)
    axis.set_ylim(value(1) - 0.045, value(a) + 0.075)
    axis.set_xlabel("coordinate $t$")
    axis.set_ylabel(r"$F_i(t)=\frac{1}{2} q_i t^2+r_i t$")
    output_directory.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 240})):
        figure.savefig(output_directory / f"box_states.{suffix}",
                       bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)


def plot_box_escape(
    problem: BoxQP, rows: list[dict[str, Any]], output_directory: Path
) -> None:
    configure_matplotlib()
    colors = {"saddle": "#D55E00", "local": "#E69F00", "global": "#0072B2"}
    labels = {
        "saddle": "IUU strict saddle",
        "local": "LUU nonglobal local minimum",
        "global": "UUU global minimum",
    }
    representatives = _representatives(rows)
    local_radius = float(representatives["local"]["escape_radius"])
    radii = np.linspace(0.0, max(0.64, 1.45 * local_radius), 700)
    figure, axis = plt.subplots(figsize=(7.2, 3.05), constrained_layout=True)
    for role, linestyle in (("saddle", "-"), ("local", "-"), ("global", "--")):
        axis.plot(
            radii,
            _box_drop_curve(problem, representatives[role], role, radii),
            linestyle=linestyle,
            linewidth=2.1 if role != "global" else 1.9,
            color=colors[role],
            label=labels[role],
        )
    axis.axvspan(0.0, local_radius, color=colors["local"], alpha=0.07, linewidth=0)
    axis.axvline(local_radius, color=colors["local"], linestyle=":", linewidth=1.2)
    axis.annotate(
        r"exact $r_{\rm esc}$ for LUU",
        xy=(local_radius, 0.0),
        xytext=(local_radius + 0.025, 0.065),
        color=colors["local"],
        fontsize=8.2,
        arrowprops={"arrowstyle": "->", "color": colors["local"], "linewidth": 0.85},
    )
    maximum = max(
        float(np.max(_box_drop_curve(problem, representatives[role], role, radii)))
        for role in ("saddle", "local")
    )
    axis.set_xlim(0.0, float(radii[-1]))
    axis.set_ylim(-0.015 * maximum, 1.08 * maximum)
    axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.set_xlabel(r"affine-feasible center-ball radius $\delta$")
    axis.set_ylabel(r"$\mathcal{R}_{\lambda,\delta}(x)$")
    axis.set_title("Exact full-ball escape profiles", loc="left")
    axis.legend(loc="upper left", frameon=False)
    output_directory.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 320})):
        figure.savefig(
            output_directory / f"box_qp_proximal_escape.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(figure)


def _q3_value(t: float) -> float:
    return 1.0 + (t * t - 1.0) ** 3


def _q3_prox(center: float, lam: float) -> float:
    """Return the radial safe prox for the center ``center*e_1``."""

    def stationarity(t: float) -> float:
        return 6.0 * t * (t * t - 1.0) ** 2 + (t - center) / lam

    if not 0.0 < center < 1.0 or not 0.0 < lam * (24.0 / 5.0) < 1.0:
        raise ValueError("radial illustration requires 0<center<1 and a safe step")
    return float(brentq(stationarity, 0.0, center, xtol=1.0e-14, rtol=2.0e-14))


def hessian_flat_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    lam = float(config["lambda"])
    rho = float(config["rho"])
    rows: list[dict[str, Any]] = []
    for delta in sorted(float(item) for item in config["radii"]):
        center = 1.0 - delta
        prox = _q3_prox(center, lam)
        # Stable algebraic differences avoid cancellation near objective 1.
        # These are floating-point evaluations, not interval certificates.
        direct = (2.0 * delta - delta * delta) ** 3
        polished = ((1.0 - prox) * (1.0 + prox)) ** 3
        envelope = polished - 0.5 / lam * (prox - center) ** 2
        root_residual = abs(6.0 * prox * (prox * prox - 1.0) ** 2
                            + (prox - center) / lam)
        if not 0.0 < direct <= envelope <= polished:
            raise RuntimeError("q3 decrease ordering failed")
        rows.append(
            {
                "model": "q3",
                "delta": delta,
                "lambda": lam,
                "rho": rho,
                "lambda_rho": lam * rho,
                "direct_drop": direct,
                "envelope_drop": envelope,
                "prox_polished_drop": polished,
                "prox_coordinate": prox,
                "prox_stationarity_residual": root_residual,
                "root_bracket_lower": 0.0,
                "root_bracket_upper": center,
            }
        )
        # The flat global minimum is a sign control: all witnessed decreases
        # are zero after taking the positive part.
        rows.append(
            {
                "model": "q_min",
                "delta": delta,
                "lambda": lam,
                "rho": 0.0,
                "lambda_rho": 0.0,
                "direct_drop": 0.0,
                "envelope_drop": 0.0,
                "prox_polished_drop": 0.0,
                "prox_coordinate": "not_needed",
                "prox_stationarity_residual": "not_needed",
                "root_bracket_lower": "not_needed",
                "root_bracket_upper": "not_needed",
            }
        )
    return rows


def plot_hessian_flat(rows: list[dict[str, Any]], output_directory: Path) -> None:
    configure_matplotlib()
    q3_rows = sorted(
        (row for row in rows if row["model"] == "q3"), key=lambda row: float(row["delta"])
    )
    fields = (
        ("direct_drop", "direct", "#1F77B4", "o"),
        ("envelope_drop", "envelope", "#E56B2F", "s"),
        ("prox_polished_drop", "prox-polished", "#4C956C", "^"),
    )
    figure, axis = plt.subplots(figsize=(5.25, 2.9))
    floor = 100.0 * np.finfo(float).eps * (1.0 + abs(_q3_value(1.0)))
    for field, label, color, marker in fields:
        reliable = [row for row in q3_rows if float(row[field]) > floor]
        axis.loglog(
            [float(row["delta"]) for row in reliable],
            [float(row[field]) for row in reliable],
            color=color,
            marker=marker,
            markersize=3.7,
            linewidth=1.35,
            label=label,
        )
    axis.text(
        0.05,
        0.08,
        r"flat control $q_{\min}$: no positive decrease",
        transform=axis.transAxes,
        fontsize=8.0,
        color="#334155",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.92,
        },
    )
    axis.text(
        0.95,
        0.94,
        r"$\lambda=1/8\quad(\lambda\rho=0.6)$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color="#475569",
    )
    axis.set_xlabel(r"center radius $\delta$")
    axis.set_ylabel("witnessed objective decrease")
    axis.grid(color="#D9E1E8", linewidth=0.6, alpha=0.75)
    axis.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        columnspacing=1.15,
        handlelength=1.8,
    )
    figure.subplots_adjust(left=0.14, right=0.985, bottom=0.22, top=0.82)
    output_directory.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 240})):
        figure.savefig(
            output_directory / f"experiment_high_order_escape.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(figure)
