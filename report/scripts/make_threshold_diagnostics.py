"""Build the compact analytic threshold diagnostics used in Section 4.

The two panels are recomputed directly from the displayed quartic models.  No
archived numerical experiment or optimizer output is treated as ground truth:

* panel (a) enumerates all real stationary points of each scalar quartic
  proximal subproblem and takes the smallest objective value;
* panel (b) does the same for the tilted quartic.  The dashed curve records
  the center-initialized local branch, which stays at the higher local
  minimizer and therefore reports a zero computed gap.

The stationary point, its pointwise threshold, and all plotted gap values are
obtained from the polynomial equations below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
FIGURE_PNG_PATH = WORKSPACE / "report" / "figures" / "threshold_diagnostics.png"
FIGURE_PDF_PATH = WORKSPACE / "report" / "figures" / "threshold_diagnostics.pdf"


def real_roots(coefficients: list[float], tolerance: float = 1.0e-9) -> list[float]:
    roots = np.roots(coefficients)
    return sorted(float(root.real) for root in roots if abs(float(root.imag)) <= tolerance)


def double_well_value(t: float) -> float:
    return 0.25 * t**4 - 0.5 * t**2


def coordinate_gap(x: float, multiplier: float, beta: float) -> float:
    candidates = real_roots([1.0, 0.0, beta - 1.0, multiplier - beta * x])
    values = [
        double_well_value(u)
        + multiplier * u
        + 0.5 * beta * (u - x) ** 2
        for u in candidates
    ]
    current = double_well_value(x) + multiplier * x
    return max(0.0, current - min(values))


def affine_double_well_gaps(beta: float) -> tuple[float, float]:
    endpoint = coordinate_gap(0.0, 0.0, beta) + coordinate_gap(1.0, 0.0, beta)
    symmetric = 2.0 * coordinate_gap(0.5, 0.375, beta)
    return endpoint, symmetric


def tilted_value(t: float, tilt: float = 0.1) -> float:
    return 0.25 * t**4 - 0.5 * t**2 + tilt * t


def tilted_true_gap(x: float, beta: float, tilt: float = 0.1) -> float:
    candidates = real_roots([1.0, 0.0, beta - 1.0, tilt - beta * x])
    values = [tilted_value(u, tilt) + 0.5 * beta * (u - x) ** 2 for u in candidates]
    gap = tilted_value(x, tilt) - min(values)
    return 0.0 if abs(gap) <= 5.0e-12 else max(0.0, gap)


def higher_local_point() -> tuple[float, float]:
    """Return the higher-valued local minimizer and its exact threshold.

    Stationarity is ``x^3-x+0.1=0``.  Among the two roots with positive second
    derivative, the one with larger objective value is the nonglobal local
    minimum.  Section 4 proves ``beta_crit=(1-x^2)_+`` at any stationary root.
    """

    stationary = real_roots([1.0, 0.0, -1.0, 0.1])
    local_minima = [x for x in stationary if 3.0 * x * x - 1.0 > 0.0]
    if len(local_minima) != 2:
        raise RuntimeError("the tilted quartic should have two local minima")
    x_high = max(local_minima, key=tilted_value)
    return x_high, max(0.0, 1.0 - x_high * x_high)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.6,
            "axes.titlesize": 9.3,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    x_high, beta_crit = higher_local_point()

    beta_affine = np.linspace(0.0, 1.08, 361)
    affine_values = np.array([affine_double_well_gaps(float(beta)) for beta in beta_affine])
    beta_tilted = np.linspace(0.0, 0.16, 321)
    true_tilted = np.array([tilted_true_gap(x_high, float(beta)) for beta in beta_tilted])
    local_tilted = np.zeros_like(beta_tilted)

    if np.max(np.abs(affine_values[beta_affine >= 1.0])) > 2.0e-10:
        raise RuntimeError("Endpoint SDG did not vanish at the analytic threshold beta=1.")
    if np.max(np.abs(affine_values[beta_affine >= 0.75, 1])) > 2.0e-10:
        raise RuntimeError("Symmetric SDG did not vanish at the analytic threshold beta=3/4.")

    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))

    ax = axes[0]
    ax.plot(
        beta_affine,
        affine_values[:, 0],
        color="#2a9d55",
        linewidth=2.25,
        label=r"endpoint KKT: $\beta_{\rm crit}=1$",
    )
    ax.plot(
        beta_affine,
        affine_values[:, 1],
        color="#7353a6",
        linewidth=2.25,
        label=r"symmetric KKT: $\beta_{\rm crit}=3/4$",
    )
    ax.axvline(0.75, color="#7353a6", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axvline(1.0, color="#343434", linestyle="--", linewidth=1.2)
    ax.text(0.75, 1.485, r"$3/4$", ha="center", va="bottom", color="#7353a6")
    ax.text(1.0, 1.485, r"$\rho=1$", ha="center", va="bottom", color="#343434")
    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(-0.015, 1.55)
    ax.set_xlabel(r"primal smoothing curvature $\beta_x$")
    ax.set_ylabel("exact SDG value")
    ax.set_title("(a) Pair-dependent zero thresholds", loc="left")
    ax.legend(loc="center right", frameon=False)

    ax = axes[1]
    ax.axvspan(0.0, beta_crit, color="#f8d7da", alpha=0.55, linewidth=0)
    ax.plot(
        beta_tilted,
        true_tilted,
        color="#d62728",
        linewidth=2.35,
    )
    ax.plot(
        beta_tilted,
        local_tilted,
        color="#2764d8",
        linestyle=(0, (5, 3)),
        linewidth=2.0,
    )
    ax.axvline(beta_crit, color="#343434", linestyle="--", linewidth=1.2)
    sample_beta = np.array([0.05, 0.08, 0.10])
    sample_gap = np.array([tilted_true_gap(x_high, float(beta)) for beta in sample_beta])
    ax.scatter(sample_beta, sample_gap, color="#d62728", s=28, zorder=4)
    ax.annotate(
        "algorithmic false-zero region",
        xy=(0.055, tilted_true_gap(x_high, 0.055)),
        xytext=(0.012, 0.158),
        arrowprops={"arrowstyle": "->", "color": "#8c1d20", "lw": 1.0},
        color="#8c1d20",
    )
    ax.text(
        0.005,
        0.19,
        "exact value-defined SDG",
        color="#d62728",
        ha="left",
        va="top",
    )
    ax.text(
        0.155,
        0.008,
        "center-initialized local value",
        color="#2764d8",
        ha="right",
        va="bottom",
    )
    ax.text(
        beta_crit + 0.003,
        0.183,
        rf"$\beta_{{\rm crit}}={beta_crit:.6f}$",
        ha="left",
        va="top",
        color="#343434",
    )
    ax.set_xlim(0.0, 0.16)
    ax.set_ylim(-0.006, 0.205)
    ax.set_xlabel(r"primal smoothing curvature $\beta_x$")
    ax.set_ylabel(r"gap at the higher local minimizer")
    ax.set_title("(b) False zero from a local inner solve", loc="left")

    fig.tight_layout(w_pad=1.5)
    FIGURE_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_PNG_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURE_PDF_PATH,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"wrote {FIGURE_PNG_PATH}")
    print(f"wrote {FIGURE_PDF_PATH}")
    print(f"higher local minimizer x={x_high:.15g}, beta_crit={beta_crit:.15g}")


if __name__ == "__main__":
    main()
