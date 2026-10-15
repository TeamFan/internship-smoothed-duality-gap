"""Draw the exact two-part decomposition of one self-centered SDG evaluation.

The numerical values are illustrative, but every curve and annotation is
computed from the displayed one-dimensional weakly convex model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def objective(u: np.ndarray | float) -> np.ndarray | float:
    """A 1-weakly convex double-well objective."""

    return 0.25 * np.asarray(u) ** 4 - 0.5 * np.asarray(u) ** 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # A safe self-centered SDG evaluation for A=1 and Au=b.
    x = 0.0
    y = 0.4
    b = 0.35
    beta_x = 1.8  # beta_x > rho = 1
    beta_y = 0.7
    residual = x - b

    # q'(u)=u^3+(beta_x-1)u+y-beta_x*x has exactly one real root.
    roots = np.roots([1.0, 0.0, beta_x - 1.0, y - beta_x * x])
    real_roots = roots[np.abs(roots.imag) < 1.0e-12].real
    if real_roots.size != 1:
        raise RuntimeError(f"Expected one real proximal root, got {roots!r}")
    p = float(real_roots[0])
    v_star = y + residual / beta_y

    def lagrangian(u: np.ndarray | float) -> np.ndarray | float:
        return objective(u) + y * (np.asarray(u) - b)

    def primal_model(u: np.ndarray | float) -> np.ndarray | float:
        return lagrangian(u) + 0.5 * beta_x * (np.asarray(u) - x) ** 2

    def dual_model(v: np.ndarray | float) -> np.ndarray | float:
        return objective(x) + np.asarray(v) * residual - 0.5 * beta_y * (
            np.asarray(v) - y
        ) ** 2

    primal_defect = float(primal_model(x) - primal_model(p))
    feasibility_defect = float(residual**2 / (2.0 * beta_y))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    blue = "#2F5D8A"
    orange = "#C96A32"
    ink = "#26323D"
    grey = "#7B8792"
    pale_blue = "#EAF1F7"
    pale_orange = "#FAEEE7"

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.65))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.29, wspace=0.30)

    # Primal comparison: global minimum of the regularized Lagrangian model.
    ax = axes[0]
    u_grid = np.linspace(-1.28, 1.08, 700)
    ax.plot(
        u_grid,
        lagrangian(u_grid),
        color=grey,
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        label=r"$\mathcal{L}(u,y)$",
    )
    ax.plot(
        u_grid,
        primal_model(u_grid),
        color=blue,
        linewidth=2.0,
        label=r"$q_x(u)=\mathcal{L}(u,y)+\frac{\beta_x}{2}\|u-x\|^2$",
    )
    ax.scatter([x], [primal_model(x)], s=28, color=ink, zorder=5)
    ax.scatter([p], [primal_model(p)], s=34, color=blue, zorder=5)
    ax.axvline(x, color=ink, linewidth=0.8, alpha=0.55)
    ax.axvline(p, color=blue, linewidth=0.8, alpha=0.55)
    ax.annotate(
        r"center $x$",
        (x, primal_model(x)),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
        color=ink,
    )
    ax.annotate(
        r"global prox $p$",
        (p, primal_model(p)),
        xytext=(-4, -20),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        color=blue,
    )
    arrow_x = 0.83
    ax.hlines(
        [primal_model(p), primal_model(x)],
        xmin=p,
        xmax=arrow_x,
        color=blue,
        linestyle=":",
        linewidth=0.9,
        alpha=0.75,
    )
    ax.annotate(
        "",
        xy=(arrow_x, primal_model(x)),
        xytext=(arrow_x, primal_model(p)),
        arrowprops=dict(arrowstyle="<->", color=blue, linewidth=1.3),
    )
    ax.text(
        arrow_x - 0.03,
        0.5 * (primal_model(x) + primal_model(p)),
        "proximal\n defect",
        fontsize=7.5,
        color=blue,
        va="center",
        ha="right",
    )
    ax.set_title("(a) Regularize the primal comparison")
    ax.set_xlabel("comparison point $u$")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", frameon=False, handlelength=2.4)

    # Dual comparison: exact maximum of a concave quadratic.
    ax = axes[1]
    v_grid = np.linspace(-0.82, 0.92, 500)
    ax.plot(v_grid, dual_model(v_grid), color=orange, linewidth=2.0)
    ax.scatter([y], [dual_model(y)], s=28, color=ink, zorder=5)
    ax.scatter([v_star], [dual_model(v_star)], s=34, color=orange, zorder=5)
    ax.axvline(y, color=ink, linewidth=0.8, alpha=0.55)
    ax.axvline(v_star, color=orange, linewidth=0.8, alpha=0.55)
    ax.annotate(
        r"center $y$",
        (y, dual_model(y)),
        xytext=(7, -18),
        textcoords="offset points",
        fontsize=8,
        color=ink,
    )
    ax.annotate(
        r"$v_\beta=y+(Ax-b)/\beta_y$",
        (v_star, dual_model(v_star)),
        xytext=(0, -35),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        color=orange,
    )
    arrow_x = 0.72
    ax.hlines(
        [dual_model(y), dual_model(v_star)],
        xmin=v_star,
        xmax=arrow_x,
        color=orange,
        linestyle=":",
        linewidth=0.9,
        alpha=0.78,
    )
    ax.annotate(
        "",
        xy=(arrow_x, dual_model(v_star)),
        xytext=(arrow_x, dual_model(y)),
        arrowprops=dict(arrowstyle="<->", color=orange, linewidth=1.3),
    )
    ax.text(
        arrow_x - 0.035,
        0.5 * (dual_model(y) + dual_model(v_star)),
        "feasibility\n defect",
        fontsize=7.5,
        color=orange,
        va="center",
        ha="right",
    )
    ax.set_title("(b) Maximize the dual comparison exactly")
    ax.set_xlabel("comparison multiplier $v$")
    ax.set_ylabel("regularized dual value")

    for axis in axes:
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.6, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3, color=grey)

    fig.text(
        0.262,
        0.135,
        rf"$\mathcal{{L}}(x,y)-\min_u q_x(u)\approx {primal_defect:.4f}$",
        ha="center",
        va="center",
        fontsize=9,
        color=blue,
        bbox=dict(boxstyle="round,pad=0.34", facecolor=pale_blue, edgecolor=blue),
    )
    fig.text(0.500, 0.135, r"$+$", ha="center", va="center", fontsize=13, color=ink)
    fig.text(
        0.738,
        0.135,
        rf"$\|Ax-b\|^2/(2\beta_y)\approx {feasibility_defect:.4f}$",
        ha="center",
        va="center",
        fontsize=9,
        color=orange,
        bbox=dict(
            boxstyle="round,pad=0.34", facecolor=pale_orange, edgecolor=orange
        ),
    )
    fig.text(
        0.5,
        0.035,
        rf"$\mathcal{{G}}_{{\beta_x,\beta_y}}(x,y)"
        rf"\approx {primal_defect + feasibility_defect:.4f}$ in this illustration",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=ink,
    )

    for suffix in ("pdf", "png"):
        target = OUT / f"sdg_decomposition_schematic.{suffix}"
        fig.savefig(target, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
