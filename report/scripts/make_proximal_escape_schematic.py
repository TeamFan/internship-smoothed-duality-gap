"""Build the explanatory proximal-escape schematic used in Chapter 5.

The left panel instantiates one safe proximal-polishing step on a tilted
double-well objective.  The right panel is qualitative and depicts the three
radius classes proved in Theorem 5.2; it is not experimental data.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.4,
        "axes.titlesize": 9.4,
        "axes.labelsize": 9.2,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def objective(t: np.ndarray | float, tilt: float = 0.15) -> np.ndarray | float:
    """Tilted double well with weak-convexity modulus one."""

    return 0.25 * (np.asarray(t) ** 2 - 1.0) ** 2 - tilt * np.asarray(t)


def prox(center: float, stepsize: float = 0.7, tilt: float = 0.15) -> float:
    """Unique safe proximal point for the scalar illustrative model."""

    roots = np.roots(
        [
            1.0,
            0.0,
            stepsize**-1 - 1.0,
            -(tilt + center / stepsize),
        ]
    )
    real_roots = [float(root.real) for root in roots if abs(root.imag) < 1.0e-10]
    if len(real_roots) != 1:
        raise RuntimeError("The safe proximal model should have one real root.")
    return real_roots[0]


def build_figure() -> plt.Figure:
    navy = "#28527A"
    orange = "#D97706"
    teal = "#2A9D8F"
    gray = "#5B6573"
    light_gray = "#D5DAE0"

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.15),
        gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.31},
    )

    # Panel (a): one exact, safe proximal-polishing witness.
    grid = np.linspace(-1.55, 1.65, 900)
    curve = objective(grid)
    stationary = np.sort(np.roots([1.0, 0.0, -1.0, -0.15]).real)
    x_local = float(stationary[0])
    x_global = float(stationary[-1])
    center = 0.35
    polished = prox(center)

    ax_left.fill_between(
        grid,
        curve,
        objective(x_local),
        where=curve < objective(x_local),
        color=teal,
        alpha=0.09,
        linewidth=0,
    )
    ax_left.plot(grid, curve, color=gray, linewidth=1.55)
    ax_left.axhline(
        objective(x_local),
        color=light_gray,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax_left.scatter(
        [x_local, x_global],
        [objective(x_local), objective(x_global)],
        s=31,
        color=[orange, teal],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax_left.scatter(
        [center, polished],
        [objective(center), objective(polished)],
        s=34,
        marker="D",
        color=[navy, teal],
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
    )

    arrow_height = -0.27
    ax_left.annotate(
        "",
        xy=(center, arrow_height),
        xytext=(x_local, arrow_height),
        arrowprops={
            "arrowstyle": "-|>",
            "color": orange,
            "linewidth": 1.25,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax_left.text(
        0.5 * (x_local + center),
        arrow_height - 0.035,
        r"$d=z-x\in\ker A$",
        color=orange,
        ha="center",
        va="top",
    )
    ax_left.annotate(
        r"$p=P_\lambda^{\mathcal{C}}(z)$",
        xy=(polished, objective(polished)),
        xytext=(0.78, 0.28),
        color=navy,
        ha="left",
        arrowprops={
            "arrowstyle": "->",
            "color": navy,
            "linewidth": 1.0,
            "connectionstyle": "arc3,rad=-0.22",
        },
    )
    ax_left.annotate(
        r"$x$ (nonglobal local minimum)",
        xy=(x_local, objective(x_local)),
        xytext=(-1.48, 0.39),
        color=orange,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": orange, "linewidth": 0.8},
    )
    ax_left.annotate(
        r"$x^\star$",
        xy=(x_global, objective(x_global)),
        xytext=(1.23, -0.32),
        color=teal,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": teal, "linewidth": 0.8},
    )
    ax_left.text(
        center,
        objective(center) + 0.055,
        r"center $z$",
        color=navy,
        ha="center",
    )
    ax_left.annotate(
        r"$f(p)<f(x)$",
        xy=(polished, objective(polished)),
        xytext=(-0.18, -0.18),
        color=teal,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": teal, "linewidth": 0.9},
    )
    ax_left.set_xlim(-1.58, 1.67)
    ax_left.set_ylim(-0.38, 0.62)
    ax_left.set_xlabel(r"feasible coordinate on $\mathcal{C}$")
    ax_left.set_ylabel(r"$f$")
    ax_left.set_title("(a) Perturb the center, then polish")
    ax_left.grid(axis="y", color="#E8EBEF", linewidth=0.55)

    # Panel (b): qualitative shapes implied by the radius trichotomy.
    delta = np.linspace(0.0, 1.6, 500)
    threshold = 0.62
    nonminimum = 0.34 * (1.0 - np.exp(-4.0 * delta))
    excess_radius = np.maximum(delta - threshold, 0.0)
    nonglobal_local = 0.46 * (1.0 - np.exp(-4.7 * excess_radius))
    global_minimum = np.zeros_like(delta)

    ax_right.plot(
        delta,
        nonminimum,
        color=orange,
        linewidth=1.65,
        label=r"not locally minimizing: $r^{\rm esc}=0$",
    )
    ax_right.plot(
        delta,
        nonglobal_local,
        color=navy,
        linewidth=1.65,
        label=r"nonglobal local minimum: $0<r^{\rm esc}<\infty$",
    )
    ax_right.scatter(
        [threshold],
        [0.0],
        s=25,
        facecolor=navy,
        edgecolor=navy,
        linewidth=1.0,
        zorder=6,
    )
    ax_right.plot(
        delta,
        global_minimum,
        color=teal,
        linewidth=1.85,
        linestyle=(0, (4, 2)),
        label=r"global minimum: $r^{\rm esc}=\infty$",
    )
    ax_right.axvline(
        threshold,
        color=gray,
        linewidth=0.9,
        linestyle=(0, (2.5, 2.5)),
    )
    ax_right.annotate(
        r"$\delta=r_\lambda^{\rm esc}(x)$:"
        "\n"
        r"zero decrease at the threshold",
        xy=(threshold, 0.0),
        xytext=(threshold + 0.08, 0.14),
        color=gray,
        arrowprops={"arrowstyle": "->", "color": gray, "linewidth": 0.8},
    )
    ax_right.set_xlim(0.0, 1.6)
    ax_right.set_ylim(-0.015, 0.48)
    ax_right.set_xlabel(r"search radius $\delta$")
    ax_right.set_ylabel(r"$\mathcal{R}_{\lambda,\delta}(x)$")
    ax_right.set_title("(b) Three possible radius classes")
    ax_right.grid(axis="y", color="#E8EBEF", linewidth=0.55)
    ax_right.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.24),
        frameon=False,
        handlelength=2.5,
    )

    for axis in (ax_left, ax_right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3, width=0.7)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.31)
    return fig


def main() -> None:
    report_root = Path(__file__).resolve().parents[1]
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    figure.savefig(
        figure_dir / "proximal_escape_schematic.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        figure_dir / "proximal_escape_schematic.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
