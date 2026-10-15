"""Draw the exact two-dimensional quartic geometry used in Section 4.

The panels share the same analytic objective; no optimization or stochastic
data enter this illustration.  Outputs are local to the report directory.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quartic(t):
    return 0.25 * t**4 - 0.5 * t**2


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.4, "axes.labelsize": 9.2, "axes.titlesize": 9.4,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    # Match the restrained palette and marker treatment of the Section 5
    # schematic; circles identify minima and a diamond the central maximum.
    blue, teal, orange = "#28527A", "#2A9D8F", "#D97706"
    gray, light_gray = "#5B6573", "#D5DAE0"
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), layout="constrained")
    t = np.linspace(-0.3, 1.3, 501)
    xx, yy = np.meshgrid(t, t)
    left, right = axes
    left.contour(xx, yy, quartic(xx) + quartic(yy),
                 levels=[-0.45, -0.35, -0.25, -0.15, -0.05],
                 colors="#AEB8C2", linewidths=0.65, linestyles="solid")
    left.plot(t, 1-t, color=blue, linewidth=1.65)
    left.scatter([0, 1], [1, 0], color=teal, s=31,
                 edgecolor="white", linewidth=0.6, zorder=4)
    left.scatter([0.5], [0.5], color=orange, marker="D", s=34,
                 edgecolor="white", linewidth=0.6, zorder=4)
    for xy, label, offset in [
        ((0, 1), r"$(0,1)$", (7, 7)),
        ((1, 0), r"$(1,0)$", (-36, -15)),
        ((0.5, 0.5), r"$(\frac{1}{2},\frac{1}{2})$", (8, 6)),
    ]:
        left.annotate(label, xy, xytext=offset, textcoords="offset points",
                      fontsize=8.2, color=gray)
    left.text(-0.24, -0.20, r"$\mathcal{C}:\ x_1+x_2=1$", color=blue,
              fontsize=8.5)
    left.set(xlim=(-0.3, 1.3), ylim=(-0.3, 1.3), xlabel=r"$x_1$",
             ylabel=r"$x_2$", title="(a) Feasible geometry", aspect="equal")
    left.set_xticks([0, 0.5, 1])
    left.set_yticks([0, 0.5, 1])

    right.plot(t, quartic(t)+quartic(1-t), color=blue, linewidth=1.65)
    right.axhline(-0.25, color=light_gray, linestyle=(0, (3, 2)), linewidth=0.9)
    right.scatter([0, 1], [-0.25, -0.25], color=teal, s=31,
                  edgecolor="white", linewidth=0.6, zorder=4)
    right.scatter([0.5], [-7/32], color=orange, marker="D", s=34,
                  edgecolor="white", linewidth=0.6, zorder=4)
    right.annotate("local maximum", (0.5, -7/32), xytext=(0, 10),
                   textcoords="offset points", ha="center", color=orange,
                   fontsize=8.5)
    right.text(0.5, -0.265, "global minima", ha="center", color=teal,
               fontsize=8.5)
    right.set(xlim=(-0.3, 1.3), ylim=(-0.275, -0.17), xlabel=r"$t$",
              ylabel=r"$f(t,1-t)$", title="(b) Objective on the feasible line")
    right.set_xticks([0, 0.5, 1])
    right.set_yticks([-0.25, -0.22, -0.19])
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E8EBEF", linewidth=0.55)
        axis.tick_params(length=3, width=0.7, color=gray)
    target = Path(__file__).resolve().parents[1] / "figures"
    target.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(target / f"affine_quartic_geometry.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
