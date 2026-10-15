# Smoothed Duality Gaps Beyond Convexity

M2 internship (Master in Optimization, Université Paris-Saclay) at Télécom
Paris, LTCI, supervised by Olivier Fercoq.

> **Smoothed Duality Gaps Beyond Convexity: KKT Detection, Pointwise Smoothing
> Thresholds, and Proximal Escape under Affine Constraints**

Smoothed duality gaps (SDG) are computable merit functions for convex
primal–dual optimization. This work studies the self-centered SDG for a
ρ-weakly convex objective under affine constraints:

* if `β_x > ρ` and `β_y > 0`, the zero set of the SDG is **exactly the KKT set**,
  and the gap controls feasibility and a Lagrangian proximal fixed-point
  residual;
* for arbitrary smoothing, the zero set is characterized with the
  level-proximal subdifferential through a **pointwise support threshold**;
  below the safe threshold a center-initialized local inner solver may return
  false zeros;
* since KKT points need not be global minimizers, a one-sided **proximal
  escape** test (feasible perturbation followed by hard-constrained proximal
  polishing) rejects nonglobal KKT points;
* experiments with complete analytic stationary catalogs compare the SDG, a
  proximal residual and a KKT residual as root-discovery objectives (the SDG
  recovers all 27 KKT branches of a weakly convex box problem).

## Repository layout

| Folder | Content |
|---|---|
| [`report/`](report) | Internship report (`main.tex`, sections, bibliography, figure scripts) and the compiled `main.pdf` |
| [`experiments/`](experiments) | Self-contained Python package reproducing every number and figure of the experimental section (`python -m sdg_report_experiments`), with tests |

## Quick start

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
python -m sdg_report_experiments          # strict reproduction of Table 6.1 and Figures 6.1–6.3
```

## References

* O. Fercoq (2023). Quadratic error bound of the smoothed gap and the restarted
  averaged primal-dual hybrid gradient. *Open Journal of Mathematical Optimization* 4.
* I. Walwil, O. Fercoq (2025). The smoothed duality gap as a stopping criterion.
  *Mathematical Programming Computation* 17.
* R. T. Rockafellar (1974). Augmented Lagrange multiplier functions and duality in
  nonconvex programming. *SIAM Journal on Control* 12(2).
* G. Dinc Yalcin, R. Kasimbeyli (2020). On weak conjugacy, augmented Lagrangians and
  duality in nonconvex optimization. *Mathematical Methods of Operations Research* 92.

The full bibliography is in `report/references.bib`.
