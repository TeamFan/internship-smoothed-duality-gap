# Experiments for *Smoothed Duality Gaps Beyond Convexity*

This directory is the self-contained numerical companion to the internship
report.  It contains only the experiments retained in Section 6.  All problem
data are synthetic and are generated from `config/report.json`; the code reads
no other results or modules.

## Results covered

| Report item | Code | Generated artifact |
| --- | --- | --- |
| Table 6.1, affine rank-one rows | `affine_rank_one.py` | `results/tables/branch_recovery_summary.csv` |
| Table 6.1, box-QP rows | `box_qp.py` | `results/tables/branch_recovery_summary.csv` |
| Box-state illustration, Figure 6.1 | `escape.py` | `box_states.pdf` |
| Proposition 6.1 audit and Figure 6.2 | `escape.py` | `box_escape_catalog.csv`, `box_qp_proximal_escape.pdf` |
| Hessian-inconclusive illustration and Figure 6.3 | `escape.py` | `hessian_flat_decreases.csv`, `experiment_high_order_escape.pdf` |

The per-start endpoint tables are retained in `results/candidates/` so each
aggregate count can be audited without rerunning the optimizers.  They are
generated evidence, never source inputs.

## Repository layout

```text
experiments/
├── config/report.json                 # seeds, dimensions and report checks
├── src/                              # package mapped to sdg_report_experiments
├── tests/test_invariants.py           # fast catalog/radius/order checks
├── results/candidates/                # one audited row per initial point
├── results/inputs/reported_inputs.npz  # actual matrices, coefficients and starts
├── results/tables/                    # compact catalogs and report table
├── results/figures/                   # Figures 6.1--6.3, PDF and PNG
├── requirements.txt                   # exact reported dependency versions
└── CITATION.cff                       # GitHub citation metadata
```

## Environment

The reported run used CPython 3.14 on Windows 11.  Exact NumPy, SciPy and
Matplotlib versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

No GPU or non-standard hardware is required.
On the recorded Windows/CPython environment, the complete 1,248-start run takes
about seven seconds; the measured value for each run is written to
`results/run_metadata.json`.

## Reproduce the report artifacts

From this directory, run:

```bash
python -m sdg_report_experiments
```

The command performs the following steps in a fixed order:

1. generate the common affine Gaussian starts and the scrambled box Sobol set;
2. minimize the three reported merit functions with the stated L-BFGS-B rules;
3. audit every endpoint independently and match it to the complete catalog;
4. assert the six counts printed in Table 6.1;
5. generate the exact box escape catalog and the three report figures.

If any reported count or analytic invariant changes, the command exits with an
error instead of silently replacing the published result.

**Platform note.** The counts in Table 6.1 were produced with the pinned
versions on Windows 11 / CPython 3.14. On macOS arm64 (same pinned NumPy,
SciPy and Matplotlib, CPython 3.11) one affine start of the SC-SDG method
(start 45) terminates at a merit value of 0 just inside the acceptance
tolerance, giving 160/160 instead of 159/160; all 768 box-QP endpoints are
identical. To run on such a platform without touching the published files, use

```bash
python -m sdg_report_experiments --no-strict
```

which reports the deviations and writes every output to `results_local/`.

Run the quick analytic tests separately with:

```bash
python -m unittest discover -s tests -v
```

To re-audit retained box endpoints without changing the optimization runs:

```bash
python -m sdg_report_experiments.reaudit
```

This compares the former multiplier-scaled acceptance test with the revised
primal-scaled test and records separate acceptance and catalog-match counts
in `results/tables/box_audit_revision.csv`. The retained 768 endpoints give
unchanged decisions and no accepted unmatched endpoints. The same command
verifies the documented generators against every saved initial point and
saves the matrices, coefficients and common starting arrays in
`results/inputs/reported_inputs.npz`.

## Reading the code

- [Implementation details](IMPLEMENTATION.md) records random substreams,
  optimizer options, derivative conventions, and endpoint matching rules.
- `config/report.json` is the single source of seeds, dimensions, tolerances
  and expected report counts.
- `affine_rank_one.py` documents the secular-root proximal oracle and the
  complete `2n-1` KKT catalog.
- `box_qp.py` separates the box-only Lagrangian prox used for branch discovery
  from the equality-constrained prox used for escape.
- `escape.py` marks which reductions are exact full-ball calculations and
  which direction in the Hessian-flat example is prescribed analytically.
- `runner.py` owns all output paths and records exact software versions in
  `results/run_metadata.json`.

## Scope

The repository reproduces the finite runs reported in the text.  It does not
contain the discarded unconstrained calibration, local-minimizer baselines,
parameter scans, heatmaps or historical development experiments.

## Citation and archival release

`CITATION.cff` provides GitHub's machine-readable citation metadata.  Before a
public release, select an explicit open-source license, create a versioned tag,
archive that tag in a DOI-minting repository such as Zenodo, and add the final
repository URL and DOI to `CITATION.cff`.
