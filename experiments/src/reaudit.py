"""Recheck retained box endpoints without rerunning the local optimizers.

The revision table compares the former multiplier-scaled tests with the
primal-scaled tests on exactly the same points. Catalog matching is assessed
separately from residual acceptance. No search trajectories are changed.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from . import affine_rank_one, box_qp
from .common import load_json, write_csv
from .runner import DEFAULT_CONFIG, RESULTS, _write_table_fragment


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    problem = box_qp.make_instance(config["box_qp"], int(config["seed"]))
    catalog = box_qp.analytic_catalog(problem)
    path = RESULTS / "candidates" / "box_qp_endpoints.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    changes = {method: [] for method in box_qp.METHODS}
    for row in candidates:
        x = np.asarray(json.loads(row["x"]), dtype=float)
        y = float(row["multiplier"])
        natural, _, gradient = problem.natural_mapping(x, y)
        prox, _ = problem.lagrangian_prox(x, y)
        audit = problem.audit(x, y)
        legacy_accepted = max(
            audit["R_feas"], audit["R_Cprox"],
            np.linalg.norm(natural) / (1 + np.linalg.norm(gradient)),
            problem.beta_x * np.linalg.norm(x - prox)
            / (1 + problem.beta_x * np.linalg.norm(x) + abs(y)),
        ) <= box_qp.ACCEPTANCE_TOL
        accepted = bool(audit["accepted"])
        nearest, distance = box_qp._nearest_catalog(x, y, catalog)
        matched = accepted and distance <= box_qp.MATCHING_TOL
        changes[row["method"]].append((legacy_accepted, accepted, matched, nearest["oracle_id"]))
        row.update(audit)
        row.update(matched=matched, oracle_distance=distance,
                   oracle_id=nearest["oracle_id"] if matched else "unmatched",
                   oracle_type=nearest["type"] if matched else "unmatched")
    revisions = []
    for method, checks in changes.items():
        revisions.append({
            "method": method,
            "legacy_accepted": sum(int(item[0]) for item in checks),
            "revised_accepted": sum(int(item[1]) for item in checks),
            "matched_endpoints": sum(int(item[2]) for item in checks),
            "accepted_unmatched": sum(int(item[1] and not item[2]) for item in checks),
            "distinct_roots": len({item[3] for item in checks if item[2]}),
            "changed_decisions": sum(int(item[0] != item[1]) for item in checks),
        })
    summary_path = RESULTS / "tables" / "branch_recovery_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    for row in summary:
        if row["family"] == "box QP":
            revision = next(item for item in revisions if item["method"] == row["method"])
            row.update(accepted_endpoints=revision["revised_accepted"],
                       distinct_roots=revision["distinct_roots"])
    write_csv(path, candidates)
    write_csv(summary_path, summary)
    write_csv(RESULTS / "tables" / "box_audit_revision.csv", revisions)
    _write_table_fragment(summary, RESULTS / "tables" / "branch_recovery_rows.tex")

    # Archive the inputs underlying the retained run, after checking that the
    # saved initial points agree with the documented generator substreams.
    affine = affine_rank_one.make_instance(config["affine_rank_one"], int(config["seed"]))
    arrays = {}
    for family, expected in (
        ("affine_rank_one", affine_rank_one._starts(affine, int(config["affine_rank_one"]["starts"]), int(config["seed"]))),
        ("box_qp", box_qp._sobol_starts(problem, int(config["box_qp"]["sobol_power"]), int(config["seed"]))),
    ):
        with (RESULTS / "candidates" / f"{family}_endpoints.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            np.testing.assert_array_equal(json.loads(row["initial"]), expected[int(row["start_id"])])
        arrays[f"{family}_starts"] = expected
    inputs = RESULTS / "inputs"
    inputs.mkdir(exist_ok=True)
    np.savez(inputs / "reported_inputs.npz", V=affine.objective.eigenvectors,
             M=affine.objective.matrix, eigenvalues=affine.objective.eigenvalues,
             box_q=problem.q, box_r=problem.r, **arrays)
    print(json.dumps(revisions, indent=2))


if __name__ == "__main__":
    main()
