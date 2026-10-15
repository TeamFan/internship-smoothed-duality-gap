"""Fast mathematical smoke tests; these do not run the multistart searches."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from sdg_report_experiments import affine_rank_one, box_qp, escape


ROOT = Path(__file__).resolve().parents[1]


class ReportExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "config" / "report.json").read_text(encoding="utf-8")
        )
        cls.seed = int(cls.config["seed"])

    def test_affine_catalog(self) -> None:
        instance = affine_rank_one.make_instance(
            self.config["affine_rank_one"], self.seed
        )
        catalog = affine_rank_one.analytic_catalog(instance)
        self.assertEqual(len(catalog), 15)
        residual = max(
            max(
                float(value)
                for key, value in instance.audit(row["x"], row["multiplier"]).items()
                if key != "accepted"
            )
            for row in catalog
        )
        self.assertLessEqual(residual, 2.0e-10)

    def test_box_catalog_and_escape_classes(self) -> None:
        problem = box_qp.make_instance(self.config["box_qp"], self.seed)
        catalog = box_qp.analytic_catalog(problem)
        self.assertEqual(len(catalog), 27)
        self.assertEqual(sum(row["type"] == "strict saddle" for row in catalog), 19)
        self.assertEqual(
            sum(row["type"] == "non-global local minimum" for row in catalog), 7
        )
        rows = escape.box_escape_catalog(problem, catalog)
        self.assertEqual(sum(row["radius_class"] == "zero" for row in rows), 19)
        self.assertEqual(
            sum(row["radius_class"] == "positive finite" for row in rows), 7
        )
        self.assertEqual(sum(row["radius_class"] == "infinite" for row in rows), 1)

    def test_hessian_flat_ordering(self) -> None:
        rows = escape.hessian_flat_rows(self.config["hessian_flat"])
        q3 = [row for row in rows if row["model"] == "q3"]
        self.assertEqual(len(q3), 10)
        for row in q3:
            self.assertGreater(row["direct_drop"], 0.0)
            self.assertLessEqual(row["direct_drop"], row["envelope_drop"])
            self.assertLessEqual(row["envelope_drop"], row["prox_polished_drop"])
        self.assertAlmostEqual(
            float(self.config["hessian_flat"]["lambda"])
            * float(self.config["hessian_flat"]["rho"]),
            0.6,
        )

    def test_box_audit_rejects_divergent_multiplier(self) -> None:
        """A KKT primal point with the wrong dual variable must be rejected."""
        problem = box_qp.make_instance(self.config["box_qp"], self.seed)
        for row in box_qp.analytic_catalog(problem):
            x, y = np.asarray(row["x"]), float(row["multiplier"])
            self.assertTrue(problem.audit(x, y)["accepted"])
            for shift in (1e8, -1e8, 1e12):
                audit = problem.audit(x, y + shift)
                self.assertFalse(audit["accepted"])
                self.assertGreater(audit["R_stat"], 0.10)
                self.assertGreater(audit["R_prox"], 0.10)

    def test_positive_stationary_gap(self) -> None:
        """The safe SDG stationary point in Section 3 has positive value."""
        beta = 30.0 / 19.0
        x, p = 2 / np.sqrt(19), 3 / np.sqrt(19)
        f = lambda t: t**4 / 4 - t**2 / 2
        self.assertAlmostEqual(p**3 - p, beta * (x - p))
        self.assertAlmostEqual(x**3 - x - beta * (x - p), 0)
        self.assertAlmostEqual(f(x) - f(p) - beta / 2 * (p - x)**2, 65 / 1444)

    def test_box_sdg_extension_gradient(self) -> None:
        """Audit the smooth value-function gradient, without differentiating clip."""
        problem = box_qp.make_instance(self.config["box_qp"], self.seed)
        merit = box_qp.Merit(problem, "sc_sdg")
        rng = np.random.default_rng(619)
        step = 1e-6
        for _ in range(20):
            z = np.r_[rng.uniform(0.01, 0.99, 7), rng.uniform(-4, 4)]
            _, gradient = merit(z)
            finite_difference = np.array([
                (merit(z + step * e)[0] - merit(z - step * e)[0]) / (2 * step)
                for e in np.eye(8)
            ])
            np.testing.assert_allclose(gradient, finite_difference, atol=2e-8, rtol=2e-6)

    def test_inexact_projected_envelope_bound(self) -> None:
        """Check the Section 5 bound on an exactly soluble concave quadratic."""
        rho, lam = 0.8, 0.5
        mu = 1 / lam - rho
        alpha = 0.75 / max(1 / lam, rho / (1 - lam * rho))
        envelope = lambda d: -rho * np.dot(d, d) / (2 * (1 - lam * rho))
        rng = np.random.default_rng(520)
        for _ in range(20):
            d = rng.normal(size=3)
            d /= max(1, np.linalg.norm(d))
            error = 0.01 * rng.normal(size=3)
            p = d / (1 - lam * rho)
            p_hat = p + error
            epsilon = mu * np.linalg.norm(error)
            trial = d - alpha * (d - p_hat) / lam
            next_d = trial / max(1, np.linalg.norm(trial))
            upper = (envelope(d) - np.linalg.norm(next_d - d)**2 / (4 * alpha)
                     + alpha * epsilon**2 / (1 - lam * rho)**2)
            self.assertLessEqual(envelope(next_d), upper + 1e-14)

    def test_stable_radial_values(self) -> None:
        """Check the cancellation-free formulas against high-precision values."""
        rows = {row["delta"]: row for row in escape.hessian_flat_rows(
            self.config["hessian_flat"]) if row["model"] == "q3"}
        expected = {
            1e-4: (8.002400780e-12, 8.006003662e-12),
            1e-2: (8.248037345e-6, 8.638675728e-6),
            1e-1: (1.137645391e-2, 2.059846140e-2),
        }
        for delta, (envelope, polished) in expected.items():
            row = rows[delta]
            np.testing.assert_allclose(
                [row["envelope_drop"], row["prox_polished_drop"]],
                [envelope, polished], rtol=1e-9, atol=0)
            self.assertLess(row["prox_stationarity_residual"], 1e-12)


if __name__ == "__main__":
    unittest.main()
