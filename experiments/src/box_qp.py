"""Separable weakly-convex box-QP used in Section 6.

The diagonal construction is deliberate: its complete KKT catalog has
``3^3=27`` branches, while both the Lagrangian prox used by the search merits
and the hard-constrained prox used by the escape diagnostic are available in
closed form.  The three merit functions are compared with identical scrambled
Sobol starts and the same L-BFGS-B settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


Array = np.ndarray
METHODS = ("sc_sdg", "proximal_residual", "natural_kkt_residual")
ACCEPTANCE_TOL = 1.0e-7
MATCHING_TOL = 2.0e-5


@dataclass(frozen=True)
class BoxQP:
    q: Array
    r: Array
    rhs: float
    beta_x: float
    beta_y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", np.asarray(self.q, dtype=float).copy())
        object.__setattr__(self, "r", np.asarray(self.r, dtype=float).copy())
        if not self.beta_x > self.rho:
            raise ValueError("the report requires beta_x > rho")

    @property
    def dimension(self) -> int:
        return int(self.q.size)

    @property
    def rho(self) -> float:
        return max(0.0, -float(np.min(self.q)))

    @property
    def lam(self) -> float:
        return 1.0 / self.beta_x

    def objective(self, x: Array) -> float:
        point = np.asarray(x, dtype=float)
        return 0.5 * float(np.dot(self.q * point, point)) + float(
            np.dot(self.r, point)
        )

    def gradient(self, x: Array) -> Array:
        return self.q * np.asarray(x, dtype=float) + self.r

    def feasibility(self, x: Array) -> float:
        return float(np.asarray(x, dtype=float)[0] - self.rhs)

    def lagrangian_prox(
        self, center: Array, multiplier: float
    ) -> tuple[Array, Array]:
        """Return the exact box-only prox appearing in the SDG merits."""

        numerator = self.beta_x * np.asarray(center, dtype=float) - self.r
        numerator = numerator.copy()
        numerator[0] -= float(multiplier)
        raw = numerator / (self.q + self.beta_x)
        return np.clip(raw, 0.0, 1.0), raw

    def constrained_prox(self, center: Array) -> Array:
        """Return the exact prox with both the box and equality imposed."""

        raw = (self.beta_x * np.asarray(center, dtype=float) - self.r) / (
            self.q + self.beta_x
        )
        prox = np.clip(raw, 0.0, 1.0)
        prox[0] = self.rhs
        return prox

    def natural_mapping(
        self, x: Array, multiplier: float
    ) -> tuple[Array, Array, Array]:
        point = np.asarray(x, dtype=float)
        lagrangian_gradient = self.gradient(point)
        lagrangian_gradient = lagrangian_gradient.copy()
        lagrangian_gradient[0] += float(multiplier)
        shifted = point - lagrangian_gradient
        return point - np.clip(shifted, 0.0, 1.0), shifted, lagrangian_gradient

    def audit(self, x: Array, multiplier: float) -> dict[str, float | bool]:
        """Apply the four independent endpoint checks reported in the text.

        ``R_prox`` uses the box-constrained Lagrangian prox from the branch
        search.  ``R_Cprox`` is a separate audit based on the hard-constrained
        prox used by the escape diagnostic.  The names deliberately keep the
        two proximal constructions distinct.
        """

        point = np.asarray(x, dtype=float)
        feasibility = self.feasibility(point)
        natural, _, _ = self.natural_mapping(point, multiplier)
        prox, _ = self.lagrangian_prox(point, multiplier)
        constrained = self.constrained_prox(point)
        r_feas = abs(feasibility) / (1.0 + abs(self.rhs))
        # Projection residuals saturate on the fixed box; an unbounded
        # multiplier must not be able to inflate their normalization.
        r_stat = float(np.linalg.norm(natural)) / (1.0 + float(np.linalg.norm(point)))
        r_prox = self.beta_x * float(np.linalg.norm(point - prox)) / (
            1.0
            + self.beta_x * float(np.linalg.norm(point))
        )
        r_cprox = self.beta_x * float(np.linalg.norm(point - constrained)) / (
            1.0 + self.beta_x * float(np.linalg.norm(point))
        )
        accepted = bool(
            np.all(np.isfinite(point))
            and math.isfinite(multiplier)
            and max(r_feas, r_stat, r_prox, r_cprox) <= ACCEPTANCE_TOL
        )
        return {
            "R_feas": r_feas,
            "R_stat": r_stat,
            "R_prox": r_prox,
            "R_Cprox": r_cprox,
            "accepted": accepted,
        }


def make_instance(config: Mapping[str, Any], seed: int) -> BoxQP:
    """Reconstruct the fixed seven-dimensional synthetic instance."""

    dimension = int(config["dimension"])
    concave = int(config["concave_coordinates"])
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 1414, dimension])
    )
    q_values = np.empty(dimension, dtype=float)
    r_values = np.empty(dimension, dtype=float)
    q_values[0], r_values[0] = 1.1, -0.22
    for offset, index in enumerate(range(1, 1 + concave)):
        q_values[index] = -2.0 - 0.12 * offset
        stationary = 0.30 + 0.035 * offset
        r_values[index] = -q_values[index] * stationary
    for offset, index in enumerate(range(1 + concave, dimension)):
        q_values[index] = 1.25 + 0.15 * offset
        stationary = 0.28 + 0.08 * ((offset + 1) % 4)
        r_values[index] = -q_values[index] * stationary

    # The perturbation removes repeated Hessian entries.  Recomputing r keeps
    # the stationary coordinates printed in the report exactly unchanged.
    q_values += rng.uniform(-2.0e-4, 2.0e-4, size=dimension)
    for offset, index in enumerate(range(1, 1 + concave)):
        r_values[index] = -q_values[index] * (0.30 + 0.035 * offset)
    for offset, index in enumerate(range(1 + concave, dimension)):
        r_values[index] = -q_values[index] * (
            0.28 + 0.08 * ((offset + 1) % 4)
        )
    return BoxQP(
        q=q_values,
        r=r_values,
        rhs=float(config["constraint_rhs"]),
        beta_x=float(config["beta_x"]),
        beta_y=float(config["beta_y"]),
    )


def _coordinate_states(problem: BoxQP, index: int) -> list[tuple[str, float]]:
    q_i, r_i = float(problem.q[index]), float(problem.r[index])
    states: list[tuple[str, float]] = []
    if r_i >= -1.0e-12:
        states.append(("lower", 0.0))
    if q_i + r_i <= 1.0e-12:
        states.append(("upper", 1.0))
    if abs(q_i) > 1.0e-14:
        stationary = -r_i / q_i
        if 1.0e-12 < stationary < 1.0 - 1.0e-12:
            states.append(("interior", stationary))
    return states


def analytic_catalog(problem: BoxQP) -> list[dict[str, Any]]:
    choices = [
        _coordinate_states(problem, index)
        for index in range(1, problem.dimension)
    ]
    multiplier = -(float(problem.q[0]) * problem.rhs + float(problem.r[0]))
    rows: list[dict[str, Any]] = []
    for root_number, states in enumerate(itertools.product(*choices)):
        x = np.empty(problem.dimension, dtype=float)
        x[0] = problem.rhs
        labels: list[str] = []
        negative_interiors: list[int] = []
        wrong_bounds: list[int] = []
        for index, (label, value) in enumerate(states, start=1):
            x[index] = value
            labels.append(label)
            if label == "interior" and problem.q[index] < 0.0:
                negative_interiors.append(index)
        if not negative_interiors:
            for index, label in enumerate(labels, start=1):
                if problem.q[index] >= 0.0:
                    continue
                preferred = (
                    "upper"
                    if 0.5 * float(problem.q[index]) + float(problem.r[index]) < 0.0
                    else "lower"
                )
                if label != preferred:
                    wrong_bounds.append(index)
        if negative_interiors:
            point_type = "strict saddle"
        elif wrong_bounds:
            point_type = "non-global local minimum"
        else:
            point_type = "global minimum"
        rows.append(
            {
                "oracle_id": f"box_qp_root_{root_number:04d}",
                "state": "|".join(labels),
                "type": point_type,
                "global": point_type == "global minimum",
                "x": x,
                "multiplier": multiplier,
                "objective": problem.objective(x),
                "negative_interiors": negative_interiors,
                "wrong_bounds": wrong_bounds,
            }
        )
    return rows


def _sobol_starts(problem: BoxQP, power: int, seed: int) -> Array:
    sampler = qmc.Sobol(
        d=problem.dimension + 1,
        scramble=True,
        seed=np.random.SeedSequence(
            [seed, 991, problem.dimension]
        ).generate_state(1)[0],
    )
    unit = sampler.random_base2(m=power)
    starts = unit.copy()
    starts[:, -1] = -4.0 + 8.0 * unit[:, -1]
    return starts


def _nearest_catalog(
    x: Array, multiplier: float, catalog: list[dict[str, Any]]
) -> tuple[dict[str, Any], float]:
    distances = [
        math.hypot(
            float(np.linalg.norm(x - np.asarray(row["x"], dtype=float))),
            float(multiplier - float(row["multiplier"])),
        )
        for row in catalog
    ]
    index = int(np.argmin(distances))
    return catalog[index], float(distances[index])


class Merit:
    """Piecewise-analytic value/Jacobian selection used by L-BFGS-B."""

    def __init__(self, problem: BoxQP, method: str) -> None:
        self.problem = problem
        self.method = method

    def __call__(self, variable: Array) -> tuple[float, Array]:
        x = np.asarray(variable[:-1], dtype=float)
        multiplier = float(variable[-1])
        if self.method == "sc_sdg":
            return self._sdg(x, multiplier)
        if self.method == "proximal_residual":
            return self._proximal_residual(x, multiplier)
        if self.method == "natural_kkt_residual":
            return self._natural_kkt_residual(x, multiplier)
        raise ValueError(f"unknown box merit {self.method!r}")

    def _sdg(self, x: Array, multiplier: float) -> tuple[float, Array]:
        p = self.problem
        prox, _ = p.lagrangian_prox(x, multiplier)
        feasibility = p.feasibility(x)
        prox_feasibility = p.feasibility(prox)
        displacement = prox - x
        value = (
            p.objective(x)
            + multiplier * feasibility
            - p.objective(prox)
            - multiplier * prox_feasibility
            - 0.5 * p.beta_x * float(np.dot(displacement, displacement))
            + 0.5 / p.beta_y * feasibility**2
        )
        gradient_x = p.gradient(x)
        gradient_x[0] += multiplier + feasibility / p.beta_y
        gradient_x -= p.beta_x * (x - prox)
        gradient_y = float(x[0] - prox[0])
        return float(value), np.r_[gradient_x, gradient_y]

    def _proximal_residual(
        self, x: Array, multiplier: float
    ) -> tuple[float, Array]:
        p = self.problem
        prox, raw = p.lagrangian_prox(x, multiplier)
        feasibility = p.feasibility(x)
        residual = p.beta_x * (x - prox)
        # At clipping surfaces we use the deterministic inactive-piece value
        # ``free=0``.  This is a generalized-derivative convention, not a
        # claim that the merit is globally C1.
        free = ((raw > 0.0) & (raw < 1.0)).astype(float)
        dp_dx = free * p.beta_x / (p.q + p.beta_x)
        dp_dy = np.zeros(p.dimension, dtype=float)
        dp_dy[0] = -free[0] / (p.q[0] + p.beta_x)
        gradient_x = p.beta_x * (1.0 - dp_dx) * residual
        gradient_x[0] += feasibility
        gradient_y = float(np.dot(-p.beta_x * dp_dy, residual))
        value = 0.5 * (float(np.dot(residual, residual)) + feasibility**2)
        return value, np.r_[gradient_x, gradient_y]

    def _natural_kkt_residual(
        self, x: Array, multiplier: float
    ) -> tuple[float, Array]:
        p = self.problem
        residual, shifted, _ = p.natural_mapping(x, multiplier)
        feasibility = p.feasibility(x)
        free = ((shifted > 0.0) & (shifted < 1.0)).astype(float)
        jacobian_x = 1.0 - free + free * p.q
        jacobian_y = np.zeros(p.dimension, dtype=float)
        jacobian_y[0] = free[0]
        gradient_x = jacobian_x * residual
        gradient_x[0] += feasibility
        gradient_y = float(np.dot(jacobian_y, residual))
        value = 0.5 * (float(np.dot(residual, residual)) + feasibility**2)
        return value, np.r_[gradient_x, gradient_y]


def run(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    problem = make_instance(config, seed)
    catalog = analytic_catalog(problem)
    starts = _sobol_starts(problem, int(config["sobol_power"]), seed)
    bounds = [(0.0, 1.0)] * problem.dimension + [(None, None)]
    candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for method in METHODS:
        accepted_ids: set[str] = set()
        accepted_count = 0
        for start_id, initial in enumerate(starts):
            result = minimize(
                Merit(problem, method),
                np.asarray(initial, dtype=float),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": int(config["max_iterations"]),
                    "ftol": 1.0e-15,
                    "gtol": 2.0e-10,
                    "maxls": 60,
                },
            )
            x = np.asarray(result.x[:-1], dtype=float)
            multiplier = float(result.x[-1])
            audit = problem.audit(x, multiplier)
            accepted = bool(audit.pop("accepted"))
            nearest, distance = _nearest_catalog(x, multiplier, catalog)
            matched = bool(accepted and distance <= MATCHING_TOL)
            if accepted:
                accepted_count += 1
            if matched:
                accepted_ids.add(str(nearest["oracle_id"]))
            candidates.append(
                {
                    "family": "box QP",
                    "method": method,
                    "start_id": start_id,
                    "initial": initial,
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "optimizer_success": bool(result.success),
                    "merit_value": float(Merit(problem, method)(result.x)[0]),
                    "accepted": accepted,
                    "matched": matched,
                    "oracle_id": nearest["oracle_id"] if matched else "unmatched",
                    "oracle_type": nearest["type"] if matched else "unmatched",
                    "oracle_distance": distance,
                    "objective": problem.objective(x),
                    "multiplier": multiplier,
                    "x": x,
                    "nfev": int(result.nfev),
                    **audit,
                }
            )
        summaries.append(
            {
                "family": "box QP",
                "method": method,
                "accepted_endpoints": accepted_count,
                "starts": int(starts.shape[0]),
                "distinct_roots": len(accepted_ids),
                "catalog_roots": len(catalog),
            }
        )

    maximum_catalog_residual = 0.0
    for row in catalog:
        audit = problem.audit(np.asarray(row["x"]), float(row["multiplier"]))
        audit.pop("accepted")
        maximum_catalog_residual = max(
            maximum_catalog_residual, *(float(value) for value in audit.values())
        )
    if len(catalog) != 27 or maximum_catalog_residual > 2.0e-12:
        raise RuntimeError(
            "box analytic catalog failed its cardinality or residual audit"
        )

    return {
        "summary": summaries,
        "candidates": candidates,
        "catalog": catalog,
        "problem": problem,
        "metadata": {
            "seed": seed,
            "rho": problem.rho,
            "beta_x": problem.beta_x,
            "beta_y": problem.beta_y,
            "catalog_residual_max": maximum_catalog_residual,
        },
    }
