r"""Affine rank-one stationary-branch recovery.

This module implements exactly the smooth calibration reported in Section 6.1:

.. math::

   f(x)=\tfrac14\lVert xx^\top-M\rVert_F^2,
   \qquad v_1^\top x=0.4.

The spectrum is simple, so all ``2n-1`` KKT pairs are known analytically.  The
catalog is used only after local optimization for matching; it is never used
to initialize a run.  All three merit functions receive the same 160 starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import brentq, minimize


Array = np.ndarray
METHODS = ("sc_sdg", "proximal_residual", "kkt_residual")
ACCEPTANCE_TOL = 1.0e-7
MATCHING_TOL = 8.0e-5


def _orthogonal_matrix(dimension: int, seed: int) -> Array:
    """Return the sign-normalized QR factor fixed by the report seed."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, 1401]))
    raw = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[np.newaxis, :]


@dataclass(frozen=True)
class RankOneProblem:
    """Rank-one PSD factorization with its safe global proximal oracle."""

    eigenvalues: Array
    eigenvectors: Array
    matrix: Array
    beta_x: float

    @property
    def dimension(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def rho(self) -> float:
        return float(self.eigenvalues[0])

    @property
    def matrix_norm_squared(self) -> float:
        return float(np.dot(self.eigenvalues, self.eigenvalues))

    def value(self, x: Array) -> float:
        point = np.asarray(x, dtype=float)
        squared_norm = float(np.dot(point, point))
        return 0.25 * (
            squared_norm**2
            - 2.0 * float(point @ self.matrix @ point)
            + self.matrix_norm_squared
        )

    def gradient(self, x: Array) -> Array:
        point = np.asarray(x, dtype=float)
        return float(np.dot(point, point)) * point - self.matrix @ point

    def hessian(self, x: Array) -> Array:
        point = np.asarray(x, dtype=float)
        return (
            2.0 * np.outer(point, point)
            + float(np.dot(point, point)) * np.eye(self.dimension)
            - self.matrix
        )

    def prox(self, center: Array) -> Array:
        """Solve the unique prox through the monotone spectral secular root.

        In the eigenbasis, stationarity gives

        ``p_i = beta_x*z_i/(beta_x + ||p||^2 - d_i)``.

        Since ``beta_x > max(d_i)``, the proximal model is strongly convex and
        the scalar equation for ``||p||^2`` has one nonnegative root.
        """

        z = np.asarray(center, dtype=float)
        coordinates = self.eigenvectors.T @ z
        if float(np.linalg.norm(coordinates)) == 0.0:
            return np.zeros_like(z)

        def secular(squared_norm: float) -> float:
            denominator = self.beta_x + squared_norm - self.eigenvalues
            p_coordinates = self.beta_x * coordinates / denominator
            return float(np.dot(p_coordinates, p_coordinates) - squared_norm)

        upper = max(1.0, float(np.dot(z, z)))
        while secular(upper) > 0.0:
            upper *= 2.0
            if not math.isfinite(upper):
                raise FloatingPointError("failed to bracket the secular root")
        squared_norm = float(
            brentq(secular, 0.0, upper, xtol=1.0e-13, rtol=2.0e-14)
        )
        p_coordinates = self.beta_x * coordinates / (
            self.beta_x + squared_norm - self.eigenvalues
        )
        return self.eigenvectors @ p_coordinates


@dataclass(frozen=True)
class AffineInstance:
    objective: RankOneProblem
    normal: Array
    rhs: float
    beta_y: float

    def feasibility(self, x: Array) -> float:
        return float(np.dot(self.normal, x) - self.rhs)

    def lagrangian_prox(self, x: Array, multiplier: float) -> Array:
        shifted_center = x - (multiplier / self.objective.beta_x) * self.normal
        return self.objective.prox(shifted_center)

    def audit(self, x: Array, multiplier: float) -> dict[str, float | bool]:
        gradient = self.objective.gradient(x)
        stationarity = gradient + multiplier * self.normal
        feasibility = self.feasibility(x)
        prox = self.lagrangian_prox(x, multiplier)
        normal_term = abs(multiplier)  # ||A^T y|| because ||v_1||=1.
        residuals = {
            "R_feas": abs(feasibility) / (1.0 + abs(self.rhs)),
            "R_stat": float(np.linalg.norm(stationarity))
            / (1.0 + float(np.linalg.norm(gradient)) + normal_term),
            "R_prox": self.objective.beta_x * float(np.linalg.norm(x - prox))
            / (
                1.0
                + self.objective.beta_x * float(np.linalg.norm(x))
                + normal_term
            ),
        }
        residuals["accepted"] = bool(max(residuals.values()) <= ACCEPTANCE_TOL)
        return residuals


def make_instance(config: Mapping[str, Any], seed: int) -> AffineInstance:
    dimension = int(config["dimension"])
    eigenvalues = np.linspace(4.0, 0.5, dimension, dtype=float)
    eigenvectors = _orthogonal_matrix(dimension, seed)
    matrix = (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.T
    # lambda*rho=0.8, hence beta_x=5>rho=4.
    objective = RankOneProblem(eigenvalues, eigenvectors, matrix, beta_x=5.0)
    return AffineInstance(
        objective=objective,
        normal=eigenvectors[:, 0].copy(),
        rhs=float(config["constraint_rhs"]),
        beta_y=float(config["beta_y"]),
    )


def analytic_catalog(instance: AffineInstance) -> list[dict[str, Any]]:
    """Enumerate the complete ``2n-1`` KKT catalog."""

    c = instance.rhs
    values = instance.objective.eigenvalues
    vectors = instance.objective.eigenvectors
    base = c * vectors[:, 0]
    rows: list[dict[str, Any]] = [
        {
            "oracle_id": "affine_axis",
            "type": "strict local maximum",
            "x": base,
            "multiplier": (float(values[0]) - c * c) * c,
        }
    ]
    for index in range(1, instance.objective.dimension):
        amplitude = math.sqrt(float(values[index]) - c * c)
        multiplier = (float(values[0]) - float(values[index])) * c
        point_type = "global minimum" if index == 1 else "strict saddle"
        for sign in (-1, 1):
            rows.append(
                {
                    "oracle_id": f"affine_eig_{index:02d}_{'pos' if sign > 0 else 'neg'}",
                    "type": point_type,
                    "x": base + sign * amplitude * vectors[:, index],
                    "multiplier": multiplier,
                }
            )
    return rows


def _starts(instance: AffineInstance, count: int, seed: int) -> Array:
    """Generate the common oracle-free isotropic starts used in the report."""

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 1441, instance.objective.dimension])
    )
    directions = rng.normal(size=(count, instance.objective.dimension))
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-15
    )
    radii = np.exp(rng.uniform(math.log(0.08), math.log(3.2), size=count))
    x = directions * radii[:, None]
    y = rng.normal(scale=1.8, size=count)
    return np.column_stack((x, y))


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
    """Value/Jacobian callback for one of the three reported search merits."""

    def __init__(self, instance: AffineInstance, method: str) -> None:
        self.instance = instance
        self.method = method

    def __call__(self, variable: Array) -> tuple[float, Array]:
        x = np.asarray(variable[:-1], dtype=float)
        multiplier = float(variable[-1])
        if self.method == "sc_sdg":
            return self._sdg(x, multiplier)
        if self.method == "proximal_residual":
            return self._proximal_residual(x, multiplier)
        if self.method == "kkt_residual":
            return self._kkt_residual(x, multiplier)
        raise ValueError(f"unknown affine merit {self.method!r}")

    def _sdg(self, x: Array, multiplier: float) -> tuple[float, Array]:
        p = self.instance
        f = p.objective
        feasibility = p.feasibility(x)
        prox = p.lagrangian_prox(x, multiplier)
        prox_feasibility = p.feasibility(prox)
        displacement = prox - x
        value = (
            f.value(x)
            + multiplier * feasibility
            - f.value(prox)
            - multiplier * prox_feasibility
            - 0.5 * f.beta_x * float(np.dot(displacement, displacement))
            + 0.5 / p.beta_y * feasibility**2
        )
        gradient_x = (
            f.gradient(x)
            + multiplier * p.normal
            - f.beta_x * (x - prox)
            + (feasibility / p.beta_y) * p.normal
        )
        gradient_y = float(np.dot(p.normal, x - prox))
        return float(value), np.r_[gradient_x, gradient_y]

    def _proximal_residual(
        self, x: Array, multiplier: float
    ) -> tuple[float, Array]:
        p = self.instance
        f = p.objective
        prox = p.lagrangian_prox(x, multiplier)
        feasibility = p.feasibility(x)
        residual = f.beta_x * (x - prox)
        proximal_hessian = f.hessian(prox) + f.beta_x * np.eye(f.dimension)
        inverse_residual = np.linalg.solve(proximal_hessian, residual)
        gradient_x = f.beta_x * (
            residual - f.beta_x * inverse_residual
        ) + feasibility * p.normal
        gradient_y = f.beta_x * float(np.dot(p.normal, inverse_residual))
        value = 0.5 * (float(np.dot(residual, residual)) + feasibility**2)
        return value, np.r_[gradient_x, gradient_y]

    def _kkt_residual(self, x: Array, multiplier: float) -> tuple[float, Array]:
        p = self.instance
        stationarity = p.objective.gradient(x) + multiplier * p.normal
        feasibility = p.feasibility(x)
        gradient_x = p.objective.hessian(x) @ stationarity + feasibility * p.normal
        gradient_y = float(np.dot(p.normal, stationarity))
        value = 0.5 * (float(np.dot(stationarity, stationarity)) + feasibility**2)
        return value, np.r_[gradient_x, gradient_y]


def run(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Run the three merit searches and return compact auditable records."""

    instance = make_instance(config, seed)
    catalog = analytic_catalog(instance)
    starts = _starts(instance, int(config["starts"]), seed)
    max_iterations = int(config["max_iterations"])
    candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for method in METHODS:
        accepted_ids: set[str] = set()
        accepted_count = 0
        for start_id, initial in enumerate(starts):
            result = minimize(
                Merit(instance, method),
                np.asarray(initial, dtype=float),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxfun": max_iterations,
                    "maxiter": max_iterations,
                    "ftol": 1.0e-15,
                    "gtol": 2.0e-10,
                    "maxls": 40,
                },
            )
            x = np.asarray(result.x[:-1], dtype=float)
            multiplier = float(result.x[-1])
            audit = instance.audit(x, multiplier)
            accepted = bool(audit.pop("accepted"))
            nearest, distance = _nearest_catalog(x, multiplier, catalog)
            matched = bool(
                accepted
                and distance
                <= MATCHING_TOL * (1.0 + float(np.linalg.norm(np.r_[x, multiplier])))
            )
            if accepted:
                accepted_count += 1
            if matched:
                accepted_ids.add(str(nearest["oracle_id"]))
            candidates.append(
                {
                    "family": "affine rank one",
                    "method": method,
                    "start_id": start_id,
                    "initial": initial,
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "optimizer_success": bool(result.success),
                    "merit_value": float(Merit(instance, method)(result.x)[0]),
                    "accepted": accepted,
                    "matched": matched,
                    "oracle_id": nearest["oracle_id"] if matched else "unmatched",
                    "oracle_type": nearest["type"] if matched else "unmatched",
                    "oracle_distance": distance,
                    "objective": instance.objective.value(x),
                    "multiplier": multiplier,
                    "x": x,
                    "nfev": int(result.nfev),
                    **audit,
                }
            )
        summaries.append(
            {
                "family": "affine rank one",
                "method": method,
                "accepted_endpoints": accepted_count,
                "starts": int(config["starts"]),
                "distinct_roots": len(accepted_ids),
                "catalog_roots": len(catalog),
            }
        )

    # The analytic catalog itself is checked independently of the searches.
    maximum_catalog_residual = 0.0
    for row in catalog:
        audit = instance.audit(
            np.asarray(row["x"], dtype=float), float(row["multiplier"])
        )
        audit.pop("accepted")
        maximum_catalog_residual = max(maximum_catalog_residual, *audit.values())
    if maximum_catalog_residual > 2.0e-10:
        raise RuntimeError(
            f"affine analytic catalog failed audit: {maximum_catalog_residual:.3e}"
        )

    return {
        "summary": summaries,
        "candidates": candidates,
        "catalog": catalog,
        "metadata": {
            "seed": seed,
            "rho": instance.objective.rho,
            "beta_x": instance.objective.beta_x,
            "beta_y": instance.beta_y,
            "catalog_residual_max": maximum_catalog_residual,
        },
    }
