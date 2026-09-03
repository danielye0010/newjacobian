"""Compensation methods and convergence bookkeeping for fixed-bottom study."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from fixed_bottom_core import FixedBottomJaxResponse


MAX_ITER = 12
TOL = 1e-6
DIVERGENCE_RATIO = 1e8
SCALAR_ALPHAS = np.unique(
    np.concatenate(
        [
            np.logspace(-2.0, math.log10(2.0), 8),
            np.linspace(0.15, 1.45, 9),
        ]
    )
)


@dataclass
class MethodOutcome:
    method: str
    initial_projected: float
    final_projected: float
    initial_physical: float
    final_physical: float
    iterations: int
    converged: bool
    accepted_steps: int
    rejected_steps: int
    best_alpha: float | None
    notes: str
    final_c: np.ndarray | None
    final_q: np.ndarray
    final_residual: np.ndarray
    history: List[Dict[str, object]]

    def as_row(self, model: FixedBottomJaxResponse) -> Dict[str, object]:
        bottom_inf, bottom_l2 = model.bottom_violation(self.final_q)
        return {
            "geometry": model.mesh.name,
            "response_case": model.case.name,
            "mode_count": model.mode_count,
            "method": self.method,
            "initial_projected_residual_norm": self.initial_projected,
            "final_projected_residual_norm": self.final_projected,
            "projected_residual_ratio": self.final_projected / max(self.initial_projected, 1e-15),
            "initial_physical_residual_norm": self.initial_physical,
            "final_physical_residual_norm": self.final_physical,
            "physical_residual_ratio": self.final_physical / max(self.initial_physical, 1e-15),
            "final_bottom_violation_inf": bottom_inf,
            "final_bottom_violation_l2": bottom_l2,
            "iterations": self.iterations,
            "converged": self.converged,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "best_alpha_if_scalar": self.best_alpha if self.best_alpha is not None else "",
            "notes": self.notes,
        }


def _project(model: FixedBottomJaxResponse, residual: np.ndarray) -> np.ndarray:
    return model.psi_np.T @ (model.weights_dof_np * np.asarray(residual, dtype=float))


def _evaluate_c(model: FixedBottomJaxResponse, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    residual = model.residual(c)
    b = _project(model, residual)
    return residual, b, float(np.linalg.norm(b)), model.physical_norm(residual)


def _evaluate_q(model: FixedBottomJaxResponse, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    residual = model.residual_from_q(q)
    b = _project(model, residual)
    return residual, b, float(np.linalg.norm(b)), model.physical_norm(residual)


def _history_row(
    model: FixedBottomJaxResponse,
    method: str,
    iteration: int,
    projected: float,
    initial_projected: float,
    physical: float,
    initial_physical: float,
    step_norm: float,
    trust_parameter: float | str = "",
    predicted: float | str = "",
    actual: float | str = "",
    eta: float | str = "",
    accepted: bool = True,
) -> Dict[str, object]:
    return {
        "geometry": model.mesh.name,
        "response_case": model.case.name,
        "mode_count": model.mode_count,
        "method": method,
        "iteration": iteration,
        "projected_residual_norm": projected,
        "projected_residual_ratio": projected / max(initial_projected, 1e-15),
        "physical_residual_norm": physical,
        "physical_residual_ratio": physical / max(initial_physical, 1e-15),
        "step_norm": step_norm,
        "trust_region_radius_or_lambda": trust_parameter,
        "predicted_reduction": predicted,
        "actual_reduction": actual,
        "eta_ratio": eta,
        "accepted": accepted,
    }


def _initial_state(model: FixedBottomJaxResponse):
    c = model.c0()
    residual, b, projected, physical = _evaluate_c(model, c)
    return c, residual, b, projected, physical


def run_free_direct(model: FixedBottomJaxResponse) -> MethodOutcome:
    _, r0, _, initial_projected, initial_physical = _initial_state(model)
    q = -r0
    residual, _, projected, physical = _evaluate_q(model, q)
    history = [
        _history_row(model, "free_direct_inversion_reference", 0, initial_projected, initial_projected, initial_physical, initial_physical, 0.0),
        _history_row(
            model,
            "free_direct_inversion_reference",
            1,
            projected,
            initial_projected,
            physical,
            initial_physical,
            float(np.linalg.norm(q)),
        ),
    ]
    return MethodOutcome(
        "free_direct_inversion_reference",
        initial_projected,
        projected,
        initial_physical,
        physical,
        1,
        bool(projected / max(initial_projected, 1e-15) < TOL),
        1,
        0,
        None,
        "infeasible free-space reference q=-r0; bottom nodes are allowed to move",
        None,
        q,
        residual,
        history,
    )


def run_fixed_projected_direct(model: FixedBottomJaxResponse) -> MethodOutcome:
    _, r0, _, initial_projected, initial_physical = _initial_state(model)
    q = -r0
    q[model.bottom_dof_mask_np] = 0.0
    residual, _, projected, physical = _evaluate_q(model, q)
    history = [
        _history_row(
            model,
            "fixed_bottom_projected_direct_inversion",
            0,
            initial_projected,
            initial_projected,
            initial_physical,
            initial_physical,
            0.0,
        ),
        _history_row(
            model,
            "fixed_bottom_projected_direct_inversion",
            1,
            projected,
            initial_projected,
            physical,
            initial_physical,
            float(np.linalg.norm(q)),
        ),
    ]
    return MethodOutcome(
        "fixed_bottom_projected_direct_inversion",
        initial_projected,
        projected,
        initial_physical,
        physical,
        1,
        bool(projected / max(initial_projected, 1e-15) < TOL),
        1,
        0,
        None,
        "full-field direct inversion with bottom DOFs zeroed; no additional Laplacian smoothing",
        None,
        q,
        residual,
        history,
    )


def _finish_modal(
    model: FixedBottomJaxResponse,
    method: str,
    c: np.ndarray,
    initial_projected: float,
    initial_physical: float,
    history: List[Dict[str, object]],
    accepted_steps: int,
    rejected_steps: int,
    notes: str,
    best_alpha: float | None = None,
) -> MethodOutcome:
    residual, _, projected, physical = _evaluate_c(model, c)
    ratio = projected / max(initial_projected, 1e-15)
    q = model.q_from_c(c)
    return MethodOutcome(
        method,
        initial_projected,
        projected,
        initial_physical,
        physical,
        max(0, len(history) - 1),
        bool(np.isfinite(ratio) and ratio < TOL),
        accepted_steps,
        rejected_steps,
        best_alpha,
        notes,
        c.copy(),
        q,
        residual,
        history,
    )


def run_modal_direct(model: FixedBottomJaxResponse) -> MethodOutcome:
    method = "fixed_bottom_modal_direct_inversion"
    c, _, b, initial_projected, initial_physical = _initial_state(model)
    history = [
        _history_row(model, method, 0, initial_projected, initial_projected, initial_physical, initial_physical, 0.0)
    ]
    accepted = 0
    for iteration in range(1, MAX_ITER + 1):
        delta = -b
        c = c + delta
        _, b, projected, physical = _evaluate_c(model, c)
        accepted += 1
        history.append(
            _history_row(
                model,
                method,
                iteration,
                projected,
                initial_projected,
                physical,
                initial_physical,
                float(np.linalg.norm(delta)),
            )
        )
        ratio = projected / max(initial_projected, 1e-15)
        if ratio < TOL or not np.isfinite(ratio) or ratio > DIVERGENCE_RATIO:
            break
    return _finish_modal(
        model,
        method,
        c,
        initial_projected,
        initial_physical,
        history,
        accepted,
        0,
        "assumes J approximately equals I",
    )


def _run_scalar_alpha(model: FixedBottomJaxResponse, alpha: float, keep_history: bool) -> MethodOutcome:
    method = "best_scalar_oracle"
    c, _, b, initial_projected, initial_physical = _initial_state(model)
    history = [
        _history_row(model, method, 0, initial_projected, initial_projected, initial_physical, initial_physical, 0.0)
    ]
    accepted = 0
    for iteration in range(1, MAX_ITER + 1):
        delta = -alpha * b
        c = c + delta
        _, b, projected, physical = _evaluate_c(model, c)
        accepted += 1
        if keep_history:
            history.append(
                _history_row(
                    model,
                    method,
                    iteration,
                    projected,
                    initial_projected,
                    physical,
                    initial_physical,
                    float(np.linalg.norm(delta)),
                )
            )
        ratio = projected / max(initial_projected, 1e-15)
        if ratio < TOL or not np.isfinite(ratio) or ratio > DIVERGENCE_RATIO:
            break
    if not keep_history:
        history = [
            _history_row(model, method, 0, initial_projected, initial_projected, initial_physical, initial_physical, 0.0),
            _history_row(
                model,
                method,
                accepted,
                projected,
                initial_projected,
                physical,
                initial_physical,
                float(np.linalg.norm(c)),
            ),
        ]
    return _finish_modal(
        model,
        method,
        c,
        initial_projected,
        initial_physical,
        history,
        accepted,
        0,
        f"oracle grid-search scalar alpha={alpha:.8g}",
        alpha,
    )


def run_best_scalar(model: FixedBottomJaxResponse) -> MethodOutcome:
    best_alpha = None
    best_ratio = np.inf
    for alpha in SCALAR_ALPHAS:
        outcome = _run_scalar_alpha(model, float(alpha), keep_history=False)
        ratio = outcome.final_projected / max(outcome.initial_projected, 1e-15)
        if ratio < best_ratio:
            best_ratio = ratio
            best_alpha = float(alpha)
    assert best_alpha is not None
    best = _run_scalar_alpha(model, best_alpha, keep_history=True)
    best.notes = f"optimistic oracle scalar; best alpha={best_alpha:.8g}; grid size={len(SCALAR_ALPHAS)}"
    return best


def _safe_inverse_diagonal(diagonal: np.ndarray) -> np.ndarray:
    safe = np.asarray(diagonal, dtype=float).copy()
    small = np.abs(safe) < 1e-4
    safe[small] = np.where(safe[small] >= 0.0, 1e-4, -1e-4)
    return 1.0 / safe


def run_diagonal(model: FixedBottomJaxResponse) -> MethodOutcome:
    method = "diagonal_response_calibration"
    c, _, b, initial_projected, initial_physical = _initial_state(model)
    history = [
        _history_row(model, method, 0, initial_projected, initial_projected, initial_physical, initial_physical, 0.0)
    ]
    accepted = 0
    for iteration in range(1, MAX_ITER + 1):
        J = model.J(c)
        delta = -_safe_inverse_diagonal(np.diag(J)) * b
        c = c + delta
        _, b, projected, physical = _evaluate_c(model, c)
        accepted += 1
        history.append(
            _history_row(
                model,
                method,
                iteration,
                projected,
                initial_projected,
                physical,
                initial_physical,
                float(np.linalg.norm(delta)),
            )
        )
        ratio = projected / max(initial_projected, 1e-15)
        if ratio < TOL or not np.isfinite(ratio) or ratio > DIVERGENCE_RATIO:
            break
    return _finish_modal(
        model,
        method,
        c,
        initial_projected,
        initial_physical,
        history,
        accepted,
        0,
        "inverse diagonal of current AD Jacobian; diagonal entries safeguarded at 1e-4",
    )


def _gn_delta(J: np.ndarray, b: np.ndarray, damping: float) -> np.ndarray:
    lhs = J.T @ J + damping * np.eye(J.shape[1])
    rhs = -J.T @ b
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def run_full_gn(model: FixedBottomJaxResponse) -> MethodOutcome:
    method = "full_jacobian_gauss_newton"
    c, _, b, initial_projected, initial_physical = _initial_state(model)
    history = [
        _history_row(model, method, 0, initial_projected, initial_projected, initial_physical, initial_physical, 0.0)
    ]
    accepted = 0
    for iteration in range(1, MAX_ITER + 1):
        J = model.J(c)
        delta = _gn_delta(J, b, 1e-10)
        predicted = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        trial = c + delta
        _, b_trial, projected, physical = _evaluate_c(model, trial)
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b_trial, b_trial))
        eta = actual / predicted if predicted > 0 else float("-inf")
        c = trial
        b = b_trial
        accepted += 1
        history.append(
            _history_row(
                model,
                method,
                iteration,
                projected,
                initial_projected,
                physical,
                initial_physical,
                float(np.linalg.norm(delta)),
                1e-10,
                predicted,
                actual,
                eta,
                True,
            )
        )
        ratio = projected / max(initial_projected, 1e-15)
        if ratio < TOL or not np.isfinite(ratio) or ratio > DIVERGENCE_RATIO:
            break
    return _finish_modal(
        model,
        method,
        c,
        initial_projected,
        initial_physical,
        history,
        accepted,
        0,
        "full current AD Jacobian with fixed 1e-10 Gauss-Newton damping",
    )


def run_trust_region(model: FixedBottomJaxResponse) -> MethodOutcome:
    method = "full_jacobian_trust_region"
    c, _, b, initial_projected, initial_physical = _initial_state(model)
    lam = float(model.case.trust_initial_lambda)
    max_step = float(model.case.trust_max_step)
    history = [
        _history_row(
            model,
            method,
            0,
            initial_projected,
            initial_projected,
            initial_physical,
            initial_physical,
            0.0,
            lam,
        )
    ]
    accepted_steps = 0
    rejected_steps = 0
    for iteration in range(1, MAX_ITER + 1):
        J = model.J(c)
        delta = _gn_delta(J, b, lam)
        raw_norm = float(np.linalg.norm(delta))
        if np.isfinite(max_step) and raw_norm > max_step:
            delta *= max_step / max(raw_norm, 1e-15)
        predicted = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        trial = c + delta
        _, b_trial, projected_trial, physical_trial = _evaluate_c(model, trial)
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b_trial, b_trial))
        eta = actual / predicted if predicted > 0 and np.isfinite(predicted) else float("-inf")
        accepted = bool(predicted > 0.0 and eta > 0.05 and np.all(np.isfinite(trial)))
        if accepted:
            c = trial
            b = b_trial
            projected = projected_trial
            physical = physical_trial
            accepted_steps += 1
        else:
            _, _, projected, physical = _evaluate_c(model, c)
            rejected_steps += 1

        if accepted and eta > 0.75:
            lam *= 0.35
        elif accepted and eta < 0.25:
            lam *= 3.0
        elif not accepted:
            lam *= 8.0
        lam = float(np.clip(lam, 1e-12, 1e12))
        history.append(
            _history_row(
                model,
                method,
                iteration,
                projected,
                initial_projected,
                physical,
                initial_physical,
                float(np.linalg.norm(delta)),
                lam,
                predicted,
                actual,
                eta,
                accepted,
            )
        )
        ratio = projected / max(initial_projected, 1e-15)
        if ratio < TOL or not np.isfinite(ratio) or ratio > DIVERGENCE_RATIO:
            break
    return _finish_modal(
        model,
        method,
        c,
        initial_projected,
        initial_physical,
        history,
        accepted_steps,
        rejected_steps,
        f"predicted/actual LM control; max modal step={max_step}; final lambda={lam:.3e}",
    )


def run_all_methods(model: FixedBottomJaxResponse) -> List[MethodOutcome]:
    return [
        run_free_direct(model),
        run_fixed_projected_direct(model),
        run_modal_direct(model),
        run_best_scalar(model),
        run_diagonal(model),
        run_full_gn(model),
        run_trust_region(model),
    ]
