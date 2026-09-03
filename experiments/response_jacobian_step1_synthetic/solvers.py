"""Solvers and baselines for the Step 1 synthetic diagnostic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from response_models import C_STAR, C0, ResponseModel

MAX_ITER = 20
TOL = 1e-6
DIVERGENCE_RATIO = 1e3


@dataclass
class SolverResult:
    case_name: str
    method_name: str
    converged: bool
    diverged: bool
    iterations_run: int
    final_residual_norm: float
    initial_residual_norm: float
    final_residual_ratio: float
    best_residual_ratio_seen: float
    final_c: List[float]
    error_to_c_star_norm: float
    accepted_steps: Optional[int] = None
    rejected_steps: Optional[int] = None
    final_lambda: Optional[float] = None
    selected_alpha: Optional[float] = None
    average_rho_accepted: Optional[float] = None
    diag_J0: Optional[List[float]] = None
    diag_inverse_factors: Optional[List[float]] = None

    def as_row(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _ratio(norm: float, initial: float) -> float:
    return float(norm / max(initial, 1e-15))


def _history_row(case_name: str, method_name: str, iteration: int, model: ResponseModel, c: np.ndarray,
                 initial_norm: float, step_norm: float = 0.0, accepted: Optional[bool] = None,
                 lam: Optional[float] = None, rho: Optional[float] = None, alpha: Optional[float] = None) -> Dict[str, object]:
    residual_norm = float(np.linalg.norm(model.b(c)))
    return {
        "case_name": case_name,
        "method_name": method_name,
        "iteration": iteration,
        "residual_norm": residual_norm,
        "residual_ratio": _ratio(residual_norm, initial_norm),
        "c_norm": float(np.linalg.norm(c)),
        "error_to_c_star_norm": float(np.linalg.norm(c - C_STAR)),
        "step_norm": float(step_norm),
        "accepted": accepted,
        "lambda": lam,
        "rho": rho,
        "alpha": alpha,
    }


def _finish(case_name: str, method_name: str, model: ResponseModel, c: np.ndarray, initial_norm: float,
            history: List[Dict[str, object]], best_ratio: float, converged: bool, diverged: bool,
            **extras: object) -> SolverResult:
    final_norm = float(np.linalg.norm(model.b(c)))
    return SolverResult(
        case_name=case_name,
        method_name=method_name,
        converged=bool(converged),
        diverged=bool(diverged),
        iterations_run=max(0, len(history) - 1),
        final_residual_norm=final_norm,
        initial_residual_norm=float(initial_norm),
        final_residual_ratio=_ratio(final_norm, initial_norm),
        best_residual_ratio_seen=float(best_ratio),
        final_c=[float(x) for x in c],
        error_to_c_star_norm=float(np.linalg.norm(c - C_STAR)),
        **extras,
    )


def run_iterative_method(model: ResponseModel, method_name: str, update: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, object]]],
                         max_iter: int = MAX_ITER) -> Tuple[SolverResult, List[Dict[str, object]]]:
    c = C0.copy()
    initial_norm = float(np.linalg.norm(model.b(c)))
    history = [_history_row(model.name, method_name, 0, model, c, initial_norm)]
    best_ratio = history[-1]["residual_ratio"]
    converged = best_ratio < TOL
    diverged = False

    for iteration in range(1, max_iter + 1):
        if converged or diverged:
            break
        c_next, meta = update(c)
        step_norm = float(np.linalg.norm(c_next - c))
        c = c_next
        row = _history_row(
            model.name, method_name, iteration, model, c, initial_norm,
            step_norm=step_norm, accepted=meta.get("accepted"), lam=meta.get("lambda"),
            rho=meta.get("rho"), alpha=meta.get("alpha"),
        )
        history.append(row)
        ratio = float(row["residual_ratio"])
        best_ratio = min(best_ratio, ratio)
        converged = ratio < TOL
        diverged = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))

    return _finish(model.name, method_name, model, c, initial_norm, history, best_ratio, converged, diverged), history


def direct_modal_inversion(model: ResponseModel) -> Tuple[SolverResult, List[Dict[str, object]]]:
    return run_iterative_method(model, "direct_modal_inversion", lambda c: (c - model.b(c), {"accepted": True}))


def scalar_scale_factor(model: ResponseModel, alpha: float, method_name: Optional[str] = None) -> Tuple[SolverResult, List[Dict[str, object]]]:
    name = method_name or f"scalar_alpha_{alpha:.2f}"
    return run_iterative_method(model, name, lambda c: (c - alpha * model.b(c), {"accepted": True, "alpha": alpha}))


def oracle_scalar_scale_factor(model: ResponseModel) -> Tuple[SolverResult, List[Dict[str, object]]]:
    best: Optional[Tuple[SolverResult, List[Dict[str, object]]]] = None
    for alpha in np.linspace(0.01, 1.5, 150):
        result, history = scalar_scale_factor(model, float(alpha), method_name="oracle_scalar_scale_factor")
        if best is None or result.final_residual_ratio < best[0].final_residual_ratio:
            best = (result, history)
    assert best is not None
    result, history = best
    result.selected_alpha = float(history[1]["alpha"] if len(history) > 1 else 0.0)
    result.best_residual_ratio_seen = min(float(r["residual_ratio"]) for r in history)
    return result, history


def diagonal_modal_factor_initial(model: ResponseModel) -> Tuple[SolverResult, List[Dict[str, object]]]:
    J0 = model.jacobian(C0)
    diag = np.diag(J0).astype(float)
    safe_diag = diag.copy()
    for i, value in enumerate(safe_diag):
        if abs(value) < 1e-8:
            safe_diag[i] = 1e-8 if value >= 0 else -1e-8
    inv = 1.0 / safe_diag
    D = np.diag(inv)
    result, history = run_iterative_method(
        model, "diagonal_modal_factor_initial", lambda c: (c - D @ model.b(c), {"accepted": True})
    )
    result.diag_J0 = [float(x) for x in diag]
    result.diag_inverse_factors = [float(x) for x in inv]
    return result, history


def full_jacobian_gn_no_trust(model: ResponseModel) -> Tuple[SolverResult, List[Dict[str, object]]]:
    def update(c: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        delta = np.linalg.lstsq(model.jacobian(c), -model.b(c), rcond=None)[0]
        return c + delta, {"accepted": True}
    return run_iterative_method(model, "full_jacobian_gn_no_trust", update)


def full_jacobian_trust_region(model: ResponseModel) -> Tuple[SolverResult, List[Dict[str, object]]]:
    c = C0.copy()
    lam = 1e-3
    initial_norm = float(np.linalg.norm(model.b(c)))
    history = [_history_row(model.name, "full_jacobian_trust_region", 0, model, c, initial_norm, lam=lam)]
    best_ratio = float(history[-1]["residual_ratio"])
    accepted_steps = 0
    rejected_steps = 0
    accepted_rhos: List[float] = []
    converged = best_ratio < TOL
    diverged = False

    for iteration in range(1, MAX_ITER + 1):
        if converged or diverged:
            break
        b = model.b(c)
        J = model.jacobian(c)
        lhs = J.T @ J + lam * np.eye(len(c))
        rhs = -J.T @ b
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        pred = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        c_trial = c + delta
        b_trial = model.b(c_trial)
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b_trial, b_trial))
        rho = actual / pred if pred > 0 and np.isfinite(pred) else float("-inf")
        accepted = bool(pred > 0 and rho > 0 and np.all(np.isfinite(c_trial)))
        if accepted:
            c = c_trial
            accepted_steps += 1
            accepted_rhos.append(float(rho))
        else:
            rejected_steps += 1
        if rho > 0.75:
            lam *= 0.3
        elif rho < 0.25:
            lam *= 2.0
        lam = float(np.clip(lam, 1e-10, 1e10))
        row = _history_row(model.name, "full_jacobian_trust_region", iteration, model, c, initial_norm,
                           step_norm=float(np.linalg.norm(delta)) if accepted else 0.0,
                           accepted=accepted, lam=lam, rho=float(rho))
        history.append(row)
        ratio = float(row["residual_ratio"])
        best_ratio = min(best_ratio, ratio)
        converged = ratio < TOL
        diverged = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))

    result = _finish(
        model.name, "full_jacobian_trust_region", model, c, initial_norm, history, best_ratio,
        converged, diverged, accepted_steps=accepted_steps, rejected_steps=rejected_steps,
        final_lambda=lam, average_rho_accepted=(float(np.mean(accepted_rhos)) if accepted_rhos else None),
    )
    return result, history


def run_all_methods(model: ResponseModel) -> Tuple[List[SolverResult], List[Dict[str, object]]]:
    runners = [direct_modal_inversion]
    results: List[SolverResult] = []
    histories: List[Dict[str, object]] = []
    for runner in runners:
        result, hist = runner(model)
        results.append(result)
        histories.extend(hist)
    for alpha in [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.75, 1.00, 1.25]:
        result, hist = scalar_scale_factor(model, alpha)
        results.append(result)
        histories.extend(hist)
    for runner in [oracle_scalar_scale_factor, diagonal_modal_factor_initial, full_jacobian_gn_no_trust, full_jacobian_trust_region]:
        result, hist = runner(model)
        results.append(result)
        histories.extend(hist)
    return results, histories
