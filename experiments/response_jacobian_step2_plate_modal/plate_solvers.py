"""Solver baselines for Step 2 plate modal responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from plate_modes import K
from plate_responses import C0, PlateResponse

MAX_ITER = 20
TOL = 1e-6
DIVERGENCE_RATIO = 1e3
ALPHAS = [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.75, 1.00, 1.25]


@dataclass
class SolverResult:
    case_name: str
    response_type: str
    method_name: str
    converged: bool
    diverged: bool
    iterations_run: int
    initial_residual_norm: float
    final_residual_norm: float
    final_residual_ratio: float
    best_residual_ratio_seen: float
    final_c: List[float]
    selected_alpha: Optional[float] = None
    accepted_steps: Optional[int] = None
    rejected_steps: Optional[int] = None
    final_lambda: Optional[float] = None
    average_rho_accepted: Optional[float] = None
    error_to_best_solution_norm: Optional[float] = None
    diag_J0: Optional[List[float]] = None
    diag_inverse_factors: Optional[List[float]] = None

    def as_row(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _ratio(norm: float, initial: float) -> float:
    return float(norm / max(initial, 1e-15))


def _history_row(response: PlateResponse, method_name: str, iteration: int, c: np.ndarray, initial_norm: float,
                 step_norm: float = 0.0, accepted: Optional[bool] = None, lam: Optional[float] = None,
                 rho: Optional[float] = None, alpha: Optional[float] = None) -> Dict[str, object]:
    residual_norm = float(np.linalg.norm(response.b(c)))
    return {
        "case_name": response.case_name,
        "method_name": method_name,
        "iteration": iteration,
        "residual_norm": residual_norm,
        "residual_ratio": _ratio(residual_norm, initial_norm),
        "c_norm": float(np.linalg.norm(c)),
        "step_norm": float(step_norm),
        "accepted": accepted,
        "lambda": lam,
        "rho": rho,
        "alpha": alpha,
    }


def _finish(response: PlateResponse, method_name: str, c: np.ndarray, initial_norm: float,
            history: List[Dict[str, object]], best_ratio: float, converged: bool, diverged: bool,
            **extras: object) -> SolverResult:
    final_norm = float(np.linalg.norm(response.b(c)))
    return SolverResult(
        case_name=response.case_name,
        response_type=response.response_type,
        method_name=method_name,
        converged=bool(converged),
        diverged=bool(diverged),
        iterations_run=max(0, len(history) - 1),
        initial_residual_norm=float(initial_norm),
        final_residual_norm=final_norm,
        final_residual_ratio=_ratio(final_norm, initial_norm),
        best_residual_ratio_seen=float(best_ratio),
        final_c=[float(x) for x in c],
        **extras,
    )


def _run_update(response: PlateResponse, method_name: str, update) -> Tuple[SolverResult, List[Dict[str, object]]]:
    c = C0.copy()
    initial_norm = float(np.linalg.norm(response.b(c)))
    history = [_history_row(response, method_name, 0, c, initial_norm)]
    best_ratio = float(history[-1]["residual_ratio"])
    converged = best_ratio < TOL
    diverged = False
    for iteration in range(1, MAX_ITER + 1):
        if converged or diverged:
            break
        c_next, meta = update(c)
        step_norm = float(np.linalg.norm(c_next - c))
        c = c_next
        row = _history_row(response, method_name, iteration, c, initial_norm, step_norm=step_norm,
                           accepted=meta.get("accepted"), lam=meta.get("lambda"), rho=meta.get("rho"), alpha=meta.get("alpha"))
        history.append(row)
        ratio = float(row["residual_ratio"])
        best_ratio = min(best_ratio, ratio)
        converged = ratio < TOL
        diverged = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))
    return _finish(response, method_name, c, initial_norm, history, best_ratio, converged, diverged), history


def direct_modal_inversion(response: PlateResponse):
    return _run_update(response, "direct_modal_inversion", lambda c: (c - response.b(c), {"accepted": True}))


def scalar_scale_factor(response: PlateResponse, alpha: float, method_name: Optional[str] = None):
    name = method_name or f"scalar_alpha_{alpha:.2f}"
    return _run_update(response, name, lambda c: (c - alpha * response.b(c), {"accepted": True, "alpha": alpha}))


def oracle_scalar_scale_factor(response: PlateResponse):
    best = None
    for alpha in np.linspace(0.01, 1.5, 150):
        result, history = scalar_scale_factor(response, float(alpha), method_name="oracle_scalar_scale_factor")
        if best is None or result.final_residual_ratio < best[0].final_residual_ratio:
            best = (result, history, float(alpha))
    result, history, alpha = best
    result.selected_alpha = alpha
    result.best_residual_ratio_seen = min(float(r["residual_ratio"]) for r in history)
    return result, history


def diagonal_modal_factor_initial(response: PlateResponse):
    J0 = response.jacobian_fd(C0)
    diag = np.diag(J0).astype(float)
    safe_diag = diag.copy()
    for i, value in enumerate(safe_diag):
        if abs(value) < 1e-8:
            safe_diag[i] = 1e-8 if value >= 0 else -1e-8
    inv = 1.0 / safe_diag
    D = np.diag(inv)
    result, history = _run_update(response, "diagonal_modal_factor_initial", lambda c: (c - D @ response.b(c), {"accepted": True}))
    result.diag_J0 = [float(x) for x in diag]
    result.diag_inverse_factors = [float(x) for x in inv]
    return result, history


def full_jacobian_gn_no_trust(response: PlateResponse):
    def update(c: np.ndarray):
        J = response.jacobian_fd(c)
        delta = np.linalg.lstsq(J, -response.b(c), rcond=None)[0]
        return c + delta, {"accepted": True}
    return _run_update(response, "full_jacobian_gn_no_trust", update)


def full_jacobian_trust_region(response: PlateResponse):
    method_name = "full_jacobian_trust_region"
    c = C0.copy()
    lam = 1e-3
    initial_norm = float(np.linalg.norm(response.b(c)))
    history = [_history_row(response, method_name, 0, c, initial_norm, lam=lam)]
    best_ratio = float(history[-1]["residual_ratio"])
    accepted_steps = 0
    rejected_steps = 0
    accepted_rhos: List[float] = []
    converged = best_ratio < TOL
    diverged = False

    for iteration in range(1, MAX_ITER + 1):
        if converged or diverged:
            break
        b = response.b(c)
        J = response.jacobian_fd(c)
        lhs = J.T @ J + lam * np.eye(K)
        rhs = -J.T @ b
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        pred = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        trial = c + delta
        b_trial = response.b(trial)
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b_trial, b_trial))
        rho = actual / pred if pred > 0 and np.isfinite(pred) else float("-inf")
        accepted = bool(pred > 0 and rho > 0 and np.all(np.isfinite(trial)))
        if accepted:
            c = trial
            accepted_steps += 1
            accepted_rhos.append(float(rho))
        else:
            rejected_steps += 1
        if rho > 0.75:
            lam *= 0.3
        elif rho < 0.25:
            lam *= 2.0
        lam = float(np.clip(lam, 1e-10, 1e10))
        row = _history_row(response, method_name, iteration, c, initial_norm,
                           step_norm=float(np.linalg.norm(delta)) if accepted else 0.0,
                           accepted=accepted, lam=lam, rho=float(rho))
        history.append(row)
        ratio = float(row["residual_ratio"])
        best_ratio = min(best_ratio, ratio)
        converged = ratio < TOL
        diverged = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))

    result = _finish(response, method_name, c, initial_norm, history, best_ratio, converged, diverged,
                     accepted_steps=accepted_steps, rejected_steps=rejected_steps, final_lambda=lam,
                     average_rho_accepted=(float(np.mean(accepted_rhos)) if accepted_rhos else None))
    return result, history


def run_all_methods(response: PlateResponse) -> Tuple[List[SolverResult], List[Dict[str, object]]]:
    results: List[SolverResult] = []
    histories: List[Dict[str, object]] = []
    runners = [direct_modal_inversion]
    for runner in runners:
        result, history = runner(response)
        results.append(result)
        histories.extend(history)
    for alpha in ALPHAS:
        result, history = scalar_scale_factor(response, alpha)
        results.append(result)
        histories.extend(history)
    for runner in [oracle_scalar_scale_factor, diagonal_modal_factor_initial, full_jacobian_gn_no_trust, full_jacobian_trust_region]:
        result, history = runner(response)
        results.append(result)
        histories.extend(history)

    best_c = np.asarray(min(results, key=lambda r: r.final_residual_ratio).final_c, dtype=float)
    for result in results:
        result.error_to_best_solution_norm = float(np.linalg.norm(np.asarray(result.final_c, dtype=float) - best_c))
    return results, histories
