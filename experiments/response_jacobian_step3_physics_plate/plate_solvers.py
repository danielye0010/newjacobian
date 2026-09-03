"""Solver baselines for Step 3 physics plate responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from physics_response import C0, PhysicsPlateResponse
from plate_modes import K

MAX_ITER = 20
TOL = 1e-6
DIVERGENCE_RATIO = 1e3
ALPHAS = [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.75, 1.00, 1.25]


@dataclass
class SolverResult:
    case_name: str
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
    process_parameters: Optional[Dict[str, float]] = None

    def as_row(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _ratio(norm: float, initial: float) -> float:
    return float(norm / max(initial, 1e-15))


def _history(response: PhysicsPlateResponse, method: str, iteration: int, c: np.ndarray, initial: float, step_norm=0.0, accepted=None, lam=None, rho=None, alpha=None):
    norm = float(np.linalg.norm(response.b(c)))
    return {"case_name": response.case_name, "method_name": method, "iteration": iteration, "residual_norm": norm,
            "residual_ratio": _ratio(norm, initial), "c_norm": float(np.linalg.norm(c)), "step_norm": float(step_norm),
            "accepted": accepted, "lambda": lam, "rho": rho, "alpha": alpha}


def _finish(response, method, c, initial, hist, best, conv, div, **extra):
    final = float(np.linalg.norm(response.b(c)))
    return SolverResult(response.case_name, method, bool(conv), bool(div), max(0, len(hist) - 1), float(initial), final,
                        _ratio(final, initial), float(best), [float(x) for x in c], process_parameters=response.params.as_dict(), **extra)


def _run(response: PhysicsPlateResponse, method: str, update):
    c = C0.copy()
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history(response, method, 0, c, initial)]
    best = float(hist[-1]["residual_ratio"])
    conv = best < TOL
    div = False
    for it in range(1, MAX_ITER + 1):
        if conv or div:
            break
        c_next, meta = update(c)
        step = float(np.linalg.norm(c_next - c))
        c = c_next
        row = _history(response, method, it, c, initial, step, meta.get("accepted"), meta.get("lambda"), meta.get("rho"), meta.get("alpha"))
        hist.append(row)
        ratio = float(row["residual_ratio"])
        best = min(best, ratio)
        conv = ratio < TOL
        div = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))
    return _finish(response, method, c, initial, hist, best, conv, div), hist


def direct_modal_inversion(response):
    return _run(response, "direct_modal_inversion", lambda c: (c - response.b(c), {"accepted": True}))


def scalar_scale_factor(response, alpha: float, method_name: Optional[str] = None):
    name = method_name or f"scalar_alpha_{alpha:.2f}"
    return _run(response, name, lambda c: (c - alpha * response.b(c), {"accepted": True, "alpha": alpha}))


def oracle_scalar_scale_factor(response):
    best = None
    for alpha in np.linspace(0.01, 1.5, 150):
        result, hist = scalar_scale_factor(response, float(alpha), "oracle_scalar_scale_factor")
        if best is None or result.final_residual_ratio < best[0].final_residual_ratio:
            best = (result, hist, float(alpha))
    result, hist, alpha = best
    result.selected_alpha = alpha
    result.best_residual_ratio_seen = min(float(r["residual_ratio"]) for r in hist)
    return result, hist


def diagonal_modal_factor_initial(response):
    J0 = response.jacobian_fd(C0)
    diag = np.diag(J0).copy()
    safe = diag.copy()
    for i, value in enumerate(safe):
        if abs(value) < 1e-8:
            safe[i] = 1e-8 if value >= 0 else -1e-8
    D = np.diag(1.0 / safe)
    return _run(response, "diagonal_modal_factor_initial", lambda c: (c - D @ response.b(c), {"accepted": True}))


def full_jacobian_gn_no_trust(response):
    def update(c):
        J = response.jacobian_fd(c)
        delta = np.linalg.lstsq(J, -response.b(c), rcond=None)[0]
        return c + delta, {"accepted": True}
    return _run(response, "full_jacobian_gn_no_trust", update)


def full_jacobian_trust_region(response):
    method = "full_jacobian_trust_region"
    c = C0.copy()
    lam = 1e-3
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history(response, method, 0, c, initial, lam=lam)]
    best = float(hist[-1]["residual_ratio"])
    conv = best < TOL
    div = False
    accepted_steps = 0
    rejected_steps = 0
    for it in range(1, MAX_ITER + 1):
        if conv or div:
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
        else:
            rejected_steps += 1
        if rho > 0.75:
            lam *= 0.3
        elif rho < 0.25:
            lam *= 2.0
        lam = float(np.clip(lam, 1e-10, 1e10))
        row = _history(response, method, it, c, initial, float(np.linalg.norm(delta)) if accepted else 0.0, accepted, lam, float(rho))
        hist.append(row)
        ratio = float(row["residual_ratio"])
        best = min(best, ratio)
        conv = ratio < TOL
        div = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))
    return _finish(response, method, c, initial, hist, best, conv, div, accepted_steps=accepted_steps, rejected_steps=rejected_steps, final_lambda=lam), hist


def run_all_methods(response):
    results: List[SolverResult] = []
    histories: List[Dict[str, object]] = []
    for runner in [direct_modal_inversion]:
        r, h = runner(response); results.append(r); histories.extend(h)
    for alpha in ALPHAS:
        r, h = scalar_scale_factor(response, alpha); results.append(r); histories.extend(h)
    for runner in [oracle_scalar_scale_factor, diagonal_modal_factor_initial, full_jacobian_gn_no_trust, full_jacobian_trust_region]:
        r, h = runner(response); results.append(r); histories.extend(h)
    return results, histories
