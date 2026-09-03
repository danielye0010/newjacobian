"""Compensation solvers for Step 4 FEM-like plate response."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from fem_plate_modes import K
from thermal_response import C0, ThermalFEMResponse

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

    def as_row(self):
        return self.__dict__.copy()


def _ratio(norm, initial): return float(norm / max(initial, 1e-15))


def _hist(resp, method, it, c, initial, step=0.0, accepted=None, lam=None, rho=None, alpha=None):
    norm = float(np.linalg.norm(resp.b(c)))
    return {"case_name": resp.case_name, "method_name": method, "iteration": it, "residual_norm": norm, "residual_ratio": _ratio(norm, initial), "c_norm": float(np.linalg.norm(c)), "step_norm": float(step), "accepted": accepted, "lambda": lam, "rho": rho, "alpha": alpha}


def _finish(resp, method, c, initial, hist, best, conv, div, **extra):
    final = float(np.linalg.norm(resp.b(c)))
    return SolverResult(resp.case_name, method, bool(conv), bool(div), max(0, len(hist) - 1), float(initial), final, _ratio(final, initial), float(best), [float(x) for x in c], process_parameters=resp.params.as_dict(), **extra)


def _run(resp, method, update):
    c = C0.copy(); initial = float(np.linalg.norm(resp.b(c)))
    hist = [_hist(resp, method, 0, c, initial)]; best = float(hist[-1]["residual_ratio"]); conv = best < TOL; div = False
    for it in range(1, MAX_ITER + 1):
        if conv or div: break
        c_next, meta = update(c); step = float(np.linalg.norm(c_next - c)); c = c_next
        row = _hist(resp, method, it, c, initial, step, meta.get("accepted"), meta.get("lambda"), meta.get("rho"), meta.get("alpha")); hist.append(row)
        r = float(row["residual_ratio"]); best = min(best, r); conv = r < TOL; div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))
    return _finish(resp, method, c, initial, hist, best, conv, div), hist


def direct_modal_inversion(resp): return _run(resp, "direct_modal_inversion", lambda c: (c - resp.b(c), {"accepted": True}))

def scalar_scale_factor(resp, alpha, method_name=None): return _run(resp, method_name or f"scalar_alpha_{alpha:.2f}", lambda c: (c - alpha * resp.b(c), {"accepted": True, "alpha": alpha}))

def oracle_scalar_scale_factor(resp):
    best = None
    for alpha in np.linspace(0.01, 1.5, 150):
        r, h = scalar_scale_factor(resp, float(alpha), "oracle_scalar_scale_factor")
        if best is None or r.final_residual_ratio < best[0].final_residual_ratio: best = (r, h, float(alpha))
    r, h, alpha = best; r.selected_alpha = alpha; r.best_residual_ratio_seen = min(float(x["residual_ratio"]) for x in h); return r, h

def diagonal_modal_factor_initial(resp):
    J0 = resp.jacobian_fd(C0); diag = np.diag(J0).copy(); safe = diag.copy()
    for i, v in enumerate(safe):
        if abs(v) < 1e-8: safe[i] = 1e-8 if v >= 0 else -1e-8
    D = np.diag(1.0 / safe)
    return _run(resp, "diagonal_modal_factor_initial", lambda c: (c - D @ resp.b(c), {"accepted": True}))

def full_jacobian_gn_no_trust(resp):
    def update(c):
        J = resp.jacobian_fd(c); delta = np.linalg.lstsq(J, -resp.b(c), rcond=None)[0]
        return c + delta, {"accepted": True}
    return _run(resp, "full_jacobian_gn_no_trust", update)

def full_jacobian_trust_region(resp):
    method = "full_jacobian_trust_region"; c = C0.copy(); lam = 1e-3; initial = float(np.linalg.norm(resp.b(c)))
    hist = [_hist(resp, method, 0, c, initial, lam=lam)]; best = float(hist[-1]["residual_ratio"]); conv = best < TOL; div = False; acc = 0; rej = 0
    for it in range(1, MAX_ITER + 1):
        if conv or div: break
        b = resp.b(c); J = resp.jacobian_fd(c); lhs = J.T @ J + lam * np.eye(K); rhs = -J.T @ b
        try: delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError: delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        pred = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        trial = c + delta; bt = resp.b(trial); actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(bt, bt))
        rho = actual / pred if pred > 0 and np.isfinite(pred) else float("-inf"); accepted = bool(pred > 0 and rho > 0 and np.all(np.isfinite(trial)))
        if accepted: c = trial; acc += 1
        else: rej += 1
        if rho > 0.75: lam *= 0.3
        elif rho < 0.25: lam *= 2.0
        lam = float(np.clip(lam, 1e-10, 1e10))
        row = _hist(resp, method, it, c, initial, float(np.linalg.norm(delta)) if accepted else 0.0, accepted, lam, float(rho)); hist.append(row)
        r = float(row["residual_ratio"]); best = min(best, r); conv = r < TOL; div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))
    return _finish(resp, method, c, initial, hist, best, conv, div, accepted_steps=acc, rejected_steps=rej, final_lambda=lam), hist

def run_all_methods(resp):
    results = []; histories = []
    for fn in [direct_modal_inversion]:
        r, h = fn(resp); results.append(r); histories.extend(h)
    for a in ALPHAS:
        r, h = scalar_scale_factor(resp, a); results.append(r); histories.extend(h)
    for fn in [oracle_scalar_scale_factor, diagonal_modal_factor_initial, full_jacobian_gn_no_trust, full_jacobian_trust_region]:
        r, h = fn(resp); results.append(r); histories.extend(h)
    return results, histories
