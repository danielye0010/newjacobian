"""Method-objective closure experiments for response inversion.

This runner is intentionally separate from the manuscript and from the prior
final rebuild.  It tests whether the inverse problem should optimize the
reduced modal residual, the all-DOF physical residual, or the surface residual.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_response_inversion_final_experiment_rebuild as prev  # noqa: E402
from fixed_bottom_core import compute_free_modes  # noqa: E402
from run_admissible_inherent_strain_fem_benchmark import (  # noqa: E402
    BasisData,
    InherentStrainFEM,
    build_cases,
    constrained_basis,
    project_c_allowance,
    solve_lm_delta,
)


RESULT_DIR = ROOT / "results" / "response_inversion_method_closure"
MODE_COUNT = 6
PARENT_MODE_COUNT = 18
MAX_ITER = 8
TOL = 1e-3
EPS = 1e-15
FD_STEP_R = 3e-5
FD_STEP_U = 1e-5

OBJECTIVES = ("reduced", "volume_physical", "surface_physical")
GLOBALIZATIONS = ("GN", "LM_trust")
OBJECTIVE_CASES = [
    ("A_FDM_comb_coupon", "bottom_contact", "slotted/FDM comb coupon", (0.5, 1.0, 1.5)),
    ("B_L_bracket_datum", "bottom_only", "L-bracket fixed base", (1.0, 1.5)),
    ("C_wall_allowance", "bottom_plus_allowance", "thin wall allowance", (1.0, 1.5)),
]
MANUFACTURING_CASES = [
    ("A_FDM_comb_coupon", "bottom_contact", "slotted coupon fixed contact"),
    ("B_L_bracket_datum", "bottom_only", "L-bracket fixed base"),
    ("B_L_bracket_datum", "bottom_plus_datum", "L-bracket fixed base plus datum"),
    ("C_wall_allowance", "bottom_plus_allowance", "thin wall allowance"),
]
ROBUSTNESS_CASES = [
    ("A_FDM_comb_coupon", "bottom_contact", "slotted/FDM comb coupon"),
    ("B_L_bracket_datum", "bottom_only", "L-bracket fixed base"),
    ("C_wall_allowance", "bottom_plus_allowance", "thin wall allowance"),
]


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(columns))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                elif isinstance(val, np.bool_):
                    val = bool(val)
                elif isinstance(val, (list, tuple, dict, np.ndarray)):
                    val = json.dumps(val if isinstance(val, dict) else np.asarray(val).tolist())
                clean[col] = val
            writer.writerow(clean)


def fmt(x: object) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return str(x)
    if v == 0.0:
        return "0.000e+00"
    if abs(v) >= 1e3 or abs(v) < 1e-2:
        return f"{v:.3e}"
    return f"{v:.3f}"


def md_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def scale_case(case, gamma: float):
    return replace(
        case,
        alpha_x=case.alpha_x * gamma,
        alpha_y=case.alpha_y * gamma,
        alpha_z=case.alpha_z * gamma,
        beta_z=case.beta_z * gamma,
        beta_edge=case.beta_edge * gamma,
        beta_finger=case.beta_finger * gamma,
        beta_side=case.beta_side * gamma,
        shear_xy=case.shear_xy * gamma,
    )


def build_model(case_name: str, constraint_set: str, gamma: float = 1.0, mode_count: int = MODE_COUNT) -> InherentStrainFEM:
    cases = {case.name: case for case in build_cases()}
    case = scale_case(cases[case_name], gamma)
    basis = constrained_basis(case.geometry, case.constraint_sets[constraint_set], mode_count, f"{case_name}_{constraint_set}_closure_{mode_count}")
    return InherentStrainFEM(case, basis, constraint_set, min(mode_count, basis.usable_modes))


def weighted_norm(model: InherentStrainFEM, residual: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float).reshape(-1)
    return float(np.sqrt(max(np.dot(model.weights_dof_np * r, r), 0.0)))


def surface_dofs(model: InherentStrainFEM) -> np.ndarray:
    return np.repeat(model.surface_mask_np, 3)


def surface_norm(model: InherentStrainFEM, residual: np.ndarray) -> float:
    mask = surface_dofs(model)
    n = max(int(np.sum(model.surface_mask_np)), 1)
    return float(np.linalg.norm(np.asarray(residual, dtype=float)[mask]) / math.sqrt(n))


def violation_metrics(model: InherentStrainFEM, q: np.ndarray) -> Dict[str, float]:
    return prev.violation_metrics(model, q)


def objective_norms(model: InherentStrainFEM, D: np.ndarray) -> Dict[str, float]:
    b = model.project_residual(D)
    return {
        "reduced_residual": float(np.linalg.norm(b)),
        "physical_residual": weighted_norm(model, D),
        "surface_residual": surface_norm(model, D),
    }


class CountedForward:
    def __init__(self, model: InherentStrainFEM, name: str):
        self.model = model
        self.name = name
        self.forward = prev.ForwardModel(model, name)
        self.residual_calls = 0
        self.jacobian_builds = 0
        self.fd_internal_calls = 0

    def reset_counts(self) -> None:
        self.residual_calls = 0
        self.jacobian_builds = 0
        self.fd_internal_calls = 0

    def residual_c(self, c: np.ndarray) -> np.ndarray:
        self.residual_calls += 1
        return self.forward.residual_c(c)

    def residual_q(self, q: np.ndarray) -> np.ndarray:
        self.residual_calls += 1
        return self.forward.residual_q(q)

    def G(self, c: np.ndarray) -> np.ndarray:
        self.jacobian_builds += 1
        if self.name == "reference_domain":
            return self.model.A(c)
        h = FD_STEP_U
        r0 = self.residual_c(c)
        G = np.zeros((r0.size, self.model.mode_count), dtype=float)
        for j in range(self.model.mode_count):
            step = np.zeros(self.model.mode_count)
            step[j] = h
            G[:, j] = (self.residual_c(c + step) - self.residual_c(c - step)) / (2.0 * h)
            self.fd_internal_calls += 2
        return G

    def J(self, c: np.ndarray) -> np.ndarray:
        if self.name == "reference_domain":
            self.jacobian_builds += 1
            return self.model.J(c)
        G = self.G(c)
        return self.model.psi_np.T @ (self.model.weights_dof_np[:, None] * G)


def objective_pair(model: InherentStrainFEM, fwd: CountedForward, c: np.ndarray, objective: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D = fwd.residual_c(c)
    G = fwd.G(c)
    if objective == "reduced":
        r = model.project_residual(D)
        H = model.psi_np.T @ (model.weights_dof_np[:, None] * G)
    elif objective == "volume_physical":
        sw = np.sqrt(model.weights_dof_np)
        r = sw * D
        H = sw[:, None] * G
    elif objective == "surface_physical":
        mask = surface_dofs(model)
        scale = 1.0 / math.sqrt(max(int(np.sum(model.surface_mask_np)), 1))
        r = D[mask] * scale
        H = G[mask, :] * scale
    else:
        raise ValueError(objective)
    return r, H, D, G


def gram_error(model: InherentStrainFEM) -> float:
    return float(np.linalg.norm(model.psi_np.T @ (model.weights_dof_np[:, None] * model.psi_np) - np.eye(model.mode_count)))


def residual_components(model: InherentStrainFEM, D: np.ndarray) -> Tuple[float, float, float]:
    coeff = model.project_residual(D)
    D_parallel = model.psi_np @ coeff
    D_perp = np.asarray(D, dtype=float) - D_parallel
    return weighted_norm(model, D_parallel), weighted_norm(model, D_perp), weighted_norm(model, D)


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def step_lm(H: np.ndarray, r: np.ndarray, lam: float, trust: float = np.inf) -> np.ndarray:
    z = solve_lm_delta(H, r, lam)
    n = float(np.linalg.norm(z))
    if np.isfinite(trust) and n > trust:
        z *= trust / max(n, EPS)
    return z


def objective_value(r: np.ndarray) -> float:
    return 0.5 * float(np.dot(r, r))


def run_identity_baseline(model: InherentStrainFEM, fwd: CountedForward, case_label: str, gamma: float) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    c = np.zeros(model.mode_count)
    D0 = fwd.residual_c(c)
    b0 = model.project_residual(D0)
    init = objective_norms(model, D0)
    histories: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    accepted = rejected = 0
    for iteration in range(1, MAX_ITER + 1):
        delta = -model.project_residual(fwd.residual_c(c))
        trial_raw = c + delta
        trial = project_c_allowance(trial_raw, model)
        D = fwd.residual_c(trial)
        accepted += 1
        c = trial
        par, perp, total = residual_components(model, D)
        histories.append(component_row(model, case_label, gamma, "identity_response", "baseline", "identity", iteration, D, par, perp, total))
        if np.linalg.norm(model.project_residual(D)) / max(np.linalg.norm(b0), EPS) < TOL:
            break
    Df = fwd.residual_c(c)
    final = objective_norms(model, Df)
    row = base_result_row(model, fwd, case_label, gamma, "identity_response", "baseline", "identity", init, final, c, t0, accepted, rejected)
    row["own_objective_converged"] = bool(final["reduced_residual"] / max(init["reduced_residual"], EPS) < TOL)
    return row, histories


def component_row(model: InherentStrainFEM, case_label: str, gamma: float, objective: str, globalization: str, method: str, iteration: int, D: np.ndarray, par: float, perp: float, total: float) -> Dict[str, object]:
    return {
        "case": model.case.name,
        "case_label": case_label,
        "constraint_set": model.constraint_set,
        "gamma": gamma,
        "objective": objective,
        "globalization": globalization,
        "method": method,
        "iteration": iteration,
        "gram_error": gram_error(model),
        "D_parallel_M_norm": par,
        "D_perp_M_norm": perp,
        "weighted_physical_residual": total,
        "surface_residual": surface_norm(model, D),
    }


def base_result_row(
    model: InherentStrainFEM,
    fwd: CountedForward,
    case_label: str,
    gamma: float,
    objective: str,
    globalization: str,
    method: str,
    init: Dict[str, float],
    final: Dict[str, float],
    c: np.ndarray,
    t0: float,
    accepted: int,
    rejected: int,
) -> Dict[str, object]:
    q = model.q_from_c(c)
    row = {
        "case": model.case.name,
        "case_label": case_label,
        "constraint_set": model.constraint_set,
        "gamma": gamma,
        "forward_model": fwd.name,
        "objective": objective,
        "globalization": globalization,
        "method": method,
        "basis_size": model.mode_count,
        "initial_reduced_residual": init["reduced_residual"],
        "final_reduced_residual": final["reduced_residual"],
        "reduced_residual_ratio": final["reduced_residual"] / max(init["reduced_residual"], EPS),
        "initial_physical_residual": init["physical_residual"],
        "final_physical_residual": final["physical_residual"],
        "physical_residual_ratio": final["physical_residual"] / max(init["physical_residual"], EPS),
        "initial_surface_residual": init["surface_residual"],
        "final_surface_residual": final["surface_residual"],
        "surface_residual_ratio": final["surface_residual"] / max(init["surface_residual"], EPS),
        "physical_improved": bool(final["physical_residual"] < init["physical_residual"]),
        "surface_improved": bool(final["surface_residual"] < init["surface_residual"]),
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "actual_residual_calls": fwd.residual_calls,
        "jacobian_construction_count": fwd.jacobian_builds,
        "fd_internal_residual_calls": fwd.fd_internal_calls,
        "runtime_s": time.perf_counter() - t0,
    }
    row.update(violation_metrics(model, q))
    return row


def run_objective_solver(
    model: InherentStrainFEM,
    fwd: CountedForward,
    case_label: str,
    gamma: float,
    objective: str,
    globalization: str,
    method: str | None = None,
    constrained_projection: bool = True,
    max_iter: int = MAX_ITER,
) -> Tuple[Dict[str, object], List[Dict[str, object]], np.ndarray]:
    method = method or f"{objective}_{globalization}"
    fwd.reset_counts()
    c = np.zeros(model.mode_count)
    D0 = fwd.residual_c(c)
    init = objective_norms(model, D0)
    r0, _, _, _ = objective_pair(model, fwd, c, objective)
    own0 = float(np.linalg.norm(r0))
    lam = model.case.trust_initial_lambda
    accepted = rejected = 0
    histories: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    converged = False
    for iteration in range(1, max_iter + 1):
        r, H, D, _ = objective_pair(model, fwd, c, objective)
        if np.linalg.norm(r) / max(own0, EPS) < TOL:
            converged = True
            break
        if globalization == "GN":
            delta = step_lm(H, r, 1e-10, model.case.trust_max_step)
        elif globalization == "LM_trust":
            delta = step_lm(H, r, lam, model.case.trust_max_step)
        else:
            raise ValueError(globalization)
        trial_raw = c + delta
        trial = project_c_allowance(trial_raw, model) if constrained_projection else trial_raw
        r_trial, _, D_trial, _ = objective_pair(model, fwd, trial, objective)
        pred = objective_value(r) - objective_value(r + H @ (trial - c))
        actual = objective_value(r) - objective_value(r_trial)
        feasible = violation_metrics(model, model.q_from_c(trial))["constraint_violation_max"] < 1e-8
        if globalization == "LM_trust":
            rho = actual / pred if pred > 0.0 else -np.inf
            ok = bool(feasible and pred > 0.0 and rho > 0.05)
            if ok:
                c = trial
                accepted += 1
                lam *= 0.5 if rho > 0.75 else 1.0
                D_hist = D_trial
            else:
                rejected += 1
                lam *= 5.0
                D_hist = D
        else:
            c = trial
            accepted += 1
            D_hist = D_trial
        par, perp, total = residual_components(model, D_hist)
        histories.append(component_row(model, case_label, gamma, objective, globalization, method, iteration, D_hist, par, perp, total))
    Df = fwd.residual_c(c)
    final = objective_norms(model, Df)
    row = base_result_row(model, fwd, case_label, gamma, objective, globalization, method, init, final, c, t0, accepted, rejected)
    rf, _, _, _ = objective_pair(model, fwd, c, objective)
    row["own_objective_converged"] = bool(converged or np.linalg.norm(rf) / max(own0, EPS) < TOL)
    row["iterations"] = accepted + rejected
    return row, histories, c


def objective_comparison() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[Tuple[str, str, float], np.ndarray]]:
    rows: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []
    align: List[Dict[str, object]] = []
    finals: Dict[Tuple[str, str, float], np.ndarray] = {}
    for case_name, cset, label, gammas in OBJECTIVE_CASES:
        for gamma in gammas:
            model = build_model(case_name, cset, gamma)
            fwd = CountedForward(model, "reference_domain")
            identity, hist = run_identity_baseline(model, fwd, label, gamma)
            identity["iterations"] = identity["accepted_steps"] + identity["rejected_steps"]
            rows.append(identity)
            histories.extend(hist)
            align.append(alignment_row(model, CountedForward(model, "reference_domain"), label, gamma))
            for objective in OBJECTIVES:
                for glob in GLOBALIZATIONS:
                    row, hist, c_final = run_objective_solver(model, CountedForward(model, "reference_domain"), label, gamma, objective, glob)
                    rows.append(row)
                    histories.extend(hist)
                    finals[(case_name, objective, gamma)] = c_final
    return rows, histories, align, finals


def alignment_row(model: InherentStrainFEM, fwd: CountedForward, case_label: str, gamma: float) -> Dict[str, object]:
    c = np.zeros(model.mode_count)
    D = fwd.residual_c(c)
    G = fwd.G(c)
    b = model.project_residual(D)
    J = model.psi_np.T @ (model.weights_dof_np[:, None] * G)
    mask = surface_dofs(model)
    n_s = max(int(np.sum(model.surface_mask_np)), 1)
    gb = J.T @ b
    gV = G.T @ (model.weights_dof_np * D)
    gS = G[mask, :].T @ (D[mask] / n_s)
    db = step_lm(J, b, model.case.trust_initial_lambda, model.case.trust_max_step)
    sw = np.sqrt(model.weights_dof_np)
    dV = step_lm(sw[:, None] * G, sw * D, model.case.trust_initial_lambda, model.case.trust_max_step)
    scale = 1.0 / math.sqrt(n_s)
    dS = step_lm(scale * G[mask, :], scale * D[mask], model.case.trust_initial_lambda, model.case.trust_max_step)
    return {
        "case": model.case.name,
        "case_label": case_label,
        "constraint_set": model.constraint_set,
        "gamma": gamma,
        "cos_gb_gV": safe_cos(gb, gV),
        "cos_gb_gS": safe_cos(gb, gS),
        "cos_gV_gS": safe_cos(gV, gS),
        "cos_step_b_V": safe_cos(db, dV),
        "cos_step_b_S": safe_cos(db, dS),
        "cos_step_V_S": safe_cos(dV, dS),
        "reduced_gradient_norm": float(np.linalg.norm(gb)),
        "volume_gradient_norm": float(np.linalg.norm(gV)),
        "surface_gradient_norm": float(np.linalg.norm(gS)),
    }


def decide_objective(rows: List[Dict[str, object]], align: List[Dict[str, object]], histories: List[Dict[str, object]]) -> Tuple[str, str]:
    lm = [r for r in rows if r["globalization"] == "LM_trust" and r["objective"] in OBJECTIVES]
    reduced = [r for r in lm if r["objective"] == "reduced"]
    volume = [r for r in lm if r["objective"] == "volume_physical"]
    surface = [r for r in lm if r["objective"] == "surface_physical"]
    red_surface_good = np.mean([float(r["surface_residual_ratio"]) < 1.0 for r in reduced]) if reduced else 0.0
    red_phys_good = np.mean([float(r["physical_residual_ratio"]) < 1.0 for r in reduced]) if reduced else 0.0
    surf_good = np.mean([float(r["surface_residual_ratio"]) < 1.0 for r in surface]) if surface else 0.0
    vol_good = np.mean([float(r["physical_residual_ratio"]) < 1.0 for r in volume]) if volume else 0.0
    med_surface = np.median([float(r["surface_residual_ratio"]) for r in surface]) if surface else np.inf
    med_volume_surface = np.median([float(r["surface_residual_ratio"]) for r in volume]) if volume else np.inf
    med_cos_b_s = np.nanmedian([float(a["cos_gb_gS"]) for a in align])
    comb_red = [h for h in histories if h["case"] == "A_FDM_comb_coupon" and h["objective"] == "reduced"]
    perp_growth = any(float(h["D_perp_M_norm"]) > float(h["weighted_physical_residual"]) * 0.95 for h in comb_red)
    if red_surface_good >= 0.85 and red_phys_good >= 0.85 and med_cos_b_s > 0.5 and not perp_growth:
        return "KEEP_REDUCED_OBJECTIVE", "Reduced LM/trust consistently improved physical/surface residuals and stayed aligned with surface gradients."
    if surf_good >= max(vol_good - 0.05, 0.5) and med_surface <= med_volume_surface * 1.02:
        return "SURFACE_OBJECTIVE_PRIMARY", "Surface residual is the predeclared engineering evaluation target and the surface-G objective was at least as consistent as all-DOF physical optimization."
    if vol_good >= 0.5 and (red_surface_good < vol_good or red_phys_good < vol_good):
        return "G_PRIMARY_J_DIAGNOSTIC", "Physical G-based optimization was more consistent than reduced J optimization; J remains a reduced response diagnostic."
    return "OBJECTIVE_UNRESOLVED", "No objective gave a consistent enough improvement/alignment pattern for a final method decision."


def local_G_q(fwd: CountedForward, q_base: np.ndarray, P: np.ndarray, h: float = FD_STEP_R) -> np.ndarray:
    r0 = fwd.residual_q(q_base)
    G = np.zeros((r0.size, P.shape[1]), dtype=float)
    for j in range(P.shape[1]):
        dq = h * P[:, j]
        G[:, j] = (fwd.residual_q(q_base + dq) - fwd.residual_q(q_base - dq)) / (2.0 * h)
        fwd.fd_internal_calls += 2
    fwd.jacobian_builds += 1
    return G


def objective_rH_q(model: InherentStrainFEM, D: np.ndarray, Gq: np.ndarray, objective: str) -> Tuple[np.ndarray, np.ndarray]:
    if objective == "volume_physical":
        sw = np.sqrt(model.weights_dof_np)
        return sw * D, sw[:, None] * Gq
    if objective == "surface_physical":
        mask = surface_dofs(model)
        scale = 1.0 / math.sqrt(max(int(np.sum(model.surface_mask_np)), 1))
        return scale * D[mask], scale * Gq[mask, :]
    if objective == "reduced":
        return model.project_residual(D), model.psi_np.T @ (model.weights_dof_np[:, None] * Gq)
    raise ValueError(objective)


def weighted_trust_matrix(model: InherentStrainFEM, P: np.ndarray) -> np.ndarray:
    return P.T @ (model.weights_dof_np[:, None] * P)


def trust_lsq(H: np.ndarray, r: np.ndarray, R: np.ndarray, radius: float) -> np.ndarray:
    HtH = H.T @ H
    g = H.T @ r
    try:
        z = -np.linalg.solve(HtH + 1e-12 * np.eye(HtH.shape[0]), g)
    except np.linalg.LinAlgError:
        z = -np.linalg.lstsq(H, r, rcond=1e-10)[0]
    norm = math.sqrt(max(float(z.T @ R @ z), 0.0))
    if norm <= radius:
        return z
    lo, hi = 0.0, 1.0
    eye = np.eye(HtH.shape[0])
    while True:
        zhi = -np.linalg.solve(HtH + hi * R + 1e-12 * eye, g)
        nhi = math.sqrt(max(float(zhi.T @ R @ zhi), 0.0))
        if nhi <= radius or hi > 1e12:
            break
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        zm = -np.linalg.solve(HtH + mid * R + 1e-12 * eye, g)
        nm = math.sqrt(max(float(zm.T @ R @ zm), 0.0))
        if nm > radius:
            lo = mid
        else:
            hi = mid
    return -np.linalg.solve(HtH + hi * R + 1e-12 * eye, g)


def nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if A.size == 0:
        return np.eye(A.shape[1])
    _, s, vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > tol))
    return vt[rank:].T


def solve_authority_level(
    model: InherentStrainFEM,
    q_base: np.ndarray,
    P: np.ndarray,
    r: np.ndarray,
    H: np.ndarray,
    radius: float,
    level: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    eq_dofs = np.repeat(model.equality_mask_np, 3)
    Aeq = P[eq_dofs, :] if np.any(eq_dofs) else np.zeros((0, P.shape[1]))
    if level == "U0_unconstrained":
        N = np.eye(P.shape[1])
    else:
        N = nullspace(Aeq)
    if N.shape[1] == 0:
        y = np.zeros(P.shape[1])
        return y, {"status": "zero_nullspace", "first_order": 0.0}
    Hn = H @ N
    Rn = N.T @ weighted_trust_matrix(model, P) @ N
    t0 = trust_lsq(Hn, r, Rn, radius)
    if level != "U2_fully_admissible" or model.case.allowance_value <= 0.0 or not np.any(model.allowance_mask_np):
        y = N @ t0
        grad = H.T @ (r + H @ y)
        return y, {"status": "linear_trust_nullspace", "first_order": float(np.linalg.norm(N.T @ grad))}
    try:
        from scipy.optimize import minimize
    except Exception:
        y = N @ t0
        return y, {"status": "scipy_unavailable_used_U1", "first_order": ""}
    allow_dofs = np.repeat(model.allowance_mask_np, 3)
    Pn_allow = P[allow_dofs, :] @ N
    q_allow = q_base[allow_dofs]
    bnd = model.case.allowance_value

    def obj(t: np.ndarray) -> float:
        rr = r + Hn @ t
        return 0.5 * float(np.dot(rr, rr))

    def grad(t: np.ndarray) -> np.ndarray:
        return Hn.T @ (r + Hn @ t)

    constraints = [
        {"type": "ineq", "fun": lambda t, R=Rn, rad=radius: rad**2 - float(t.T @ R @ t)},
        {"type": "ineq", "fun": lambda t, Pn=Pn_allow, q=q_allow, b=bnd: b - np.abs(q + Pn @ t)},
    ]
    res = minimize(obj, t0, jac=grad, method="SLSQP", constraints=constraints, options={"ftol": 1e-11, "maxiter": 300})
    y = N @ np.asarray(res.x, dtype=float)
    return y, {"status": f"SLSQP:{res.message}", "first_order": float(np.linalg.norm(grad(np.asarray(res.x, dtype=float))))}


def parent_basis(model: InherentStrainFEM) -> np.ndarray:
    basis = compute_free_modes(model.mesh, PARENT_MODE_COUNT)
    return basis.psi[:, :PARENT_MODE_COUNT]


def run_nested_authority(preferred: str, constrained_finals: Dict[Tuple[str, str], np.ndarray]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    assertions: List[Dict[str, object]] = []
    for case_name, cset, setting in MANUFACTURING_CASES:
        model = build_model(case_name, cset, 1.0)
        Phi = parent_basis(model)
        nested_error_6_12 = float(np.linalg.norm(Phi[:, :6] - Phi[:, :12][:, :6]))
        nested_error_12_18 = float(np.linalg.norm(Phi[:, :12] - Phi[:, :18][:, :12]))
        for state in ("initial", "converged_preferred_constrained"):
            if state == "initial":
                q_base = np.zeros(model.mesh.num_dofs)
            else:
                q_base = constrained_finals.get((case_name, cset), np.zeros(model.mesh.num_dofs))
            fwd = CountedForward(model, "reference_domain")
            D0 = fwd.residual_q(q_base)
            delta_ref = max(weighted_norm(model, D0), 1.0)
            for m in (6, 12, 18):
                P = Phi[:, :m]
                Gq = local_G_q(fwd, q_base, P, FD_STEP_R)
                r, H = objective_rH_q(model, D0, Gq, preferred)
                for radius_mult in (0.25, 0.5, 1.0, 2.0):
                    radius = delta_ref * radius_mult
                    level_values = {}
                    for level in ("U0_unconstrained", "U1_equality_constrained", "U2_fully_admissible"):
                        y, info = solve_authority_level(model, q_base, P, r, H, radius, level)
                        pred = r + H @ y
                        q_trial = q_base + P @ y
                        actual_D = fwd.residual_q(q_trial)
                        actual_r, _ = objective_rH_q(model, actual_D, Gq, preferred)
                        pred_norm = float(np.linalg.norm(pred))
                        actual_norm = float(np.linalg.norm(actual_r))
                        level_values[level] = pred_norm
                        row = {
                            "manufacturing_setting": setting,
                            "case": case_name,
                            "constraint_set": cset,
                            "evaluation_state": state,
                            "preferred_objective": preferred,
                            "basis_size": m,
                            "constraint_level": level,
                            "trust_radius": radius,
                            "trust_radius_normalized": radius_mult,
                            "delta_ref": delta_ref,
                            "predicted_residual_norm": pred_norm,
                            "current_residual_norm": float(np.linalg.norm(r)),
                            "actual_replay_norm": actual_norm,
                            "predicted_residual_ratio": pred_norm / max(float(np.linalg.norm(r)), EPS),
                            "actual_residual_ratio": actual_norm / max(float(np.linalg.norm(r)), EPS),
                            "linearization_error": float(np.linalg.norm(actual_r - pred) / max(np.linalg.norm(pred), EPS)),
                            "nested_error_6_12": nested_error_6_12,
                            "nested_error_12_18": nested_error_12_18,
                            "solver_status": info["status"],
                            "first_order_metric": info["first_order"],
                        }
                        row.update(violation_metrics(model, q_trial))
                        rows.append(row)
                    ok_level = level_values["U0_unconstrained"] <= level_values["U1_equality_constrained"] + 1e-7 and level_values["U1_equality_constrained"] <= level_values["U2_fully_admissible"] + 1e-7
                    assertions.append({
                        "assertion_type": "constraint_level_monotonicity",
                        "case": case_name,
                        "constraint_set": cset,
                        "state": state,
                        "basis_size": m,
                        "radius": radius_mult,
                        "status": "PASS" if ok_level else "FAIL",
                        "lhs": level_values["U0_unconstrained"],
                        "mid": level_values["U1_equality_constrained"],
                        "rhs": level_values["U2_fully_admissible"],
                    })
            for level in ("U0_unconstrained", "U1_equality_constrained", "U2_fully_admissible"):
                for radius_mult in (0.25, 0.5, 1.0, 2.0):
                    vals = {
                        int(r["basis_size"]): float(r["predicted_residual_norm"])
                        for r in rows
                        if r["case"] == case_name and r["constraint_set"] == cset and r["evaluation_state"] == state and r["constraint_level"] == level and float(r["trust_radius_normalized"]) == radius_mult
                    }
                    ok_basis = vals[18] <= vals[12] + 1e-7 and vals[12] <= vals[6] + 1e-7
                    assertions.append({
                        "assertion_type": "basis_size_monotonicity",
                        "case": case_name,
                        "constraint_set": cset,
                        "state": state,
                        "constraint_level": level,
                        "radius": radius_mult,
                        "status": "PASS" if ok_basis else "FAIL",
                        "lhs": vals[18],
                        "mid": vals[12],
                        "rhs": vals[6],
                    })
    return rows, assertions


def constrained_step_scipy(model: InherentStrainFEM, r: np.ndarray, H: np.ndarray, c: np.ndarray, lam: float, solver: str) -> Tuple[np.ndarray, Dict[str, object]]:
    try:
        from scipy.optimize import Bounds, NonlinearConstraint, minimize
    except Exception as exc:
        return step_lm(H, r, lam, model.case.trust_max_step), {"status": f"scipy unavailable: {exc}", "first_order": ""}
    scale = np.maximum(np.linalg.norm(H, axis=0), 1e-8)
    Hs = H / scale[None, :]
    trust = model.case.trust_max_step

    def unpack(y):
        return y / scale

    def obj(y):
        z = unpack(y)
        rr = r + H @ z
        return 0.5 * float(np.dot(rr, rr)) + 0.5 * lam * float(np.dot(z, z))

    def grad(y):
        z = unpack(y)
        gz = H.T @ (r + H @ z) + lam * z
        return gz / scale

    y0 = scale * step_lm(H, r, lam, trust)
    constraints = []
    if np.isfinite(trust):
        constraints.append({"type": "ineq", "fun": lambda y: trust**2 - float(np.dot(unpack(y), unpack(y)))})
    if model.case.allowance_value > 0.0 and np.any(model.allowance_mask_np):
        allow_dofs = np.repeat(model.allowance_mask_np, 3)
        P = model.psi_np[allow_dofs, :]
        q0 = model.q_from_c(c)[allow_dofs]
        bnd = model.case.allowance_value
        constraints.append({"type": "ineq", "fun": lambda y, P=P, q0=q0, bnd=bnd: bnd - np.abs(q0 + P @ unpack(y))})
    if solver == "trust-constr":
        cons = []
        if np.isfinite(trust):
            cons.append(NonlinearConstraint(lambda y: trust**2 - float(np.dot(unpack(y), unpack(y))), 0.0, np.inf))
        if model.case.allowance_value > 0.0 and np.any(model.allowance_mask_np):
            allow_dofs = np.repeat(model.allowance_mask_np, 3)
            P = model.psi_np[allow_dofs, :]
            q0 = model.q_from_c(c)[allow_dofs]
            bnd = model.case.allowance_value
            cons.append(NonlinearConstraint(lambda y, P=P, q0=q0, bnd=bnd: bnd - np.abs(q0 + P @ unpack(y)), 0.0, np.inf))
        res = minimize(obj, y0, jac=grad, method="trust-constr", constraints=cons, options={"gtol": 1e-9, "xtol": 1e-9, "maxiter": 300, "verbose": 0})
    else:
        res = minimize(obj, y0, jac=grad, method="SLSQP", constraints=constraints, options={"ftol": 1e-11, "maxiter": 300})
    z = unpack(np.asarray(res.x, dtype=float))
    return z, {"status": f"{solver}:{res.message}", "first_order": float(np.linalg.norm(H.T @ (r + H @ z) + lam * z))}


def run_constrained_solver(model: InherentStrainFEM, preferred: str, solver_mode: str) -> Tuple[Dict[str, object], np.ndarray]:
    fwd = CountedForward(model, "reference_domain")
    c = np.zeros(model.mode_count)
    D0 = fwd.residual_c(c)
    init = objective_norms(model, D0)
    lam = model.case.trust_initial_lambda
    accepted = rejected = 0
    t0 = time.perf_counter()
    status = ""
    first_order = ""
    for _ in range(MAX_ITER):
        r, H, _, _ = objective_pair(model, fwd, c, preferred)
        if solver_mode == "projection_scaling":
            z = step_lm(H, r, lam, model.case.trust_max_step)
            trial = project_c_allowance(c + z, model)
            status = "projected_lm_step"
            first_order = float(np.linalg.norm(H.T @ (r + H @ (trial - c)) + lam * (trial - c)))
        else:
            z_slsqp, info_s = constrained_step_scipy(model, r, H, c, lam, "SLSQP")
            z_trust, info_t = constrained_step_scipy(model, r, H, c, lam, "trust-constr")
            if info_t["first_order"] != "" and (info_s["first_order"] == "" or float(info_t["first_order"]) <= float(info_s["first_order"])):
                z, info = z_trust, info_t
            else:
                z, info = z_slsqp, info_s
            trial = c + z
            status = info["status"]
            first_order = info["first_order"]
        r_trial, _, D_trial, _ = objective_pair(model, fwd, trial, preferred)
        if objective_value(r_trial) <= objective_value(r) and violation_metrics(model, model.q_from_c(trial))["constraint_violation_max"] < 1e-8:
            c = trial
            accepted += 1
            lam *= 0.5
        else:
            rejected += 1
            lam *= 5.0
        if np.linalg.norm(r_trial) / max(np.linalg.norm(r), EPS) < TOL:
            break
    Df = fwd.residual_c(c)
    final = objective_norms(model, Df)
    row = base_result_row(model, fwd, model.case.name, 1.0, preferred, "constrained_closure", solver_mode, init, final, c, t0, accepted, rejected)
    row["iterations"] = accepted + rejected
    row["solver_status"] = status
    row["first_order_metric"] = first_order
    row["closure_status"] = "PASS" if first_order != "" and float(first_order) < 1e-6 else ("CONSTRAINED_SOLVER_UNRESOLVED" if model.case.name == "A_FDM_comb_coupon" and solver_mode != "projection_scaling" else "CHECK")
    return row, model.q_from_c(c)


def constrained_solver_closure(preferred: str) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], np.ndarray]]:
    rows: List[Dict[str, object]] = []
    finals: Dict[Tuple[str, str], np.ndarray] = {}
    for case_name, cset, _ in MANUFACTURING_CASES:
        model = build_model(case_name, cset, 1.0)
        for mode in ("projection_scaling", "iterative_constrained_local_solve"):
            row, q_final = run_constrained_solver(model, preferred, mode)
            row["manufacturing_setting"] = _
            rows.append(row)
            if mode == "iterative_constrained_local_solve":
                finals[(case_name, cset)] = q_final
    return rows, finals


def sid_scope_validation() -> List[Dict[str, object]]:
    rows = []
    for case_name, cset, label, gammas in OBJECTIVE_CASES:
        for gamma in gammas:
            model = build_model(case_name, cset, gamma)
            fwd = CountedForward(model, "reference_domain")
            c = np.zeros(model.mode_count)
            D0 = fwd.residual_c(c)
            b0 = model.project_residual(D0)
            J = fwd.J(c)
            raw = c - b0
            projected = project_c_allowance(raw, model)
            D1 = fwd.residual_c(projected)
            projection_active = bool(np.linalg.norm(projected - raw) > 1e-10)
            constraints_active = bool(np.sum(model.equality_mask_np) > 0 or model.case.allowance_value > 0.0)
            rows.append({
                "case": case_name,
                "case_label": label,
                "constraint_set": cset,
                "gamma": gamma,
                "s_id": prev.spectral_radius_identity(J),
                "first_identity_step_contraction_ratio": np.linalg.norm(model.project_residual(D1)) / max(np.linalg.norm(b0), EPS),
                "projection_active": projection_active,
                "constraints_active": constraints_active,
                "final_convergence": run_identity_baseline(model, CountedForward(model, "reference_domain"), label, gamma)[0]["own_objective_converged"],
                "scope_label": "NOT_DIRECTLY_APPLICABLE" if projection_active else "LOCAL_UNPROJECTED_COORDINATE_DIAGNOSTIC",
            })
    return rows


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xr = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    yr = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return safe_cos(xr - np.mean(xr), yr - np.mean(yr))


def cj_scope_validation() -> List[Dict[str, object]]:
    base_rows = []
    for case_name, cset, label, gammas in OBJECTIVE_CASES:
        for gamma in gammas:
            model = build_model(case_name, cset, gamma)
            fwd = CountedForward(model, "reference_domain")
            c = np.zeros(model.mode_count)
            D0 = fwd.residual_c(c)
            b = model.project_residual(D0)
            J = fwd.J(c)
            diag = np.diag(J)
            safe = np.where(np.abs(diag) < 1e-4, np.sign(diag + 1e-12) * 1e-4, diag)
            d_diag = -b / safe
            d_full = step_lm(J, b, 1e-10, model.case.trust_max_step)
            D_diag = fwd.residual_c(project_c_allowance(c + d_diag, model))
            D_full = fwd.residual_c(project_c_allowance(c + d_full, model))
            full_row, _, _ = run_objective_solver(model, CountedForward(model, "reference_domain"), label, gamma, "reduced", "GN", "full_reduced_GN")
            diag_row = run_reduced_diagonal(model, label, gamma)
            base_rows.append({
                "case": case_name,
                "case_label": label,
                "constraint_set": cset,
                "gamma": gamma,
                "c_J": prev.coupling_ratio(J),
                "local_diag_full_step_direction_difference": 1.0 - safe_cos(d_diag, d_full),
                "one_step_physical_gap": weighted_norm(model, D_diag) - weighted_norm(model, D_full),
                "one_step_surface_gap": surface_norm(model, D_diag) - surface_norm(model, D_full),
                "final_physical_gap": float(diag_row["physical_residual_ratio"]) - float(full_row["physical_residual_ratio"]),
                "final_surface_gap": float(diag_row["surface_residual_ratio"]) - float(full_row["surface_residual_ratio"]),
                "reduced_objective_gap": float(diag_row["reduced_residual_ratio"]) - float(full_row["reduced_residual_ratio"]),
            })
    metrics = [
        "local_diag_full_step_direction_difference",
        "one_step_physical_gap",
        "one_step_surface_gap",
        "final_physical_gap",
        "final_surface_gap",
        "reduced_objective_gap",
    ]
    for metric in metrics:
        rho = spearman([float(r["c_J"]) for r in base_rows], [float(r[metric]) for r in base_rows])
        for row in base_rows:
            row[f"spearman_cJ_vs_{metric}"] = rho
    return base_rows


def run_reduced_diagonal(model: InherentStrainFEM, label: str, gamma: float) -> Dict[str, object]:
    fwd = CountedForward(model, "reference_domain")
    c = np.zeros(model.mode_count)
    D0 = fwd.residual_c(c)
    init = objective_norms(model, D0)
    t0 = time.perf_counter()
    accepted = 0
    for _ in range(MAX_ITER):
        D = fwd.residual_c(c)
        b = model.project_residual(D)
        J = fwd.J(c)
        diag = np.diag(J)
        safe = np.where(np.abs(diag) < 1e-4, np.sign(diag + 1e-12) * 1e-4, diag)
        c = project_c_allowance(c - b / safe, model)
        accepted += 1
    Df = fwd.residual_c(c)
    return base_result_row(model, fwd, label, gamma, "reduced", "GN", "diagonal_reduced_GN", init, objective_norms(model, Df), c, t0, accepted, 0)


def forward_model_objective_robustness(preferred: str) -> List[Dict[str, object]]:
    rows = []
    for case_name, cset, label in ROBUSTNESS_CASES:
        for gamma in (1.0, 1.5):
            for model_name in ("reference_domain", "updated_geometry"):
                model = build_model(case_name, cset, gamma)
                identity, _ = run_identity_baseline(model, CountedForward(model, model_name), label, gamma)
                rows.append(identity)
                diag = run_reduced_diagonal_model(model, model_name, label, gamma)
                rows.append(diag)
                pref, _, _ = run_objective_solver(model, CountedForward(model, model_name), label, gamma, preferred, "LM_trust", f"preferred_{preferred}_LM_trust")
                rows.append(pref)
                red, _, _ = run_objective_solver(model, CountedForward(model, model_name), label, gamma, "reduced", "LM_trust", "diagnostic_reduced_J_LM_trust")
                rows.append(red)
    return rows


def run_reduced_diagonal_model(model: InherentStrainFEM, model_name: str, label: str, gamma: float) -> Dict[str, object]:
    fwd = CountedForward(model, model_name)
    c = np.zeros(model.mode_count)
    D0 = fwd.residual_c(c)
    init = objective_norms(model, D0)
    t0 = time.perf_counter()
    accepted = 0
    for _ in range(MAX_ITER):
        D = fwd.residual_c(c)
        b = model.project_residual(D)
        J = fwd.J(c)
        diag = np.diag(J)
        safe = np.where(np.abs(diag) < 1e-4, np.sign(diag + 1e-12) * 1e-4, diag)
        c = project_c_allowance(c - b / safe, model)
        accepted += 1
    Df = fwd.residual_c(c)
    return base_result_row(model, fwd, label, gamma, "reduced", "GN", "diagonal_response", init, objective_norms(model, Df), c, t0, accepted, 0)


def computational_cost_closure(preferred: str) -> List[Dict[str, object]]:
    rows = []
    timing_keys = [
        ("A_FDM_comb_coupon", "bottom_contact", 1.0, "reference_domain", preferred, "preferred_LM_trust"),
        ("A_FDM_comb_coupon", "bottom_contact", 1.0, "updated_geometry", preferred, "preferred_LM_trust"),
        ("B_L_bracket_datum", "bottom_only", 1.0, "reference_domain", preferred, "preferred_LM_trust"),
        ("C_wall_allowance", "bottom_plus_allowance", 1.0, "reference_domain", preferred, "preferred_LM_trust"),
    ]
    for case_name, cset, gamma, model_name, objective, method in timing_keys:
        times = []
        calls = []
        jacs = []
        fd_calls = []
        model = build_model(case_name, cset, gamma)
        run_objective_solver(model, CountedForward(model, model_name), case_name, gamma, objective, "LM_trust", method, max_iter=2)
        for repeat in range(5):
            model = build_model(case_name, cset, gamma)
            row, _, _ = run_objective_solver(model, CountedForward(model, model_name), case_name, gamma, objective, "LM_trust", method)
            times.append(float(row["runtime_s"]))
            calls.append(int(row["actual_residual_calls"]))
            jacs.append(int(row["jacobian_construction_count"]))
            fd_calls.append(int(row["fd_internal_residual_calls"]))
        rows.append({
            "case": case_name,
            "constraint_set": cset,
            "gamma": gamma,
            "forward_model": model_name,
            "objective": objective,
            "method": method,
            "basis_size": MODE_COUNT,
            "repeat_count": 5,
            "median_runtime_s": float(np.median(times)),
            "iqr_runtime_s": float(np.percentile(times, 75) - np.percentile(times, 25)),
            "median_actual_residual_calls": float(np.median(calls)),
            "median_jacobian_construction_count": float(np.median(jacs)),
            "median_fd_internal_residual_calls": float(np.median(fd_calls)),
        })
    return rows


def completeness_manifest(outputs: Dict[str, int]) -> List[Dict[str, object]]:
    expected = {
        "objective_comparison": 7 * (1 + len(OBJECTIVES) * len(GLOBALIZATIONS)),
        "residual_component_histories": "nonzero",
        "objective_alignment_diagnostics": 7,
        "nested_authority_ablation": len(MANUFACTURING_CASES) * 2 * 3 * 3 * 4,
        "constrained_solver_closure": len(MANUFACTURING_CASES) * 2,
        "sid_local_scope_validation": 7,
        "cj_scope_validation": 7,
        "forward_model_objective_robustness": 3 * 2 * 2 * 4,
        "computational_cost_closure": 4,
    }
    rows = []
    for name, exp in expected.items():
        got = outputs.get(name, 0)
        if exp == "nonzero":
            missing = 0 if got > 0 else 1
            status = "PASS" if got > 0 else "FAIL"
            exp_val = "nonzero"
        else:
            missing = max(int(exp) - got, 0)
            status = "PASS" if missing == 0 else "FAIL"
            exp_val = int(exp)
        rows.append({"product": name, "expected_rows": exp_val, "completed_rows": got, "missing_rows": missing, "status": status})
    return rows


def write_report(
    decision: str,
    reason: str,
    objective_rows: List[Dict[str, object]],
    histories: List[Dict[str, object]],
    align: List[Dict[str, object]],
    authority: List[Dict[str, object]],
    assertions: List[Dict[str, object]],
    constrained: List[Dict[str, object]],
    sid: List[Dict[str, object]],
    cj: List[Dict[str, object]],
    robust: List[Dict[str, object]],
    cost: List[Dict[str, object]],
    manifest: List[Dict[str, object]],
) -> None:
    preferred = preferred_objective_name(decision)
    def top(rows, n=20):
        return rows[: min(n, len(rows))]

    lines = [
        "# Method Objective and Validation Closure",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "No manuscript TeX was modified. Surface residual was predeclared as the engineering evaluation target; all-DOF physical residual is reported as an independent sensitivity metric.",
        "",
        "## 1. Executive decision",
        "",
        f"Decision: **{decision}**.",
        "",
        reason,
        "",
        "Constrained solver closure is not fully validated for the slotted comb case when judged by first-order optimality; that row is explicitly marked `CONSTRAINED_SOLVER_UNRESOLVED` and should not be used as a final constrained-solver validation claim.",
        "",
        "## 2. Objective comparison",
        "",
        md_table(["case", "gamma", "objective", "glob", "method", "red", "phys", "surf", "own conv"], [[r["case"], r["gamma"], r["objective"], r["globalization"], r["method"], fmt(r["reduced_residual_ratio"]), fmt(r["physical_residual_ratio"]), fmt(r["surface_residual_ratio"]), r.get("own_objective_converged", "")] for r in objective_rows]),
        "",
        "## 3. Residual-component mechanism",
        "",
        md_table(["case", "gamma", "obj", "glob", "it", "D_parallel", "D_perp", "total", "surface"], [[r["case"], r["gamma"], r["objective"], r["globalization"], r["iteration"], fmt(r["D_parallel_M_norm"]), fmt(r["D_perp_M_norm"]), fmt(r["weighted_physical_residual"]), fmt(r["surface_residual"])] for r in top(histories, 48)]),
        "",
        "## 4. Gradient and step alignment",
        "",
        md_table(["case", "gamma", "cos gb/gV", "cos gb/gS", "cos gV/gS", "cos step b/V", "cos step b/S"], [[r["case"], r["gamma"], fmt(r["cos_gb_gV"]), fmt(r["cos_gb_gS"]), fmt(r["cos_gV_gS"]), fmt(r["cos_step_b_V"]), fmt(r["cos_step_b_S"])] for r in align]),
        "",
        "## 5. Method-level consequence",
        "",
        f"The preferred downstream objective is `{preferred}`. Reduced `J=Psi^T M G` is retained as a diagnostic unless the decision is `KEEP_REDUCED_OBJECTIVE`.",
        "",
        "## 6. Corrected nested authority",
        "",
        md_table(["setting", "state", "m", "level", "radius", "pred", "actual", "lin err", "viol"], [[r["manufacturing_setting"], r["evaluation_state"], r["basis_size"], r["constraint_level"], r["trust_radius_normalized"], fmt(r["predicted_residual_ratio"]), fmt(r["actual_residual_ratio"]), fmt(r["linearization_error"]), fmt(r["constraint_violation_max"])] for r in top(authority, 64)]),
        "",
        "## 7. Authority sanity checks",
        "",
        md_table(["type", "case", "state", "m/level", "radius", "status"], [[r["assertion_type"], r["case"], r["state"], r.get("basis_size", r.get("constraint_level", "")), r["radius"], r["status"]] for r in assertions]),
        "",
        "## 8. Constrained solver closure",
        "",
        md_table(["setting", "method", "phys", "surf", "viol", "first order", "status"], [[r["manufacturing_setting"], r["method"], fmt(r["physical_residual_ratio"]), fmt(r["surface_residual_ratio"]), fmt(r["constraint_violation_max"]), fmt(r["first_order_metric"]), r["closure_status"]] for r in constrained]),
        "",
        "## 9. Correct scope of s_id",
        "",
        md_table(["case", "gamma", "s_id", "first step", "projection", "constraints", "scope"], [[r["case"], r["gamma"], fmt(r["s_id"]), fmt(r["first_identity_step_contraction_ratio"]), r["projection_active"], r["constraints_active"], r["scope_label"]] for r in sid]),
        "",
        "## 10. Correct scope of c_J",
        "",
        md_table(["case", "gamma", "c_J", "step diff", "one phys gap", "one surf gap", "final phys gap", "final surf gap"], [[r["case"], r["gamma"], fmt(r["c_J"]), fmt(r["local_diag_full_step_direction_difference"]), fmt(r["one_step_physical_gap"]), fmt(r["one_step_surface_gap"]), fmt(r["final_physical_gap"]), fmt(r["final_surface_gap"])] for r in cj]),
        "",
        "## 11. Model R/U robustness",
        "",
        md_table(["case", "gamma", "model", "method", "objective", "phys", "surf", "calls", "fd calls"], [[r["case"], r["gamma"], r["forward_model"], r["method"], r["objective"], fmt(r["physical_residual_ratio"]), fmt(r["surface_residual_ratio"]), r["actual_residual_calls"], r["fd_internal_residual_calls"]] for r in robust]),
        "",
        "## 12. Fair cost accounting",
        "",
        md_table(["case", "gamma", "model", "objective", "method", "median calls", "FD calls", "median runtime", "IQR"], [[r["case"], r["gamma"], r["forward_model"], r["objective"], r["method"], fmt(r["median_actual_residual_calls"]), fmt(r["median_fd_internal_residual_calls"]), fmt(r["median_runtime_s"]), fmt(r["iqr_runtime_s"])] for r in cost]),
        "",
        "## 13. Completeness manifest",
        "",
        md_table(["product", "expected", "completed", "missing", "status"], [[r["product"], r["expected_rows"], r["completed_rows"], r["missing_rows"], r["status"]] for r in manifest]),
        "",
        "## 14. Final Method recommendation",
        "",
        final_method_recommendation(decision),
        "",
        "## 15. Final experiment recommendation",
        "",
        md_table(["prior experiment", "recommendation"], [
            ["response_regime_core", "REVISE"],
            ["forward_model_robustness", "REVISE"],
            ["manufacturing_admissibility_matched", "REPLACE"],
            ["constrained_solver_comparison", "REPLACE"],
            ["local_response_authority_ablation", "REPLACE"],
            ["sid/cJ diagnostics", "REVISE"],
            ["noise/roughness/synthetic studies", "MOVE TO SUPPLEMENT"],
            ["old projected authority", "REMOVE"],
        ]),
        "",
        "## 16. Files created",
        "",
    ]
    for name in [
        "objective_comparison.csv",
        "residual_component_histories.csv",
        "objective_alignment_diagnostics.csv",
        "nested_authority_ablation.csv",
        "authority_sanity_assertions.csv",
        "constrained_solver_closure.csv",
        "sid_local_scope_validation.csv",
        "cj_scope_validation.csv",
        "forward_model_objective_robustness.csv",
        "computational_cost_closure.csv",
        "experiment_completeness_manifest.csv",
        "method_closure_metadata.json",
        "METHOD_OBJECTIVE_AND_VALIDATION_CLOSURE.md",
    ]:
        lines.append(f"- `results/response_inversion_method_closure/{name}`")
    lines.extend([
        "- `experiments/fixed_bottom_jax_fem/run_response_inversion_method_closure.py`",
        "",
        "## 17. Reproduction commands",
        "",
        "```powershell",
        "python -B experiments\\fixed_bottom_jax_fem\\run_response_inversion_method_closure.py",
        "```",
    ])
    (RESULT_DIR / "METHOD_OBJECTIVE_AND_VALIDATION_CLOSURE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preferred_objective_name(decision: str) -> str:
    if decision == "KEEP_REDUCED_OBJECTIVE":
        return "reduced"
    if decision == "SURFACE_OBJECTIVE_PRIMARY":
        return "surface_physical"
    if decision == "G_PRIMARY_J_DIAGNOSTIC":
        return "volume_physical"
    return "surface_physical"


def final_method_recommendation(decision: str) -> str:
    if decision == "KEEP_REDUCED_OBJECTIVE":
        return "Use reduced `J`-based optimization as the mathematical root problem; keep physical residuals as validation metrics."
    if decision == "SURFACE_OBJECTIVE_PRIMARY":
        return "Use surface physical `G`-based optimization as the primary Method objective; retain `J` and identity-response quantities as reduced response diagnostics."
    if decision == "G_PRIMARY_J_DIAGNOSTIC":
        return "Use physical `G`-based optimization as the primary Method objective; retain `J` as a diagnostic/projection of physical response."
    return "The Method objective remains unresolved; downstream validation is conditional and should not be written as final."


def main() -> None:
    start = time.perf_counter()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    objective_rows, histories, align_rows, _ = objective_comparison()
    decision, reason = decide_objective(objective_rows, align_rows, histories)
    preferred = preferred_objective_name(decision)
    constrained_rows, constrained_finals = constrained_solver_closure(preferred)
    authority_rows, assertion_rows = run_nested_authority(preferred, constrained_finals)
    if any(r["status"] != "PASS" for r in assertion_rows):
        decision = "OBJECTIVE_UNRESOLVED"
        reason += " Authority monotonicity assertions failed, so downstream authority validation is not final."
    sid_rows = sid_scope_validation()
    cj_rows = cj_scope_validation()
    robust_rows = forward_model_objective_robustness(preferred)
    cost_rows = computational_cost_closure(preferred)
    outputs = {
        "objective_comparison": len(objective_rows),
        "residual_component_histories": len(histories),
        "objective_alignment_diagnostics": len(align_rows),
        "nested_authority_ablation": len(authority_rows),
        "constrained_solver_closure": len(constrained_rows),
        "sid_local_scope_validation": len(sid_rows),
        "cj_scope_validation": len(cj_rows),
        "forward_model_objective_robustness": len(robust_rows),
        "computational_cost_closure": len(cost_rows),
    }
    manifest_rows = completeness_manifest(outputs)
    if any(r["status"] != "PASS" for r in manifest_rows):
        decision = "OBJECTIVE_UNRESOLVED"
        reason += " Completeness manifest has missing rows."

    objective_cols = [
        "case", "case_label", "constraint_set", "gamma", "forward_model", "objective", "globalization", "method", "basis_size",
        "initial_reduced_residual", "final_reduced_residual", "reduced_residual_ratio", "initial_physical_residual", "final_physical_residual",
        "physical_residual_ratio", "initial_surface_residual", "final_surface_residual", "surface_residual_ratio", "own_objective_converged",
        "physical_improved", "surface_improved", "iterations", "accepted_steps", "rejected_steps", "constraint_violation_l2",
        "constraint_violation_rms", "constraint_violation_max", "runtime_s", "actual_residual_calls", "jacobian_construction_count",
        "fd_internal_residual_calls",
    ]
    write_csv(RESULT_DIR / "objective_comparison.csv", objective_rows, objective_cols)
    write_csv(RESULT_DIR / "residual_component_histories.csv", histories, ["case", "case_label", "constraint_set", "gamma", "objective", "globalization", "method", "iteration", "gram_error", "D_parallel_M_norm", "D_perp_M_norm", "weighted_physical_residual", "surface_residual"])
    write_csv(RESULT_DIR / "objective_alignment_diagnostics.csv", align_rows, ["case", "case_label", "constraint_set", "gamma", "cos_gb_gV", "cos_gb_gS", "cos_gV_gS", "cos_step_b_V", "cos_step_b_S", "cos_step_V_S", "reduced_gradient_norm", "volume_gradient_norm", "surface_gradient_norm"])
    write_csv(RESULT_DIR / "nested_authority_ablation.csv", authority_rows, ["manufacturing_setting", "case", "constraint_set", "evaluation_state", "preferred_objective", "basis_size", "constraint_level", "trust_radius", "trust_radius_normalized", "delta_ref", "current_residual_norm", "predicted_residual_norm", "actual_replay_norm", "predicted_residual_ratio", "actual_residual_ratio", "linearization_error", "nested_error_6_12", "nested_error_12_18", "solver_status", "first_order_metric", "constraint_violation_l2", "constraint_violation_rms", "constraint_violation_max"])
    write_csv(RESULT_DIR / "authority_sanity_assertions.csv", assertion_rows, ["assertion_type", "case", "constraint_set", "state", "basis_size", "constraint_level", "radius", "lhs", "mid", "rhs", "status"])
    write_csv(RESULT_DIR / "constrained_solver_closure.csv", constrained_rows, objective_cols + ["manufacturing_setting", "solver_status", "first_order_metric", "closure_status"])
    write_csv(RESULT_DIR / "sid_local_scope_validation.csv", sid_rows, ["case", "case_label", "constraint_set", "gamma", "s_id", "first_identity_step_contraction_ratio", "projection_active", "constraints_active", "final_convergence", "scope_label"])
    write_csv(RESULT_DIR / "cj_scope_validation.csv", cj_rows, ["case", "case_label", "constraint_set", "gamma", "c_J", "local_diag_full_step_direction_difference", "one_step_physical_gap", "one_step_surface_gap", "final_physical_gap", "final_surface_gap", "reduced_objective_gap", "spearman_cJ_vs_local_diag_full_step_direction_difference", "spearman_cJ_vs_one_step_physical_gap", "spearman_cJ_vs_one_step_surface_gap", "spearman_cJ_vs_final_physical_gap", "spearman_cJ_vs_final_surface_gap", "spearman_cJ_vs_reduced_objective_gap"])
    write_csv(RESULT_DIR / "forward_model_objective_robustness.csv", robust_rows, objective_cols)
    write_csv(RESULT_DIR / "computational_cost_closure.csv", cost_rows, ["case", "constraint_set", "gamma", "forward_model", "objective", "method", "basis_size", "repeat_count", "median_runtime_s", "iqr_runtime_s", "median_actual_residual_calls", "median_jacobian_construction_count", "median_fd_internal_residual_calls"])
    write_csv(RESULT_DIR / "experiment_completeness_manifest.csv", manifest_rows, ["product", "expected_rows", "completed_rows", "missing_rows", "status"])
    metadata = {"runtime_s": time.perf_counter() - start, "decision": decision, "preferred_objective": preferred, "max_iter": MAX_ITER, "tol": TOL, "fd_step_reference": FD_STEP_R, "fd_step_updated": FD_STEP_U}
    (RESULT_DIR / "method_closure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(decision, reason, objective_rows, histories, align_rows, authority_rows, assertion_rows, constrained_rows, sid_rows, cj_rows, robust_rows, cost_rows, manifest_rows)
    print(f"Decision: {decision}")
    print(f"Created {RESULT_DIR / 'METHOD_OBJECTIVE_AND_VALIDATION_CLOSURE.md'}")
    print(f"Created raw outputs in {RESULT_DIR}")
    print(f"Runtime: {metadata['runtime_s']:.2f} s")


if __name__ == "__main__":
    main()
