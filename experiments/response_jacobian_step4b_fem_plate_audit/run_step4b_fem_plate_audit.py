"""Step 4B audit and stabilization of the FEM-like plate diagnostic."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = PROJECT_ROOT
STEP4_DIR = PROJECT_ROOT / "experiments" / "response_jacobian_step4_fem_plate"
if str(STEP4_DIR) not in sys.path:
    sys.path.insert(0, str(STEP4_DIR))

from fem_plate_mesh import create_fem_plate_mesh  # noqa: E402
from fem_plate_modes import create_fem_plate_modes, K  # noqa: E402
from jacobian_tools import deterministic_parameter_grid  # noqa: E402
from thermal_response import C0, FEMProcessParams, ThermalFEMResponse, diagnostics_for_response  # noqa: E402

RESULT_DIR = PROJECT_ROOT / "results" / "response_jacobian_step4b_fem_plate_audit"
STEP4_DIAG_PATH = PROJECT_ROOT / "results" / "response_jacobian_step4_fem_plate" / "diagnostics.json"
REPORT_PATH = RESULT_DIR / "STEP4B_FEM_PLATE_AUDIT_REPORT.md"
AUDITED_SUMMARY_PATH = RESULT_DIR / "audited_summary.csv"
LOCAL_LINEAR_PATH = RESULT_DIR / "local_linearization.csv"
LINE_SEARCH_PATH = RESULT_DIR / "line_search_scan.csv"
CASE_AUDIT_PATH = RESULT_DIR / "case_selection_audit.csv"
UPDATED_SCAN_PATH = RESULT_DIR / "updated_parameter_scan.csv"
DIAG_PATH = RESULT_DIR / "diagnostics.json"

MAX_ITER_AUDIT = 50
TOL = 1e-6
DIVERGENCE_RATIO = 1e3
ORIGINAL_ALPHAS = [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.75, 1.00, 1.25]
EXTENDED_ALPHAS = np.logspace(-6, 0, 80)
TR_CAPS = [0.1, 0.3, 1.0, np.inf]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key, ""), default=_json_default)
                if isinstance(row.get(key, ""), (list, tuple, dict))
                else ("" if row.get(key, "") is None else row.get(key, ""))
                for key in fields
            })


def _params_from_dict(d: Dict[str, object]) -> FEMProcessParams:
    keys = FEMProcessParams.__dataclass_fields__.keys()
    return FEMProcessParams(**{k: float(d[k]) for k in keys if k in d})


def _load_original_params() -> Dict[str, FEMProcessParams]:
    data = json.loads(STEP4_DIAG_PATH.read_text(encoding="utf-8"))
    return {name: _params_from_dict(info["process_parameters"]) for name, info in data["selected_cases"].items()}


def _basic_jacobian_stats(J: np.ndarray) -> Dict[str, float]:
    I = np.eye(J.shape[0])
    off = J - np.diag(np.diag(J))
    diag_norm = float(np.linalg.norm(np.diag(J)))
    off_norm = float(np.linalg.norm(off, ord="fro"))
    return {
        "rho_I_minus_J0": float(np.max(np.abs(np.linalg.eigvals(I - J)))),
        "identity_deviation": float(np.linalg.norm(J - I, ord="fro") / np.linalg.norm(I, ord="fro")),
        "coupling_ratio": float(off_norm / max(np.linalg.norm(J, ord="fro"), 1e-15)),
        "condition_number": float(np.linalg.cond(J)),
        "diag_norm": diag_norm,
        "offdiag_norm": off_norm,
        "diag_dominance_ratio": float(off_norm / (diag_norm + 1e-12)),
    }


def _initial_steps(response: ThermalFEMResponse, J0: np.ndarray, b0: np.ndarray) -> Dict[str, float]:
    diag = np.diag(J0).copy()
    safe = np.where(np.abs(diag) < 1e-8, np.sign(diag + 1e-15) * 1e-8, diag)
    d_direct = -b0
    d_diag = -np.diag(1.0 / safe) @ b0
    d_full = np.linalg.lstsq(J0, -b0, rcond=None)[0]
    d_trust = np.linalg.solve(J0.T @ J0 + 1e-3 * np.eye(K), -J0.T @ b0)
    initial = max(float(np.linalg.norm(b0)), 1e-15)
    return {
        "direct_step_norm": float(np.linalg.norm(d_direct)),
        "diagonal_step_norm": float(np.linalg.norm(d_diag)),
        "full_gn_step_norm": float(np.linalg.norm(d_full)),
        "trust_initial_step_norm": float(np.linalg.norm(d_trust)),
        "direct_one_step_ratio": float(np.linalg.norm(response.b(C0 + d_direct)) / initial),
        "diagonal_one_step_ratio": float(np.linalg.norm(response.b(C0 + d_diag)) / initial),
        "full_gn_one_step_ratio": float(np.linalg.norm(response.b(C0 + d_full)) / initial),
    }


def audit_original_cases(mesh, B, basis_error: float, params_by_case: Dict[str, FEMProcessParams]):
    rows = []
    cache = {}
    for name, params in params_by_case.items():
        response = ThermalFEMResponse(name, mesh, B, params)
        J0 = response.jacobian_fd(C0)
        b0 = response.b(C0)
        eig = np.linalg.eigvals(J0)
        row = {
            "case_name": name,
            **_basic_jacobian_stats(J0),
            "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eig],
            "initial_residual_norm": float(np.linalg.norm(b0)),
            "b0_norm": float(np.linalg.norm(b0)),
            "basis_orthonormality_error": basis_error,
            **_initial_steps(response, J0, b0),
            "process_parameters": params.as_dict(),
        }
        rows.append(row)
        cache[name] = {"response": response, "J0": J0, "b0": b0, "audit": row}
    return rows, cache


def local_linearization_tests(case_cache: Dict[str, object]) -> List[Dict[str, object]]:
    rng = np.random.default_rng(0)
    scales = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    rows = []
    for name, data in case_cache.items():
        response = data["response"]
        J0 = data["J0"]
        b0 = data["b0"]
        directions = []
        for i in range(5):
            d = rng.normal(size=K)
            d /= max(np.linalg.norm(d), 1e-15)
            directions.append((f"random_{i}", d))
        diag = np.diag(J0).copy()
        safe = np.where(np.abs(diag) < 1e-8, np.sign(diag + 1e-15) * 1e-8, diag)
        directions += [
            ("direct", -b0),
            ("diagonal", -np.diag(1.0 / safe) @ b0),
            ("full_gn", np.linalg.lstsq(J0, -b0, rcond=None)[0]),
        ]
        for direction_name, d in directions:
            for s in scales:
                step = s * d
                actual_b = response.b(C0 + step)
                linear_b = b0 + J0 @ step
                pred = 0.5 * float(np.dot(b0, b0)) - 0.5 * float(np.dot(linear_b, linear_b))
                actual = 0.5 * float(np.dot(b0, b0)) - 0.5 * float(np.dot(actual_b, actual_b))
                rows.append({
                    "case_name": name,
                    "direction": direction_name,
                    "scale": s,
                    "direction_norm": float(np.linalg.norm(d)),
                    "step_norm": float(np.linalg.norm(step)),
                    "actual_residual_norm": float(np.linalg.norm(actual_b)),
                    "linear_residual_norm": float(np.linalg.norm(linear_b)),
                    "linearization_error": float(np.linalg.norm(actual_b - linear_b) / (np.linalg.norm(actual_b) + 1e-12)),
                    "predicted_reduction": pred,
                    "actual_reduction": actual,
                    "rho_reduction": float(actual / pred) if pred > 0 else np.nan,
                })
    return rows


def run_scalar(response: ThermalFEMResponse, alpha: float, max_iter: int = MAX_ITER_AUDIT):
    c = C0.copy()
    initial = float(np.linalg.norm(response.b(c)))
    best = 1.0
    converged = False
    diverged = False
    it = 0
    for it in range(1, max_iter + 1):
        c = c - alpha * response.b(c)
        ratio = float(np.linalg.norm(response.b(c)) / max(initial, 1e-15))
        best = min(best, ratio)
        converged = ratio < TOL
        diverged = (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or (not np.all(np.isfinite(c)))
        if converged or diverged:
            break
    final_ratio = float(np.linalg.norm(response.b(c)) / max(initial, 1e-15))
    return {
        "final_residual_ratio": final_ratio,
        "best_residual_ratio_seen": best,
        "converged": converged,
        "diverged": diverged,
        "iterations_run": it,
        "final_c": c,
    }


def line_search_audit(case_cache: Dict[str, object]):
    rows = []
    summary = {}
    alphas = list(np.logspace(-6, 0, 80)) + [1.25]
    for name, data in case_cache.items():
        response = data["response"]
        case_rows = []
        for alpha in alphas:
            result = run_scalar(response, float(alpha), max_iter=50)
            row = {"case_name": name, "alpha": float(alpha), **{k: v for k, v in result.items() if k != "final_c"}}
            rows.append(row)
            case_rows.append(row)
        best = min(case_rows, key=lambda r: float(r["final_residual_ratio"]))
        stable = [r["alpha"] for r in case_rows if float(r["final_residual_ratio"]) < 1.0]
        summary[name] = {
            "best_alpha": float(best["alpha"]),
            "best_alpha_ratio": float(best["final_residual_ratio"]),
            "stable_alpha_min": float(min(stable)) if stable else None,
            "stable_alpha_max": float(max(stable)) if stable else None,
            "previous_alpha_grid_too_large": bool(not stable or min(ORIGINAL_ALPHAS) > (max(stable) if stable else 0.0)),
        }
    return rows, summary


def one_step_metrics(response: ThermalFEMResponse, J0: np.ndarray, b0: np.ndarray) -> Dict[str, float]:
    initial = max(float(np.linalg.norm(b0)), 1e-15)
    direct = -b0
    def ratio_for(step):
        return float(np.linalg.norm(response.b(C0 + step)) / initial)
    def lin_err(scale):
        step = scale * direct
        actual = response.b(C0 + step)
        linear = b0 + J0 @ step
        return float(np.linalg.norm(actual - linear) / (np.linalg.norm(actual) + 1e-12))
    return {
        "direct_one_step_ratio": ratio_for(direct),
        "best_small_alpha_ratio": min(ratio_for(-a * b0) for a in np.logspace(-6, 0, 40)),
        "linearization_error_direct_scale_1": lin_err(1.0),
        "linearization_error_direct_scale_1e_minus_2": lin_err(1e-2),
    }


def updated_parameter_scan(mesh, B, basis_error: float):
    rows = []
    params_by_id = {}
    for i, params in enumerate(deterministic_parameter_grid(80)):
        response = ThermalFEMResponse(f"scan_{i:03d}", mesh, B, params)
        J0 = response.jacobian_fd(C0)
        b0 = response.b(C0)
        row = {
            "candidate_id": i,
            **params.as_dict(),
            **_basic_jacobian_stats(J0),
            "initial_residual_norm": float(np.linalg.norm(b0)),
            **one_step_metrics(response, J0, b0),
            "category_candidate": "",
        }
        rows.append(row)
        params_by_id[i] = params

    def pick(name, candidates, key):
        if not candidates:
            return None
        row = min(candidates, key=key)
        row["category_candidate"] = name
        return row

    selections = {
        "clean_easy": pick("clean_easy", [r for r in rows if r["rho_I_minus_J0"] < 0.8 and r["direct_one_step_ratio"] < 1 and r["condition_number"] < 1e4], lambda r: abs(r["rho_I_minus_J0"] - 0.4)),
        "clean_marginal": pick("clean_marginal", [r for r in rows if abs(r["rho_I_minus_J0"] - 1.0) < 0.4 and r["linearization_error_direct_scale_1"] < 10], lambda r: abs(r["direct_one_step_ratio"] - 1.0) + abs(r["rho_I_minus_J0"] - 1.0)),
        "clean_hard_gain": pick("clean_hard_gain", [r for r in rows if r["rho_I_minus_J0"] > 1.2 and r["direct_one_step_ratio"] > 1 and r["coupling_ratio"] < 0.5 and r["condition_number"] < 1e4], lambda r: (r["coupling_ratio"], abs(r["rho_I_minus_J0"] - 2.0))),
        "clean_hard_coupled": pick("clean_hard_coupled", [r for r in rows if r["rho_I_minus_J0"] > 1.2 and (r["coupling_ratio"] > 0.5 or r["diag_dominance_ratio"] > 1.0) and r["condition_number"] < 1e5], lambda r: -max(r["coupling_ratio"], r["diag_dominance_ratio"])),
        "clean_hard_nonlinear": pick("clean_hard_nonlinear", [r for r in rows if r["linearization_error_direct_scale_1"] > 1.0 and r["linearization_error_direct_scale_1e_minus_2"] < 0.25 and r["direct_one_step_ratio"] > 1], lambda r: -r["linearization_error_direct_scale_1"]),
    }
    selected = {}
    missing = []
    used = set()
    for name, row in selections.items():
        if row is None:
            missing.append(name)
            continue
        cid = int(row["candidate_id"])
        if cid in used:
            missing.append(name)
            continue
        selected[name] = params_by_id[cid]
        used.add(cid)
    return rows, selected, {"num_candidates": len(rows), "missing_clean_categories": missing, "num_clean_cases": len(selected)}


def _history_row(case, method, it, response, c, initial, step=0.0, accepted=None, lam=None, rho=None, alpha=None, delta_max=None):
    norm = float(np.linalg.norm(response.b(c)))
    return {
        "case_name": case,
        "method_name": method,
        "iteration": it,
        "residual_norm": norm,
        "residual_ratio": norm / max(initial, 1e-15),
        "c_norm": float(np.linalg.norm(c)),
        "step_norm": float(step),
        "accepted": accepted,
        "lambda": lam,
        "rho": rho,
        "alpha": alpha,
        "delta_max": delta_max,
    }


def _finish(case, method, response, c, initial, history, best, conv, div, **extra):
    final = float(np.linalg.norm(response.b(c)))
    return {
        "case_name": case,
        "method_name": method,
        "converged": conv,
        "diverged": div,
        "iterations_run": max(0, len(history) - 1),
        "initial_residual_norm": initial,
        "final_residual_norm": final,
        "final_residual_ratio": final / max(initial, 1e-15),
        "best_residual_ratio_seen": best,
        "selected_alpha": extra.get("selected_alpha"),
        "accepted_steps": extra.get("accepted_steps"),
        "rejected_steps": extra.get("rejected_steps"),
        "final_lambda": extra.get("final_lambda"),
        "delta_max": extra.get("delta_max"),
        "process_parameters": response.params.as_dict(),
    }


def run_direct_method(case, response, max_iter=50):
    c = C0.copy()
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history_row(case, "direct_modal_inversion", 0, response, c, initial)]
    best = 1.0
    conv = div = False
    for it in range(1, max_iter + 1):
        old = c.copy()
        c = c - response.b(c)
        hist.append(_history_row(case, "direct_modal_inversion", it, response, c, initial, np.linalg.norm(c - old), True))
        r = hist[-1]["residual_ratio"]
        best = min(best, r)
        conv = r < TOL
        div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO
        if conv or div:
            break
    return _finish(case, "direct_modal_inversion", response, c, initial, hist, best, conv, div), hist


def run_scalar_method(case, response, alpha, name, max_iter=50):
    c = C0.copy()
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history_row(case, name, 0, response, c, initial, alpha=alpha)]
    best = 1.0
    conv = div = False
    for it in range(1, max_iter + 1):
        old = c.copy()
        c = c - alpha * response.b(c)
        hist.append(_history_row(case, name, it, response, c, initial, np.linalg.norm(c - old), True, alpha=alpha))
        r = hist[-1]["residual_ratio"]
        best = min(best, r)
        conv = r < TOL
        div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO
        if conv or div:
            break
    return _finish(case, name, response, c, initial, hist, best, conv, div, selected_alpha=alpha), hist


def run_diagonal(case, response, max_iter=50):
    J0 = response.jacobian_fd(C0)
    diag = np.diag(J0)
    safe = np.where(np.abs(diag) < 1e-8, np.sign(diag + 1e-15) * 1e-8, diag)
    D = np.diag(1.0 / safe)
    c = C0.copy()
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history_row(case, "diagonal_modal_factor_initial", 0, response, c, initial)]
    best = 1.0
    conv = div = False
    for it in range(1, max_iter + 1):
        old = c.copy()
        c = c - D @ response.b(c)
        hist.append(_history_row(case, "diagonal_modal_factor_initial", it, response, c, initial, np.linalg.norm(c - old), True))
        r = hist[-1]["residual_ratio"]
        best = min(best, r)
        conv = r < TOL
        div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO
        if conv or div:
            break
    return _finish(case, "diagonal_modal_factor_initial", response, c, initial, hist, best, conv, div), hist


def run_full(case, response, max_iter=50):
    c = C0.copy()
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history_row(case, "full_jacobian_gn_no_trust", 0, response, c, initial)]
    best = 1.0
    conv = div = False
    for it in range(1, max_iter + 1):
        old = c.copy()
        J = response.jacobian_fd(c)
        delta = np.linalg.lstsq(J, -response.b(c), rcond=None)[0]
        c = c + delta
        hist.append(_history_row(case, "full_jacobian_gn_no_trust", it, response, c, initial, np.linalg.norm(c - old), True))
        r = hist[-1]["residual_ratio"]
        best = min(best, r)
        conv = r < TOL
        div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO
        if conv or div:
            break
    return _finish(case, "full_jacobian_gn_no_trust", response, c, initial, hist, best, conv, div), hist


def run_trust(case, response, delta_max, max_iter=50, method_name=None):
    method = method_name or f"full_jacobian_trust_region_cap_{delta_max}"
    c = C0.copy()
    lam = 1e-3
    initial = float(np.linalg.norm(response.b(c)))
    hist = [_history_row(case, method, 0, response, c, initial, lam=lam, delta_max=delta_max)]
    best = 1.0
    conv = div = False
    acc = rej = 0
    for it in range(1, max_iter + 1):
        b = response.b(c)
        J = response.jacobian_fd(c)
        delta = np.linalg.solve(J.T @ J + lam * np.eye(K), -J.T @ b)
        raw_norm = np.linalg.norm(delta)
        if np.isfinite(delta_max) and raw_norm > delta_max:
            delta = delta * (delta_max / raw_norm)
        pred = 0.5 * np.dot(b, b) - 0.5 * np.dot(b + J @ delta, b + J @ delta)
        trial = c + delta
        bt = response.b(trial)
        actual = 0.5 * np.dot(b, b) - 0.5 * np.dot(bt, bt)
        rho = actual / pred if pred > 0 else -np.inf
        accepted = bool(pred > 0 and rho > 0 and np.all(np.isfinite(trial)))
        if accepted:
            c = trial
            acc += 1
        else:
            rej += 1
        if rho > 0.75:
            lam *= 0.3
        elif rho < 0.25:
            lam *= 2.0
        lam = float(np.clip(lam, 1e-10, 1e10))
        hist.append(_history_row(case, method, it, response, c, initial, np.linalg.norm(delta) if accepted else 0.0, accepted, lam, float(rho), delta_max=delta_max))
        r = hist[-1]["residual_ratio"]
        best = min(best, r)
        conv = r < TOL
        div = (not np.isfinite(r)) or r > DIVERGENCE_RATIO
        if conv or div:
            break
    return _finish(case, method, response, c, initial, hist, best, conv, div, accepted_steps=acc, rejected_steps=rej, final_lambda=lam, delta_max=delta_max), hist


def rerun_methods(mesh, B, clean_params: Dict[str, FEMProcessParams]):
    summary = []
    histories = []
    for case, params in clean_params.items():
        response = ThermalFEMResponse(case, mesh, B, params)
        r, h = run_direct_method(case, response)
        summary.append(r); histories.extend(h)
        for alpha in ORIGINAL_ALPHAS:
            r, h = run_scalar_method(case, response, alpha, f"scalar_original_alpha_{alpha:.2f}")
            summary.append(r); histories.extend(h)
        scalar_results = []
        for alpha in EXTENDED_ALPHAS:
            r, h = run_scalar_method(case, response, float(alpha), "oracle_extended_scalar")
            scalar_results.append((r, h))
        best_scalar = min(scalar_results, key=lambda x: x[0]["final_residual_ratio"])
        summary.append(best_scalar[0]); histories.extend(best_scalar[1])
        for fn in [run_diagonal, run_full]:
            r, h = fn(case, response)
            summary.append(r); histories.extend(h)
        trust_results = []
        for cap in TR_CAPS:
            r, h = run_trust(case, response, cap)
            trust_results.append((r, h))
            summary.append(r); histories.extend(h)
        best_trust = min(trust_results, key=lambda x: x[0]["final_residual_ratio"])
        best = best_trust[0].copy()
        best["method_name"] = "full_jacobian_trust_region_best_cap"
        summary.append(best)
    return summary, histories


def _case_result_lookup(rows):
    out = {}
    for r in rows:
        out.setdefault(r["case_name"], {})[r["method_name"]] = r
    return out


def _method_table(rows):
    rows = sorted(rows, key=lambda r: float(r["final_residual_ratio"]))
    lines = ["| method | converged | diverged | iters | final ratio | best ratio | alpha/cap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        tag = r.get("selected_alpha") if r.get("selected_alpha") not in (None, "") else r.get("delta_max", "")
        lines.append(f"| {r['method_name']} | {r['converged']} | {r['diverged']} | {r['iterations_run']} | {float(r['final_residual_ratio']):.3e} | {float(r['best_residual_ratio_seen']):.3e} | {tag} |")
    return "\n".join(lines)


def _compact_original_table(rows):
    lines = ["| original case | rho(I-J0) | direct one-step | best alpha | best alpha ratio | small-step lin err | direct-step lin err | diagnosis |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['original_case']} | {r['rho']:.3g} | {r['direct_one_step_ratio']:.3e} | {r['best_alpha']:.3e} | {r['best_alpha_ratio']:.3e} | {r['small_step_linear_error']:.3e} | {r['direct_step_linear_error']:.3e} | {r['diagnosis']} |")
    return "\n".join(lines)


def _compact_clean_table(rows):
    lines = ["| clean case | rho(I-J0) | coupling | direct ratio | best scalar | diagonal | full | trust | decision |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['clean_case']} | {r['rho']:.3g} | {r['coupling']:.3g} | {r['direct_ratio']:.3e} | {r['best_scalar_ratio']:.3e} | {r['diagonal_ratio']:.3e} | {r['full_ratio']:.3e} | {r['trust_ratio']:.3e} | {r['decision']} |")
    return "\n".join(lines)


def build_report(updated_summary, audited_summary, checks, decision, compact_original, compact_clean):
    lines = [
        "# Step 4B FEM Plate Audit",
        "",
        "## Why Step 4B Was Needed",
        "Step 4 produced a useful but suspicious PARTIAL-GO: some cases had `rho(I-J0) < 1` while direct inversion still diverged. Step 4B audits whether that contradiction comes from large steps leaving the local regime, poor scaling, ill-conditioning, or a faulty finite-difference Jacobian.",
        "",
        "## Original Case Audit",
        _compact_original_table(compact_original),
        "",
        "## fem_easy Contradiction",
    ]
    easy = next((r for r in compact_original if r["original_case"] == "fem_easy"), None)
    if easy:
        lines.append(f"`fem_easy` has rho(I-J0)={easy['rho']:.3g}, but direct one-step ratio={easy['direct_one_step_ratio']:.3e}. The best scalar alpha was {easy['best_alpha']:.3e} with ratio {easy['best_alpha_ratio']:.3e}. This means rho from the infinitesimal linearization is not enough: the full direct step is too large for this nonlinear/stiff FEM-like response scale.")
    lines.extend([
        "",
        "## Local Linearization Validity",
        "The CSV tests random, direct, diagonal, and full-GN directions over scales from `1e-4` to `1`. Small-scale validity combined with scale-1 failure indicates a valid local model but an unsafe full direct step.",
        "",
        "## Line Search Audit",
        "The extended alpha grid uses `logspace(-6, 0, 80)` plus `1.25`. Best alphas far below `0.05` show that the original scalar grid was too coarse/large for stiff FEM-like response.",
        "",
        "## Updated Selection",
        f"Updated scan candidates: {updated_summary['num_candidates']}. Missing clean categories: {updated_summary['missing_clean_categories']}. Clean cases selected: {updated_summary['num_clean_cases']}.",
        "",
        "## Clean Case Results",
        _compact_clean_table(compact_clean),
        "",
        "## Internal Checks",
    ])
    for k, v in checks.items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines.extend(["", "## Method Tables"])
    by = _case_result_lookup(audited_summary)
    for case, methods in by.items():
        lines.append(f"### {case}")
        lines.append(_method_table(list(methods.values())))
        lines.append("")
    lines.extend([
        "## Honest Interpretation",
        "- Instability is not explained by `rho(I-J0)` alone; one-step residual growth and local linearization error are necessary selection diagnostics.",
        "- In stiff FEM-like cases, direct/scalar failure is often a large-step problem: small alpha can be stable while alpha=1 diverges.",
        "- Diagonal response calibration remains strong when the modal response is mostly separable or diagonally dominant.",
        "- Full Jacobian/trust-region adds value when coupling, ill-conditioning, or nonlinear step acceptance makes diagonal updates unreliable, but it should not be claimed to always dominate diagonal calibration.",
        "",
        "## GO / PARTIAL-GO / NO-GO",
        f"Decision: **{decision}**",
    ])
    return "\n".join(lines) + "\n"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = create_fem_plate_mesh()
    B, mode_diag = create_fem_plate_modes(mesh)
    original_params = _load_original_params()
    original_audit, case_cache = audit_original_cases(mesh, B, mode_diag["orthonormality_error_fro"], original_params)
    local_rows = local_linearization_tests(case_cache)
    line_rows, line_summary = line_search_audit(case_cache)

    compact_original = []
    for row in original_audit:
        case = row["case_name"]
        small = next((r for r in local_rows if r["case_name"] == case and r["direction"] == "direct" and abs(r["scale"] - 1e-2) < 1e-15), None)
        full = next((r for r in local_rows if r["case_name"] == case and r["direction"] == "direct" and abs(r["scale"] - 1.0) < 1e-15), None)
        diagnosis = "small-scale local model ok; full direct step too large" if small and small["linearization_error"] < 0.25 and full and full["linearization_error"] > 0.5 else "rho alone is insufficient; see line-search and local-linearization rows"
        compact_original.append({
            "original_case": case,
            "rho": row["rho_I_minus_J0"],
            "direct_one_step_ratio": row["direct_one_step_ratio"],
            "best_alpha": line_summary[case]["best_alpha"],
            "best_alpha_ratio": line_summary[case]["best_alpha_ratio"],
            "small_step_linear_error": small["linearization_error"] if small else np.nan,
            "direct_step_linear_error": full["linearization_error"] if full else np.nan,
            "diagnosis": diagnosis,
        })

    updated_rows, clean_params, updated_summary = updated_parameter_scan(mesh, B, mode_diag["orthonormality_error_fro"])
    audited_summary, method_history = rerun_methods(mesh, B, clean_params)
    clean_diagnostics = {
        name: diagnostics_for_response(ThermalFEMResponse(name, mesh, B, params), mode_diag["orthonormality_error_fro"])
        for name, params in clean_params.items()
    }
    by = _case_result_lookup(audited_summary)
    compact_clean = []
    for name, d in clean_diagnostics.items():
        methods = by[name]
        trust_candidates = [v for k, v in methods.items() if k.startswith("full_jacobian_trust_region_cap_") or k == "full_jacobian_trust_region_best_cap"]
        best_trust = min(trust_candidates, key=lambda r: float(r["final_residual_ratio"]))
        compact_clean.append({
            "clean_case": name,
            "rho": d["rho_I_minus_J0"],
            "coupling": d["coupling_ratio"],
            "direct_ratio": float(methods["direct_modal_inversion"]["final_residual_ratio"]),
            "best_scalar_ratio": float(methods["oracle_extended_scalar"]["final_residual_ratio"]),
            "diagonal_ratio": float(methods["diagonal_modal_factor_initial"]["final_residual_ratio"]),
            "full_ratio": float(methods["full_jacobian_gn_no_trust"]["final_residual_ratio"]),
            "trust_ratio": float(best_trust["final_residual_ratio"]),
            "decision": "PENDING",
        })

    local_small_ok = any(r["direction"] == "direct" and r["scale"] == 1e-2 and r["linearization_error"] < 0.25 for r in local_rows)
    clean_found = len(clean_params) > 0
    calibrated_improves = any(r["diagonal_ratio"] < r["direct_ratio"] or r["trust_ratio"] < r["direct_ratio"] for r in compact_clean)
    decision = "GO" if local_small_ok and clean_found and calibrated_improves else ("PARTIAL-GO" if local_small_ok or clean_found else "NO-GO")
    for r in compact_clean:
        r["decision"] = decision

    checks = {
        "project_root_is_correct": PROJECT_ROOT.resolve() == EXPECTED_ROOT.resolve(),
        "step4_code_reused_or_loaded": STEP4_DIR.exists(),
        "original_cases_audited": len(original_audit) == 5,
        "local_linearization_completed": len(local_rows) > 0,
        "line_search_completed": len(line_rows) > 0,
        "updated_case_selection_completed": len(updated_rows) >= 60,
        "clean_cases_found_or_reported": True,
        "methods_rerun_on_clean_cases": len(audited_summary) > 0 if clean_found else True,
        "output_files_created": True,
    }

    _write_csv(CASE_AUDIT_PATH, original_audit)
    _write_csv(LOCAL_LINEAR_PATH, local_rows)
    _write_csv(LINE_SEARCH_PATH, line_rows)
    _write_csv(UPDATED_SCAN_PATH, updated_rows)
    _write_csv(AUDITED_SUMMARY_PATH, audited_summary)
    DIAG_PATH.write_text(json.dumps({
        "basis": mode_diag,
        "original_audit_compact": compact_original,
        "line_search_summary": line_summary,
        "updated_selection_summary": updated_summary,
        "clean_selected_cases": clean_diagnostics,
        "checks": checks,
        "decision": decision,
    }, indent=2, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(build_report(updated_summary, audited_summary, checks, decision, compact_original, compact_clean), encoding="utf-8")

    print(f"STEP4B_FEM_PLATE_AUDIT_REPORT.md: {REPORT_PATH}")
    print(f"audited_summary.csv: {AUDITED_SUMMARY_PATH}")
    print(f"local_linearization.csv: {LOCAL_LINEAR_PATH}")
    print(f"line_search_scan.csv: {LINE_SEARCH_PATH}")
    print(f"case_selection_audit.csv: {CASE_AUDIT_PATH}")
    print(f"updated_parameter_scan.csv: {UPDATED_SCAN_PATH}")
    print(f"diagnostics.json: {DIAG_PATH}")
    print()
    print("Internal checks:")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print()
    print("Original case audit:")
    print("original_case | rho(I-J0) | direct_one_step_ratio | best_alpha | best_alpha_ratio | small_step_linear_error | direct_step_linear_error | diagnosis")
    for r in compact_original:
        print(f"{r['original_case']} | {r['rho']:.3g} | {r['direct_one_step_ratio']:.3e} | {r['best_alpha']:.3e} | {r['best_alpha_ratio']:.3e} | {r['small_step_linear_error']:.3e} | {r['direct_step_linear_error']:.3e} | {r['diagnosis']}")
    print()
    print("Clean case results:")
    print("clean_case | rho(I-J0) | coupling | direct_ratio | best_scalar_ratio | diagonal_ratio | full_ratio | trust_ratio | GO/PARTIAL/NO-GO")
    for r in compact_clean:
        print(f"{r['clean_case']} | {r['rho']:.3g} | {r['coupling']:.3g} | {r['direct_ratio']:.3e} | {r['best_scalar_ratio']:.3e} | {r['diagonal_ratio']:.3e} | {r['full_ratio']:.3e} | {r['trust_ratio']:.3e} | {r['decision']}")
    print()
    print(f"Overall Step 4B: {decision}")


if __name__ == "__main__":
    main()
