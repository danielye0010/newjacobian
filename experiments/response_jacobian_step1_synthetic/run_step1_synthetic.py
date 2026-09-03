"""Run Step 1 synthetic response-Jacobian diagnostics and write report outputs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from response_models import C0, C_STAR, K, case_diagnostics, format_matrix, make_cases, nonlinear_h
from solvers import run_all_methods

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "response_jacobian_step1_synthetic"
REPORT_PATH = RESULT_DIR / "STEP1_SYNTHETIC_REPORT.md"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
HISTORY_PATH = RESULT_DIR / "iteration_history.csv"
DIAG_PATH = RESULT_DIR / "diagnostics.json"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bool):
        return obj
    return str(obj)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, dict)):
                    normalized[key] = json.dumps(value, default=_json_default)
                elif value is None:
                    normalized[key] = ""
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _method_table(rows: List[Dict[str, object]]) -> str:
    sorted_rows = sorted(rows, key=lambda r: float(r["final_residual_ratio"]))
    header = "| method | converged | diverged | iters | final ratio | best ratio | error ||\n|---|---:|---:|---:|---:|---:|---:|"
    lines = [header]
    for r in sorted_rows:
        lines.append(
            f"| {r['method_name']} | {r['converged']} | {r['diverged']} | {r['iterations_run']} | "
            f"{float(r['final_residual_ratio']):.3e} | {float(r['best_residual_ratio_seen']):.3e} | "
            f"{float(r['error_to_c_star_norm']):.3e} |"
        )
    return "\n".join(lines)


def _compact_table(compact_rows: List[Dict[str, object]]) -> str:
    header = (
        "case | rho(I-J0) | coupling | best_method | direct_final_ratio | "
        "oracle_scalar_final_ratio | diagonal_final_ratio | trust_region_final_ratio | GO/NO-GO"
    )
    sep = "---|---:|---:|---|---:|---:|---:|---:|---"
    lines = [header, sep]
    for r in compact_rows:
        lines.append(
            f"{r['case']} | {r['rho(I-J0)']:.3f} | {r['coupling']:.3f} | {r['best_method']} | "
            f"{r['direct_final_ratio']:.3e} | {r['oracle_scalar_final_ratio']:.3e} | "
            f"{r['diagonal_final_ratio']:.3e} | {r['trust_region_final_ratio']:.3e} | {r['go_no_go']}"
        )
    return "\n".join(lines)


def _interpret(case_rows: List[Dict[str, object]], diagnostics: Dict[str, Dict[str, object]]) -> List[str]:
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in case_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    lines = []
    hard_cases = ["gain_mismatch_linear", "coupled_linear", "weak_nonlinear_coupled", "strong_nonlinear_coupled"]
    direct_failed = any(by_case[c]["direct_modal_inversion"]["diverged"] or by_case[c]["direct_modal_inversion"]["final_residual_ratio"] > 1.0 for c in hard_cases)
    lines.append(f"- Direct modal inversion failed or worsened in hard cases: {'YES' if direct_failed else 'NO'}.")
    scalar_sensitive = False
    for c, methods in by_case.items():
        scalar_ratios = [float(v["final_residual_ratio"]) for k, v in methods.items() if k.startswith("scalar_alpha_")]
        finite = [x for x in scalar_ratios if math.isfinite(x)]
        if finite and max(finite) / max(min(finite), 1e-15) > 1e3:
            scalar_sensitive = True
    lines.append(f"- Fixed scalar scale factor was sensitive to alpha: {'YES' if scalar_sensitive else 'NO'}.")
    oracle_helped = any(
        float(methods["oracle_scalar_scale_factor"]["final_residual_ratio"]) < min(float(v["final_residual_ratio"]) for k, v in methods.items() if k.startswith("scalar_alpha_")) * 1.01
        for methods in by_case.values()
    )
    lines.append(f"- Oracle scalar improved over fixed manual choices or matched the best grid value: {'YES' if oracle_helped else 'NO'}.")
    diagonal_underperformed = any(
        diagnostics[c]["coupling_ratio"] > 0.2 and float(methods["diagonal_modal_factor_initial"]["final_residual_ratio"]) > 10.0 * float(methods["full_jacobian_trust_region"]["final_residual_ratio"])
        for c, methods in by_case.items()
    )
    lines.append(f"- Diagonal factor underperformed when coupling was large: {'YES' if diagonal_underperformed else 'NO'}.")
    full_best = any(
        methods[min(methods, key=lambda m: float(methods[m]["final_residual_ratio"]))]["method_name"].startswith("full_jacobian")
        for methods in by_case.values()
    )
    lines.append(f"- Full Jacobian GN or trust-region was among the best performers: {'YES' if full_best else 'NO'}.")
    tr_rejected = any(int(by_case[c]["full_jacobian_trust_region"].get("rejected_steps") or 0) > 0 for c in ["weak_nonlinear_coupled", "strong_nonlinear_coupled"])
    lines.append(f"- Trust-region had rejected steps in nonlinear cases: {'YES' if tr_rejected else 'NO'}.")
    return lines


def _go_no_go(summary_rows: List[Dict[str, object]], diagnostics: Dict[str, Dict[str, object]], checks: Dict[str, bool]) -> Dict[str, object]:
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    hard_direct_fails = all(
        by_case[c]["direct_modal_inversion"]["diverged"] or float(by_case[c]["direct_modal_inversion"]["final_residual_ratio"]) > 1.0
        for c in ["gain_mismatch_linear", "coupled_linear"]
    )
    scalar_sensitive = any(
        max(float(v["final_residual_ratio"]) for k, v in methods.items() if k.startswith("scalar_alpha_")) /
        max(min(float(v["final_residual_ratio"]) for k, v in methods.items() if k.startswith("scalar_alpha_")), 1e-15) > 1e3
        for methods in by_case.values()
    )
    full_better = all(
        min(float(methods[m]["final_residual_ratio"]) for m in ["full_jacobian_gn_no_trust", "full_jacobian_trust_region"]) <
        min(float(methods["direct_modal_inversion"]["final_residual_ratio"]), float(methods["oracle_scalar_scale_factor"]["final_residual_ratio"]))
        for methods in by_case.values()
    )
    jac_ok = all(
        d["jacobian_check_c0"]["relative_frobenius_error"] < 1e-7 and d["jacobian_check_random_seed_123"]["relative_frobenius_error"] < 1e-7
        for d in diagnostics.values()
    )
    nonlinear_ok = all(
        float(by_case[c]["full_jacobian_trust_region"]["final_residual_ratio"]) < 1e-6
        for c in ["weak_nonlinear_coupled", "strong_nonlinear_coupled"]
    )
    criteria = {
        "hard_direct_fails": hard_direct_fails,
        "scalar_sensitive": scalar_sensitive,
        "full_jacobian_better": full_better,
        "jacobian_checks_ok": jac_ok,
        "internal_checks_ok": all(checks.values()),
        "nonlinear_trust_region_ok": nonlinear_ok,
    }
    return {"decision": "GO" if all(criteria.values()) else "NO-GO", "criteria": criteria}


def _internal_checks(cases, diagnostics) -> Dict[str, bool]:
    checks = {
        "h_zero_is_zero": bool(np.allclose(nonlinear_h(np.zeros(K)), np.zeros(K))),
        "b_c_star_is_zero": all(np.linalg.norm(model.b(C_STAR)) < 1e-12 for model in cases.values()),
        "jacobian_checks_within_tolerance": all(
            d["jacobian_check_c0"]["relative_frobenius_error"] < 1e-7 and d["jacobian_check_random_seed_123"]["relative_frobenius_error"] < 1e-7
            for d in diagnostics.values()
        ),
    }
    return checks


def build_report(summary_rows, diagnostics, checks, go_info, compact_rows) -> str:
    lines = [
        "# Step 1 Synthetic Response-Jacobian Diagnostic",
        "",
        "## Purpose",
        "This controlled low-dimensional test verifies the response-Jacobian theory before any FEM, JAX, or modal mesh implementation. Direct modal inversion assumes `J = I`; these cases deliberately introduce gain mismatch, coupling, and nonlinear response so that the assumption can be diagnosed in isolation.",
        "",
        "## Synthetic Model",
        "The modal residual is `b(c) = A (c - c_star) + beta h(c - c_star)`, with `k = 5`, `c_star = [1.0, -0.8, 0.6, -0.4, 0.3]`, and `c0 = 0`. The nonlinear term satisfies `h(0) = 0`, so the exact solution remains `c = c_star` for every case.",
        "",
        "## Why Before FEM",
        "This step isolates response modeling from mesh discretization, FEM solver details, basis construction, and numerical noise. If the Jacobian idea does not win in this synthetic setting, it should not be trusted in a more expensive FEM loop.",
        "",
        "## Cases",
        "- `gain_mismatch_linear`: diagonal gains `[0.3, 0.8, 1.2, 2.5, 3.0]`, `beta = 0`.",
        "- `coupled_linear`: orthogonally coupled eigenvalues `[0.4, 0.9, 1.5, 2.6, 3.2]`, `beta = 0`.",
        "- `weak_nonlinear_coupled`: same coupled `A`, `beta = 0.15`.",
        "- `strong_nonlinear_coupled`: same coupled `A`, `beta = 0.50`.",
        "",
        "## Internal Checks",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.extend(["", "## Compact Result", _compact_table(compact_rows), ""])
    for case_name, d in diagnostics.items():
        rows = [r for r in summary_rows if r["case_name"] == case_name]
        eig_text = ", ".join(f"{x[0]:.6g}{'+' if x[1] >= 0 else ''}{x[1]:.6g}j" for x in d["eigenvalues"])
        lines.extend([
            f"## Case: {case_name}",
            "",
            "### J0 Matrix",
            "```text",
            format_matrix(np.asarray(d["J0"])),
            "```",
            f"- eigenvalues of J0: {eig_text}",
            f"- rho(I - J0): {d['rho_I_minus_J0']:.6e}",
            f"- identity deviation: {d['identity_deviation']:.6e}",
            f"- coupling ratio: {d['coupling_ratio']:.6e}",
            f"- condition number: {d['condition_number']:.6e}",
            f"- Jacobian check at c0: max abs {d['jacobian_check_c0']['max_abs_error']:.3e}, relative Frobenius {d['jacobian_check_c0']['relative_frobenius_error']:.3e}",
            f"- Jacobian check at random c seed 123: max abs {d['jacobian_check_random_seed_123']['max_abs_error']:.3e}, relative Frobenius {d['jacobian_check_random_seed_123']['relative_frobenius_error']:.3e}",
            "",
            "### Method Summary",
            _method_table(rows),
            "",
        ])
    lines.extend(["## Automatic Interpretation", *_interpret(summary_rows, diagnostics), "", "## GO / NO-GO", f"Decision: **{go_info['decision']}**", ""])
    for key, value in go_info["criteria"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    diagnostics = {name: case_diagnostics(model, C0) for name, model in cases.items()}
    checks = _internal_checks(cases, diagnostics)
    if not all(checks.values()):
        raise RuntimeError(f"Critical internal checks failed: {checks}")

    summary_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    for model in cases.values():
        results, histories = run_all_methods(model)
        summary_rows.extend([r.as_row() for r in results])
        history_rows.extend(histories)

    gain_full = next(r for r in summary_rows if r["case_name"] == "gain_mismatch_linear" and r["method_name"] == "full_jacobian_gn_no_trust")
    checks["linear_gain_full_jacobian_fast"] = bool(gain_full["iterations_run"] <= 2 and float(gain_full["final_residual_ratio"]) < 1e-10)
    direct_gain = next(r for r in summary_rows if r["case_name"] == "gain_mismatch_linear" and r["method_name"] == "direct_modal_inversion")
    checks["divergent_direct_not_labeled_converged"] = bool(not direct_gain["converged"] and direct_gain["diverged"])

    go_info = _go_no_go(summary_rows, diagnostics, checks)
    for d in diagnostics.values():
        d["J0_matrix_note"] = "Rows and columns use zero-based modal coordinates."
    diagnostics_out = {"cases": diagnostics, "checks": checks, "go_no_go": go_info}

    compact_rows = []
    for case_name in cases.keys():
        rows = {r["method_name"]: r for r in summary_rows if r["case_name"] == case_name}
        best = min(rows.values(), key=lambda r: float(r["final_residual_ratio"]))
        compact_rows.append({
            "case": case_name,
            "rho(I-J0)": float(diagnostics[case_name]["rho_I_minus_J0"]),
            "coupling": float(diagnostics[case_name]["coupling_ratio"]),
            "best_method": best["method_name"],
            "direct_final_ratio": float(rows["direct_modal_inversion"]["final_residual_ratio"]),
            "oracle_scalar_final_ratio": float(rows["oracle_scalar_scale_factor"]["final_residual_ratio"]),
            "diagonal_final_ratio": float(rows["diagonal_modal_factor_initial"]["final_residual_ratio"]),
            "trust_region_final_ratio": float(rows["full_jacobian_trust_region"]["final_residual_ratio"]),
            "go_no_go": go_info["decision"],
        })

    _write_csv(SUMMARY_PATH, summary_rows)
    _write_csv(HISTORY_PATH, history_rows)
    DIAG_PATH.write_text(json.dumps(diagnostics_out, indent=2, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary_rows, diagnostics, checks, go_info, compact_rows), encoding="utf-8")

    print(f"STEP1_SYNTHETIC_REPORT.md: {REPORT_PATH}")
    print(f"summary.csv: {SUMMARY_PATH}")
    print(f"iteration_history.csv: {HISTORY_PATH}")
    print(f"diagnostics.json: {DIAG_PATH}")
    print()
    print(_compact_table(compact_rows))
    print()
    print("Internal checks:")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"Overall Step 1: {go_info['decision']}")


if __name__ == "__main__":
    main()
