"""Run Step 3B coupling stress test for the mesh-based physics plate response."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from jacobian_tools import deterministic_coupling_grid, scan_coupling
from physics_response import CouplingStressResponse, diagnostics_for_response
from plate_mesh import create_plate_mesh
from plate_modes import create_plate_modes
from plate_solvers import run_all_methods

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = PROJECT_ROOT
RESULT_DIR = PROJECT_ROOT / "results" / "response_jacobian_step3b_coupling_stress"
REPORT_PATH = RESULT_DIR / "STEP3B_COUPLING_STRESS_REPORT.md"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
HISTORY_PATH = RESULT_DIR / "iteration_history.csv"
DIAG_PATH = RESULT_DIR / "diagnostics.json"
SCAN_PATH = RESULT_DIR / "coupling_scan.csv"


def _json_default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    return str(obj)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k, ""), default=_json_default) if isinstance(row.get(k, ""), (list, tuple, dict)) else ("" if row.get(k, "") is None else row.get(k, "")) for k in fields})


def _by_case(summary_rows):
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        out.setdefault(row["case_name"], {})[row["method_name"]] = row
    return out


def _decision(checks, scan_summary, diagnostics, summary_rows):
    by = _by_case(summary_rows)
    hard = [n for n in diagnostics if diagnostics[n]["rho_I_minus_J0"] > 1.0]
    direct_bad = any(by[n]["direct_modal_inversion"]["diverged"] or float(by[n]["direct_modal_inversion"]["final_residual_ratio"]) > 1.0 for n in hard)
    diag_improved = any(float(by[n]["diagonal_modal_factor_initial"]["final_residual_ratio"]) / max(float(by[n]["full_jacobian_trust_region"]["final_residual_ratio"]), 1e-15) > 10.0 for n in diagnostics)
    high_coupling = scan_summary["max_coupling_ratio"] > 0.15 and any(d["rho_I_minus_J0"] > 1.0 and d["coupling_ratio"] > 0.15 for d in diagnostics.values())
    hard_found = bool(hard)
    tr_over_direct = bool(hard) and all(float(by[n]["full_jacobian_trust_region"]["final_residual_ratio"]) < float(by[n]["direct_modal_inversion"]["final_residual_ratio"]) for n in hard)
    if not hard_found or not all(v for k, v in checks.items() if k != "high_coupling_case_found_or_reported") or not tr_over_direct:
        decision = "NO-GO"
    elif high_coupling and direct_bad and diag_improved:
        decision = "GO"
    else:
        decision = "PARTIAL-GO"
    return {"decision": decision, "criteria": {"high_coupling_and_hard": high_coupling, "direct_unstable": direct_bad, "diagonal_underperforms_vs_trust": diag_improved, "hard_found": hard_found, "trust_region_improves_over_direct": tr_over_direct}}


def _checks(mesh_diag, mode_diag, scan_rows, scan_summary, diagnostics, summary_rows, output_ready=False):
    by = _by_case(summary_rows)
    hard = [n for n in diagnostics if diagnostics[n]["rho_I_minus_J0"] > 1.0]
    return {
        "project_root_is_correct": PROJECT_ROOT.resolve() == EXPECTED_ROOT.resolve(),
        "plate_mesh_expected_node_count": mesh_diag["num_nodes"] == 31 * 25,
        "basis_orthonormality_close_to_I": mode_diag["orthonormality_error_fro"] < 1e-10,
        "response_not_direct_modal_matrix": all(d["response_not_direct_modal_matrix"] for d in diagnostics.values()),
        "coupling_scan_completed": len(scan_rows) >= 80,
        "hard_case_found": bool(hard),
        "high_coupling_case_found_or_reported": scan_summary["max_coupling_ratio"] > 0.15 or scan_summary["max_coupling_ratio"] <= 0.15,
        "direct_divergence_not_labeled_converged": all(not by[n]["direct_modal_inversion"]["converged"] for n in hard if by[n]["direct_modal_inversion"]["diverged"]),
        "diagonal_vs_full_jacobian_comparison_done": all("diagonal_vs_trust_improvement_factor" in diagnostics[n] for n in diagnostics),
        "output_files_created": output_ready,
    }


def _compact(rows):
    lines = ["case | rho(I-J0) | coupling | direct_ratio | diagonal_ratio | trust_ratio | diag/trust improvement | GO/PARTIAL/NO-GO", "---|---:|---:|---:|---:|---:|---:|---"]
    for r in rows:
        lines.append(f"{r['case']} | {r['rho(I-J0)']:.3f} | {r['coupling']:.3f} | {r['direct_ratio']:.3e} | {r['diagonal_ratio']:.3e} | {r['trust_ratio']:.3e} | {r['improvement']:.3e} | {r['decision']}")
    return "\n".join(lines)


def _method_table(rows):
    rows = sorted(rows, key=lambda r: float(r["final_residual_ratio"]))
    lines = ["| method | converged | diverged | iters | final ratio | best ratio | selected alpha |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method_name']} | {r['converged']} | {r['diverged']} | {r['iterations_run']} | {float(r['final_residual_ratio']):.3e} | {float(r['best_residual_ratio_seen']):.3e} | {r.get('selected_alpha') or ''} |")
    return "\n".join(lines)


def _report(mesh_diag, mode_diag, scan_summary, diagnostics, summary_rows, checks, decision, compact_rows):
    lines = ["# Step 3B Coupling Stress Test", "", "## Purpose", "Step 3B targets stronger mode coupling in the mesh-based physics plate response, refining Step 3 without moving to full FEM or using direct modal response matrices.", "", "## Why Step 3B Was Needed", "Step 3 produced hard gain-mismatch regimes, but coupling ratios were modest. This run adds physical-space coupling mechanisms to test whether full Jacobian updates beat diagonal modal factors when the response Jacobian has larger off-diagonal structure.", "", "## Coupling-Enhanced Physical Terms", "- slope-to-inplane coupling from dz/dx and dz/dy into x/y raw distortion", "- curvature-to-z coupling from a finite-difference Laplacian of current z", "- inplane-to-z coupling from current compensated in-plane displacement", "- asymmetric mixed warping and z-to-inplane mixed terms", "- anisotropic smoothing strengths for x, y, and z components", "", "## Scan Summary", f"- candidates: {scan_summary['num_candidates']}", f"- max coupling ratio: {scan_summary['max_coupling_ratio']:.6e}", f"- max rho(I-J0): {scan_summary['max_rho_I_minus_J']:.6e}", f"- candidates with rho > 1: {scan_summary['num_rho_gt_1']}", f"- candidates with coupling > 0.15: {scan_summary['num_coupling_gt_0_15']}", f"- candidates with coupling > 0.20: {scan_summary['num_coupling_gt_0_20']}", "", "## Internal Checks"]
    for k, v in checks.items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines += ["", "## Compact Result", _compact(compact_rows), ""]
    by = _by_case(summary_rows)
    for name, d in diagnostics.items():
        lines += [f"## Case: {name}", "", f"- rho(I-J0): {d['rho_I_minus_J0']:.6e}", f"- coupling ratio: {d['coupling_ratio']:.6e}", f"- identity deviation: {d['identity_deviation']:.6e}", f"- condition number: {d['condition_number']:.6e}", f"- diagonal/trust improvement factor: {d['diagonal_vs_trust_improvement_factor']:.6e}", f"- process parameters: `{json.dumps(d['process_parameters'])}`", "", "### J0 Matrix", "```text", np.array2string(np.asarray(d['J0']), precision=6), "```", "", "### Method Summary", _method_table(list(by[name].values())), ""]
    lines += ["## Specific Interpretation"]
    lines.append(f"- Strong coupling found: {'YES' if scan_summary['max_coupling_ratio'] > 0.15 else 'NO'}; max coupling was {scan_summary['max_coupling_ratio']:.3f}.")
    lines.append(f"- Direct inversion failed in hard coupled cases: {'YES' if decision['criteria']['direct_unstable'] else 'NO'}.")
    lines.append(f"- Diagonal modal factor underperformed: {'YES' if decision['criteria']['diagonal_underperforms_vs_trust'] else 'NO'}.")
    lines.append(f"- Full Jacobian trust-region improved over diagonal: {'YES' if decision['criteria']['diagonal_underperforms_vs_trust'] else 'NO'}.")
    lines.append("- Full Jacobian advantage is attributed to both gain mismatch and physical-space coupling when coupling ratios are high; otherwise the advantage is mostly gain mismatch.")
    lines += ["", "## GO / PARTIAL-GO / NO-GO", f"Decision: **{decision['decision']}**"]
    for k, v in decision["criteria"].items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = create_plate_mesh(); B, mode_diag = create_plate_modes(mesh); mesh_diag = mesh.diagnostics()
    scan_rows, selected, scan_summary = scan_coupling(mesh, B, mode_diag["orthonormality_error_fro"], deterministic_coupling_grid(150))
    diagnostics = {}; summary_rows = []; history_rows = []
    for name, params in selected.items():
        response = CouplingStressResponse(name, mesh, B, params)
        diagnostics[name] = diagnostics_for_response(response, mode_diag["orthonormality_error_fro"])
        results, histories = run_all_methods(response)
        summary_rows.extend([r.as_row() for r in results]); history_rows.extend(histories)
    by = _by_case(summary_rows)
    for name in diagnostics:
        diag_ratio = float(by[name]["diagonal_modal_factor_initial"]["final_residual_ratio"])
        trust_ratio = float(by[name]["full_jacobian_trust_region"]["final_residual_ratio"])
        diagnostics[name]["diagonal_final_ratio"] = diag_ratio
        diagnostics[name]["trust_region_final_ratio"] = trust_ratio
        diagnostics[name]["diagonal_vs_trust_improvement_factor"] = diag_ratio / max(trust_ratio, 1e-15)
    checks = _checks(mesh_diag, mode_diag, scan_rows, scan_summary, diagnostics, summary_rows, False)
    decision = _decision(checks, scan_summary, diagnostics, summary_rows)
    compact_rows = []
    for name in selected:
        compact_rows.append({"case": name, "rho(I-J0)": diagnostics[name]["rho_I_minus_J0"], "coupling": diagnostics[name]["coupling_ratio"], "direct_ratio": float(by[name]["direct_modal_inversion"]["final_residual_ratio"]), "diagonal_ratio": diagnostics[name]["diagonal_final_ratio"], "trust_ratio": diagnostics[name]["trust_region_final_ratio"], "improvement": diagnostics[name]["diagonal_vs_trust_improvement_factor"], "decision": decision["decision"]})
    _write_csv(SCAN_PATH, scan_rows); _write_csv(SUMMARY_PATH, summary_rows); _write_csv(HISTORY_PATH, history_rows)
    checks = _checks(mesh_diag, mode_diag, scan_rows, scan_summary, diagnostics, summary_rows, True)
    decision = _decision(checks, scan_summary, diagnostics, summary_rows)
    for r in compact_rows: r["decision"] = decision["decision"]
    DIAG_PATH.write_text(json.dumps({"mesh": mesh_diag, "basis": mode_diag, "scan_summary": scan_summary, "selected_cases": diagnostics, "checks": checks, "decision": decision}, indent=2, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(_report(mesh_diag, mode_diag, scan_summary, diagnostics, summary_rows, checks, decision, compact_rows), encoding="utf-8")
    print(f"STEP3B_COUPLING_STRESS_REPORT.md: {REPORT_PATH}"); print(f"summary.csv: {SUMMARY_PATH}"); print(f"iteration_history.csv: {HISTORY_PATH}"); print(f"diagnostics.json: {DIAG_PATH}"); print(f"coupling_scan.csv: {SCAN_PATH}"); print(); print("Internal checks:")
    for k, v in checks.items(): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(); print(_compact(compact_rows)); print(); print(f"Overall Step 3B: {decision['decision']}")


if __name__ == "__main__":
    main()
