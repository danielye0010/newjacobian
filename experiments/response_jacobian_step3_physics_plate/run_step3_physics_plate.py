"""Run Step 3 fixed-connectivity physics-like plate response diagnostic."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from jacobian_tools import deterministic_parameter_grid, scan_parameters
from physics_response import PhysicsPlateResponse, diagnostics_for_response, jax_status
from plate_mesh import create_plate_mesh
from plate_modes import create_plate_modes
from plate_solvers import run_all_methods

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = PROJECT_ROOT
RESULT_DIR = PROJECT_ROOT / "results" / "response_jacobian_step3_physics_plate"
REPORT_PATH = RESULT_DIR / "STEP3_PHYSICS_PLATE_REPORT.md"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
HISTORY_PATH = RESULT_DIR / "iteration_history.csv"
DIAG_PATH = RESULT_DIR / "diagnostics.json"
SCAN_PATH = RESULT_DIR / "parameter_scan.csv"


def _json_default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
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
            out = {}
            for key in fields:
                v = row.get(key, "")
                out[key] = json.dumps(v, default=_json_default) if isinstance(v, (list, tuple, dict)) else ("" if v is None else v)
            writer.writerow(out)


def _compact_table(rows):
    lines = ["case | rho(I-J0) | coupling | direct_ratio | oracle_scalar_ratio | diagonal_ratio | trust_region_ratio | GO/NO-GO",
             "---|---:|---:|---:|---:|---:|---:|---"]
    for r in rows:
        lines.append(f"{r['case']} | {r['rho(I-J0)']:.3f} | {r['coupling']:.3f} | {r['direct_ratio']:.3e} | {r['oracle_scalar_ratio']:.3e} | {r['diagonal_ratio']:.3e} | {r['trust_region_ratio']:.3e} | {r['go_no_go']}")
    return "\n".join(lines)


def _method_table(rows):
    rows = sorted(rows, key=lambda r: float(r["final_residual_ratio"]))
    lines = ["| method | converged | diverged | iters | final ratio | best ratio | selected alpha |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        alpha = r.get("selected_alpha") or ""
        lines.append(f"| {r['method_name']} | {r['converged']} | {r['diverged']} | {r['iterations_run']} | {float(r['final_residual_ratio']):.3e} | {float(r['best_residual_ratio_seen']):.3e} | {alpha} |")
    return "\n".join(lines)


def _checks(mesh_diag, mode_diag, scan_rows, selected, diagnostics, summary_rows, output_ready=False):
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    hard = [n for n in selected if "hard" in n]
    return {
        "project_root_is_correct": bool(PROJECT_ROOT.resolve() == EXPECTED_ROOT.resolve()),
        "plate_mesh_expected_node_count": bool(mesh_diag["num_nodes"] == 31 * 25),
        "basis_orthonormality_close_to_I": bool(mode_diag["orthonormality_error_fro"] < 1e-10),
        "response_not_direct_modal_matrix": all(diagnostics[n]["response_not_direct_modal_matrix"] for n in diagnostics),
        "parameter_scan_completed": bool(len(scan_rows) >= 50),
        "hard_case_found": bool(any(diagnostics[n]["rho_I_minus_J0"] > 1.0 for n in hard)),
        "direct_divergence_not_labeled_converged": all(not by_case[n]["direct_modal_inversion"]["converged"] for n in hard if by_case[n]["direct_modal_inversion"]["diverged"]),
        "trust_region_improves_over_direct_in_hard_cases": bool(hard) and all(float(by_case[n]["full_jacobian_trust_region"]["final_residual_ratio"]) < float(by_case[n]["direct_modal_inversion"]["final_residual_ratio"]) for n in hard),
        "output_files_created": bool(output_ready),
    }


def _go_no_go(checks, diagnostics, summary_rows):
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    hard = [n for n in diagnostics if "hard" in n]
    criteria = {
        "mesh_and_basis_checks_pass": checks["plate_mesh_expected_node_count"] and checks["basis_orthonormality_close_to_I"],
        "hard_case_rho_gt_one": any(diagnostics[n]["rho_I_minus_J0"] > 1.0 for n in hard),
        "mesh_solve_response_not_direct_modal": checks["response_not_direct_modal_matrix"],
        "direct_worse_in_hard_case": any(by_case[n]["direct_modal_inversion"]["diverged"] or float(by_case[n]["direct_modal_inversion"]["final_residual_ratio"]) > 10.0 * float(by_case[n]["full_jacobian_trust_region"]["final_residual_ratio"]) for n in hard),
        "trust_region_strongly_improves": all(float(by_case[n]["full_jacobian_trust_region"]["final_residual_ratio"]) < float(by_case[n]["direct_modal_inversion"]["final_residual_ratio"]) for n in hard),
        "outputs_complete": checks["output_files_created"],
    }
    return {"decision": "GO" if all(criteria.values()) and all(checks.values()) else "NO-GO", "criteria": criteria}


def _interpret(scan_info, diagnostics, summary_rows):
    by_case: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows: by_case.setdefault(row["case_name"], {})[row["method_name"]] = row
    scalar_sensitive = False
    for methods in by_case.values():
        ratios = [float(v["final_residual_ratio"]) for k, v in methods.items() if k.startswith("scalar_alpha_") and math.isfinite(float(v["final_residual_ratio"]))]
        if ratios and max(ratios) / max(min(ratios), 1e-15) > 1e3: scalar_sensitive = True
    hard = [n for n in diagnostics if "hard" in n]
    return [
        f"- Physics-based response produced J != I: {'YES' if any(d['identity_deviation'] > 1e-2 for d in diagnostics.values()) else 'NO'}.",
        f"- Scan found easy/marginal/hard categories: {'YES' if not scan_info['missing_categories'] else 'NO: missing ' + ', '.join(scan_info['missing_categories'])}.",
        f"- Direct inversion failed or was much worse in hard cases: {'YES' if any(by_case[n]['direct_modal_inversion']['diverged'] or float(by_case[n]['direct_modal_inversion']['final_residual_ratio']) > 1.0 for n in hard) else 'NO'}.",
        f"- Scalar factor was sensitive: {'YES' if scalar_sensitive else 'NO'}.",
        f"- Diagonal factor underperformed in coupled cases: {'YES' if any(diagnostics[n]['coupling_ratio'] > 0.15 and float(by_case[n]['diagonal_modal_factor_initial']['final_residual_ratio']) > 10 * float(by_case[n]['full_jacobian_trust_region']['final_residual_ratio']) for n in diagnostics) else 'NO'}.",
        f"- Full Jacobian trust-region remained stable: {'YES' if all(float(by_case[n]['full_jacobian_trust_region']['final_residual_ratio']) < 1e-5 for n in by_case) else 'NO'}.",
    ]


def _report(mesh_diag, mode_diag, scan_rows, scan_info, diagnostics, summary_rows, checks, go_info, compact_rows, jax_info):
    lines = ["# Step 3 Physics Plate Response Diagnostic", "", "## Purpose", "Step 3 replaces Step 2's prescribed modal response with a mesh-based geometry-dependent response solve. Compensation remains modal, `Xc = X0 + B c`, but manufacturing response is computed from current node coordinates, raw distortion, and a fixed graph-Laplacian smoothing solve.", "", "## What Changed From Step 2", "Step 2 used controlled modal residual behavior after projection. Step 3 computes `U = PhysicsSolve(Xc, process_params)`, then `Xp = Xc + U`, `r(c) = Xp - X0`, and `b(c) = B^T M r(c)`. The response is not generated by `q_response = G @ c`.", "", "## Plate Mesh And Basis", f"- nodes: {mesh_diag['num_nodes']}", f"- DOFs: {mesh_diag['num_dofs']}", f"- edges: {mesh_diag['num_edges']}", f"- total area: {mesh_diag['total_area']:.6g}", f"- coordinate bounds: {mesh_diag['min_coordinates']} to {mesh_diag['max_coordinates']}", f"- modal k: {mode_diag['k']}", f"- orthonormality error: {mode_diag['orthonormality_error_fro']:.6e}", "", "## Physics-Like Response Model", "Raw in-plane shrink/shear/z-coupling and out-of-plane warp/z/slope feedback are computed from the current compensated geometry. Each component is relaxed through `(I + smooth_strength L^T L + anchor_strength I_boundary) U = U_raw`, then scaled by `response_gain`.", "", "## Jacobian Computation", f"- finite difference eps: 1e-5", f"- JAX status: {jax_info}", "", "## Parameter Scan", f"- scanned candidates: {len(scan_rows)}", f"- missing categories: {scan_info['missing_categories']}", "", "## Internal Checks"]
    for k, v in checks.items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines += ["", "## Compact Result", _compact_table(compact_rows), ""]
    for name, d in diagnostics.items():
        rows = [r for r in summary_rows if r["case_name"] == name]
        eig = ", ".join(f"{e[0]:.5g}{'+' if e[1] >= 0 else ''}{e[1]:.5g}j" for e in d["eigenvalues"])
        lines += [f"## Case: {name}", "", f"- selected parameters: `{json.dumps(d['process_parameters'])}`", f"- eigenvalues of J0: {eig}", f"- rho(I-J0): {d['rho_I_minus_J0']:.6e}", f"- identity deviation: {d['identity_deviation']:.6e}", f"- coupling ratio: {d['coupling_ratio']:.6e}", f"- condition number: {d['condition_number']:.6e}", f"- initial residual norm: {d['initial_residual_norm']:.6e}", "", "### J0 Matrix", "```text", np.array2string(np.asarray(d['J0']), precision=6), "```", "", "### Method Summary", _method_table(rows), ""]
    lines += ["## Automatic Interpretation", *_interpret(scan_info, diagnostics, summary_rows), "", "## GO / NO-GO", f"Decision: **{go_info['decision']}**"]
    for k, v in go_info["criteria"].items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    if scan_info["missing_categories"]:
        lines += ["", "## Top 10 Closest Candidates", "```json", json.dumps(scan_info["top_10_closest_candidates"], indent=2, default=_json_default), "```"]
    return "\n".join(lines) + "\n"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = create_plate_mesh(); B, mode_diag = create_plate_modes(mesh); mesh_diag = mesh.diagnostics()
    scan_rows, selected, scan_info = scan_parameters(mesh, B, mode_diag["orthonormality_error_fro"], deterministic_parameter_grid(90))
    diagnostics = {}; summary_rows = []; history_rows = []
    for name, params in selected.items():
        response = PhysicsPlateResponse(name, mesh, B, params)
        diagnostics[name] = diagnostics_for_response(response, mode_diag["orthonormality_error_fro"])
        results, histories = run_all_methods(response)
        summary_rows.extend([r.as_row() for r in results]); history_rows.extend(histories)
    checks = _checks(mesh_diag, mode_diag, scan_rows, selected, diagnostics, summary_rows, output_ready=False)
    go_info = _go_no_go(checks, diagnostics, summary_rows)
    compact_rows = []
    for name in selected:
        rows = {r["method_name"]: r for r in summary_rows if r["case_name"] == name}
        compact_rows.append({"case": name, "rho(I-J0)": diagnostics[name]["rho_I_minus_J0"], "coupling": diagnostics[name]["coupling_ratio"], "direct_ratio": float(rows["direct_modal_inversion"]["final_residual_ratio"]), "oracle_scalar_ratio": float(rows["oracle_scalar_scale_factor"]["final_residual_ratio"]), "diagonal_ratio": float(rows["diagonal_modal_factor_initial"]["final_residual_ratio"]), "trust_region_ratio": float(rows["full_jacobian_trust_region"]["final_residual_ratio"]), "go_no_go": go_info["decision"]})
    _write_csv(SCAN_PATH, scan_rows); _write_csv(SUMMARY_PATH, summary_rows); _write_csv(HISTORY_PATH, history_rows)
    checks = _checks(mesh_diag, mode_diag, scan_rows, selected, diagnostics, summary_rows, output_ready=True)
    go_info = _go_no_go(checks, diagnostics, summary_rows)
    for r in compact_rows: r["go_no_go"] = go_info["decision"]
    jax_info = jax_status()
    DIAG_PATH.write_text(json.dumps({"mesh": mesh_diag, "basis": mode_diag, "jax": jax_info, "parameter_scan_summary": scan_info, "selected_cases": diagnostics, "checks": checks, "go_no_go": go_info}, indent=2, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(_report(mesh_diag, mode_diag, scan_rows, scan_info, diagnostics, summary_rows, checks, go_info, compact_rows, jax_info), encoding="utf-8")
    print(f"STEP3_PHYSICS_PLATE_REPORT.md: {REPORT_PATH}"); print(f"summary.csv: {SUMMARY_PATH}"); print(f"iteration_history.csv: {HISTORY_PATH}"); print(f"diagnostics.json: {DIAG_PATH}"); print(f"parameter_scan.csv: {SCAN_PATH}"); print(); print("Internal checks:")
    for k, v in checks.items(): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(); print(_compact_table(compact_rows)); print(); print(f"Overall Step 3: {go_info['decision']}")


if __name__ == "__main__":
    main()
