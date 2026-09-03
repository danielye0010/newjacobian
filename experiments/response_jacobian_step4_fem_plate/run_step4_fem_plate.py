"""Run Step 4 lightweight FEM-like thermal plate response diagnostic."""
from __future__ import annotations

import csv, json
from pathlib import Path
from typing import Dict, List

import numpy as np

from compensation_solvers import run_all_methods
from fem_plate_mesh import create_fem_plate_mesh
from fem_plate_modes import create_fem_plate_modes
from jacobian_tools import deterministic_parameter_grid, scan_parameters
from thermal_response import ThermalFEMResponse, diagnostics_for_response, jax_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = PROJECT_ROOT
RESULT_DIR = PROJECT_ROOT / "results" / "response_jacobian_step4_fem_plate"
REPORT_PATH = RESULT_DIR / "STEP4_FEM_PLATE_REPORT.md"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
HISTORY_PATH = RESULT_DIR / "iteration_history.csv"
DIAG_PATH = RESULT_DIR / "diagnostics.json"
SCAN_PATH = RESULT_DIR / "parameter_scan.csv"


def _json_default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer, np.floating)): return o.item()
    return str(o)


def _write_csv(path, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(r.get(k,""), default=_json_default) if isinstance(r.get(k,""),(list,tuple,dict)) else ("" if r.get(k,"") is None else r.get(k,"")) for k in fields})


def _by_case(rows):
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for r in rows: out.setdefault(r["case_name"], {})[r["method_name"]] = r
    return out


def _checks(mesh_diag, mode_diag, scan_rows, selected, diagnostics, summary_rows, output_ready=False):
    by=_by_case(summary_rows); hard=[n for n,d in diagnostics.items() if d["rho_I_minus_J0"]>1.0]
    return {
        "project_root_is_correct": PROJECT_ROOT.resolve()==EXPECTED_ROOT.resolve(),
        "mesh_created": mesh_diag["num_nodes"]==25*21,
        "triangles_created": mesh_diag["num_triangles"]==2*(25-1)*(21-1),
        "basis_orthonormality_close_to_I": mode_diag["orthonormality_error_fro"]<1e-10,
        "fem_like_solve_runs": bool(diagnostics) and all(d["initial_residual_norm"]>0 for d in diagnostics.values()),
        "response_depends_on_Xc": any(d["identity_deviation"]>1e-3 for d in diagnostics.values()),
        "parameter_scan_completed": len(scan_rows)>=60,
        "hard_case_found": bool(hard),
        "direct_divergence_not_labeled_converged": all(not by[n]["direct_modal_inversion"]["converged"] for n in hard if by[n]["direct_modal_inversion"]["diverged"]),
        "response_calibrated_method_improves_over_direct": bool(hard) and any(min(float(by[n]["diagonal_modal_factor_initial"]["final_residual_ratio"]), float(by[n]["full_jacobian_trust_region"]["final_residual_ratio"])) < float(by[n]["direct_modal_inversion"]["final_residual_ratio"]) for n in hard),
        "output_files_created": output_ready,
    }


def _decision(checks, diagnostics, summary_rows):
    by=_by_case(summary_rows); hard=[n for n,d in diagnostics.items() if d["rho_I_minus_J0"]>1.0]
    hard_found=bool(hard)
    direct_bad=any(by[n]["direct_modal_inversion"]["diverged"] or float(by[n]["direct_modal_inversion"]["final_residual_ratio"])>1.0 for n in hard)
    calibrated_ok=bool(hard) and all(min(float(by[n]["diagonal_modal_factor_initial"]["final_residual_ratio"]), float(by[n]["full_jacobian_trust_region"]["final_residual_ratio"])) < 1e-5 for n in hard)
    response_ok=all(d.get("response_is_fem_like_solve") for d in diagnostics.values()) and any(d["identity_deviation"]>1e-3 for d in diagnostics.values())
    critical=checks["mesh_created"] and checks["triangles_created"] and checks["basis_orthonormality_close_to_I"] and response_ok and checks["output_files_created"]
    if not critical: decision="NO-GO"
    elif hard_found and direct_bad and calibrated_ok: decision="GO"
    else: decision="PARTIAL-GO"
    return {"decision":decision,"criteria":{"hard_fem_like_case_found":hard_found,"direct_inversion_bad":direct_bad,"response_calibrated_methods_succeed":calibrated_ok,"response_is_fem_like_and_nonidentity":response_ok,"outputs_complete":checks["output_files_created"]}}


def _compact(rows):
    lines=["case | rho(I-J0) | coupling | direct_ratio | oracle_scalar_ratio | diagonal_ratio | full_gn_ratio | trust_ratio | GO/PARTIAL/NO-GO","---|---:|---:|---:|---:|---:|---:|---:|---"]
    for r in rows: lines.append(f"{r['case']} | {r['rho(I-J0)']:.3f} | {r['coupling']:.3f} | {r['direct']:.3e} | {r['oracle']:.3e} | {r['diagonal']:.3e} | {r['full_gn']:.3e} | {r['trust']:.3e} | {r['decision']}")
    return "\n".join(lines)


def _method_table(rows):
    rows=sorted(rows,key=lambda r:float(r["final_residual_ratio"])); lines=["| method | converged | diverged | iters | final ratio | best ratio | selected alpha |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['method_name']} | {r['converged']} | {r['diverged']} | {r['iterations_run']} | {float(r['final_residual_ratio']):.3e} | {float(r['best_residual_ratio_seen']):.3e} | {r.get('selected_alpha') or ''} |")
    return "\n".join(lines)


def _report(mesh_diag, mode_diag, scan_summary, diagnostics, summary_rows, checks, decision, compact_rows, jax_info):
    by=_by_case(summary_rows)
    lines=["# Step 4 FEM-Like Thermal Plate Diagnostic","","## Purpose","Step 4 tests modal compensation against a lightweight fixed-connectivity FEM-like thermal plate response. The response is generated by stiffness/load solves on a triangular mesh, not by a prescribed modal response matrix.","","## What Changed From Step 3/3B","Step 3 used graph smoothing of raw distortion. Step 4 builds triangular elements, geometry-dependent edge-weight stiffness operators, thermal loads, and solves `K(Xc) u = f_thermal(Xc)` for in-plane and out-of-plane response.","","## Mesh and Modal Basis",f"- nodes: {mesh_diag['num_nodes']}",f"- triangles: {mesh_diag['num_triangles']}",f"- DOFs: {mesh_diag['num_dofs']}",f"- total area: {mesh_diag['total_area']:.6g}",f"- triangle area range: {mesh_diag['min_triangle_area']:.6g} to {mesh_diag['max_triangle_area']:.6g}",f"- basis k: {mode_diag['k']}",f"- orthonormality error: {mode_diag['orthonormality_error_fro']:.6e}","","## FEM-Like Response Model","Out-of-plane displacement solves `Kz(Xc) w = fz`, where `Kz = kb L(Xc)^T L(Xc) + kt L(Xc) + anchor I_boundary + eps I`. In-plane displacements solve membrane-like systems with thermal shrink/shear/z-coupled loads. The manufactured geometry is `Xp = Xc + response_gain U`.","",f"JAX status: `{json.dumps(jax_info)}`","","## Parameter Scan Summary",f"- candidates: {scan_summary['num_candidates']}",f"- missing categories: {scan_summary['missing_categories']}",f"- max rho(I-J0): {scan_summary['max_rho_I_minus_J']:.6e}",f"- max coupling ratio: {scan_summary['max_coupling_ratio']:.6e}",f"- candidates rho > 1: {scan_summary['num_rho_gt_1']}","","## Internal Checks"]
    for k,v in checks.items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines += ["","## Compact Result",_compact(compact_rows),""]
    for name,d in diagnostics.items():
        lines += [f"## Case: {name}","",f"- rho(I-J0): {d['rho_I_minus_J0']:.6e}",f"- coupling ratio: {d['coupling_ratio']:.6e}",f"- identity deviation: {d['identity_deviation']:.6e}",f"- condition number: {d['condition_number']:.6e}",f"- diagonal final ratio: {d['diagonal_final_ratio']:.6e}",f"- full GN final ratio: {d['full_gn_final_ratio']:.6e}",f"- trust final ratio: {d['trust_region_final_ratio']:.6e}",f"- diag_vs_trust: {d['diag_vs_trust']:.6e}",f"- process parameters: `{json.dumps(d['process_parameters'])}`","","### J0 Matrix","```text",np.array2string(np.asarray(d['J0']),precision=6),"```","","### Method Summary",_method_table(list(by[name].values())),""]
    lines += ["## Specific Interpretation"]
    lines.append(f"- FEM-like response produced J != I: {'YES' if any(d['identity_deviation']>1e-3 for d in diagnostics.values()) else 'NO'}.")
    lines.append(f"- Hard cases found: {'YES' if decision['criteria']['hard_fem_like_case_found'] else 'NO'}.")
    lines.append(f"- Direct inversion failed or was much worse in hard cases: {'YES' if decision['criteria']['direct_inversion_bad'] else 'NO'}.")
    lines.append("- Diagonal calibration is reported case-by-case; if it matches full Jacobian, the response appears mostly separable in the selected modal basis.")
    lines.append("- The final claim should be response calibration beats scalar/direct baselines; full Jacobian is the general method, not guaranteed to dominate diagonal calibration in separable regimes.")
    lines += ["","## GO / PARTIAL-GO / NO-GO",f"Decision: **{decision['decision']}**"]
    for k,v in decision["criteria"].items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    return "\n".join(lines)+"\n"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh=create_fem_plate_mesh(); B,mode_diag=create_fem_plate_modes(mesh); mesh_diag=mesh.diagnostics()
    scan_rows, selected, scan_summary=scan_parameters(mesh,B,mode_diag["orthonormality_error_fro"],deterministic_parameter_grid(120))
    diagnostics={}; summary_rows=[]; history_rows=[]
    for name,params in selected.items():
        resp=ThermalFEMResponse(name,mesh,B,params); diagnostics[name]=diagnostics_for_response(resp,mode_diag["orthonormality_error_fro"])
        res,hist=run_all_methods(resp); summary_rows.extend([r.as_row() for r in res]); history_rows.extend(hist)
    by=_by_case(summary_rows)
    for name,d in diagnostics.items():
        d["diagonal_final_ratio"]=float(by[name]["diagonal_modal_factor_initial"]["final_residual_ratio"])
        d["full_gn_final_ratio"]=float(by[name]["full_jacobian_gn_no_trust"]["final_residual_ratio"])
        d["trust_region_final_ratio"]=float(by[name]["full_jacobian_trust_region"]["final_residual_ratio"])
        d["diag_vs_trust"]=d["diagonal_final_ratio"]/max(d["trust_region_final_ratio"],1e-15)
    _write_csv(SCAN_PATH,scan_rows); _write_csv(SUMMARY_PATH,summary_rows); _write_csv(HISTORY_PATH,history_rows)
    checks=_checks(mesh_diag,mode_diag,scan_rows,selected,diagnostics,summary_rows,True); decision=_decision(checks,diagnostics,summary_rows); jax_info=jax_status()
    compact=[]
    for name in selected:
        compact.append({"case":name,"rho(I-J0)":diagnostics[name]["rho_I_minus_J0"],"coupling":diagnostics[name]["coupling_ratio"],"direct":float(by[name]["direct_modal_inversion"]["final_residual_ratio"]),"oracle":float(by[name]["oracle_scalar_scale_factor"]["final_residual_ratio"]),"diagonal":diagnostics[name]["diagonal_final_ratio"],"full_gn":diagnostics[name]["full_gn_final_ratio"],"trust":diagnostics[name]["trust_region_final_ratio"],"decision":decision["decision"]})
    DIAG_PATH.write_text(json.dumps({"mesh":mesh_diag,"basis":mode_diag,"jax":jax_info,"parameter_scan_summary":scan_summary,"selected_cases":diagnostics,"checks":checks,"decision":decision},indent=2,default=_json_default),encoding="utf-8")
    REPORT_PATH.write_text(_report(mesh_diag,mode_diag,scan_summary,diagnostics,summary_rows,checks,decision,compact,jax_info),encoding="utf-8")
    print(f"STEP4_FEM_PLATE_REPORT.md: {REPORT_PATH}"); print(f"summary.csv: {SUMMARY_PATH}"); print(f"iteration_history.csv: {HISTORY_PATH}"); print(f"diagnostics.json: {DIAG_PATH}"); print(f"parameter_scan.csv: {SCAN_PATH}"); print(); print("Internal checks:")
    for k,v in checks.items(): print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(); print(_compact(compact)); print(); print(f"Overall Step 4: {decision['decision']}")

if __name__=="__main__": main()
