"""Parameter scan and Jacobian helpers for Step 4."""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple

from thermal_response import FEMProcessParams, ThermalFEMResponse, diagnostics_for_response


def deterministic_parameter_grid(limit: int = 120) -> List[FEMProcessParams]:
    response_gain = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    thermal_amp = [0.5, 1.0, 2.0, 4.0]
    asymmetry = [0.0, 0.5, 1.0]
    z_feedback = [0.0, 0.2, 0.5, 1.0]
    slope_feedback = [0.0, 0.1, 0.3]
    shrink = [(-0.002, -0.002), (-0.005, 0.002), (-0.010, 0.005)]
    kb = [0.1, 1.0, 5.0]
    kt = [0.01, 0.1, 1.0]
    anchor = [0.01, 0.1, 1.0]
    combos = list(product(response_gain, thermal_amp, asymmetry, z_feedback, slope_feedback, shrink, kb, kt, anchor))
    stride = max(1, len(combos) // limit)
    params: List[FEMProcessParams] = []
    for combo in combos[::stride]:
        rg, ta, asym, zf, sf, sh, kbv, ktv, anch = combo
        sx, sy = sh
        params.append(FEMProcessParams(rg, ta, asym, zf, sf, sx, sy, kbv, ktv, anch))
        if len(params) >= limit:
            break
    params.extend([
        FEMProcessParams(0.2, 0.5, 0.0, 0.0, 0.0, -0.002, -0.002, 5.0, 1.0, 1.0),
        FEMProcessParams(1.0, 1.0, 0.5, 0.2, 0.0, -0.005, 0.002, 1.0, 0.1, 0.1),
        FEMProcessParams(2.0, 2.0, 1.0, 0.5, 0.1, -0.010, 0.005, 0.1, 0.01, 0.01),
        FEMProcessParams(3.0, 4.0, 1.0, 1.0, 0.3, -0.010, 0.005, 0.1, 0.01, 0.01),
    ])
    return params


def scan_parameters(mesh, B, basis_error: float, params_list: Iterable[FEMProcessParams]) -> Tuple[List[Dict[str, object]], Dict[str, FEMProcessParams], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_id: Dict[int, FEMProcessParams] = {}
    for i, params in enumerate(params_list):
        response = ThermalFEMResponse(f"scan_{i:03d}", mesh, B, params)
        d = diagnostics_for_response(response, basis_error)
        row = {"candidate_id": i, **params.as_dict(), "rho_I_minus_J": d["rho_I_minus_J0"], "identity_deviation": d["identity_deviation"], "coupling_ratio": d["coupling_ratio"], "condition_number": d["condition_number"], "initial_residual_norm": d["initial_residual_norm"], "category_candidate": ""}
        rows.append(row); by_id[i] = params

    easy_pool = [r for r in rows if r["rho_I_minus_J"] < 0.7]
    easy = min(easy_pool, key=lambda r: abs(r["rho_I_minus_J"] - 0.4)) if easy_pool else None
    marginal = min(rows, key=lambda r: abs(r["rho_I_minus_J"] - 1.0)) if rows else None
    hard_pool = [r for r in rows if r["rho_I_minus_J"] > 1.2]
    hard_gain = min(hard_pool, key=lambda r: (r["coupling_ratio"], -r["rho_I_minus_J"])) if hard_pool else None
    hard_coupled = max(hard_pool, key=lambda r: r["coupling_ratio"]) if hard_pool else None
    nonlinear_pool = [r for r in rows if r["rho_I_minus_J"] > 1.0 and (r["z_feedback"] >= 0.5 or r["slope_feedback"] >= 0.1)]
    hard_nonlinear = max(nonlinear_pool, key=lambda r: (r["z_feedback"] + r["slope_feedback"], r["rho_I_minus_J"])) if nonlinear_pool else None

    selected_rows = {"fem_easy": easy, "fem_marginal": marginal, "fem_hard_gain": hard_gain, "fem_hard_coupled": hard_coupled, "fem_hard_nonlinear": hard_nonlinear}
    selected = {}; missing = []; used = set()
    for name, row in selected_rows.items():
        if row is None:
            missing.append(name); continue
        cid = int(row["candidate_id"])
        if cid in used:
            continue
        row["category_candidate"] = name
        selected[name] = by_id[cid]
        used.add(cid)
    summary = {"num_candidates": len(rows), "missing_categories": missing, "max_rho_I_minus_J": max((r["rho_I_minus_J"] for r in rows), default=0.0), "max_coupling_ratio": max((r["coupling_ratio"] for r in rows), default=0.0), "num_rho_gt_1": sum(1 for r in rows if r["rho_I_minus_J"] > 1.0), "num_rho_gt_1_2": sum(1 for r in rows if r["rho_I_minus_J"] > 1.2), "top_10_hard": sorted(rows, key=lambda r: r["rho_I_minus_J"], reverse=True)[:10]}
    return rows, selected, summary
