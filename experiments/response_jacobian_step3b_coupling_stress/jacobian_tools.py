"""Targeted scan utilities for Step 3B coupling stress test."""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple

from physics_response import CouplingStressResponse, ProcessParams, diagnostics_for_response


def deterministic_coupling_grid(limit: int = 150) -> List[ProcessParams]:
    response_gain = [0.8, 1.0, 1.5, 2.0, 3.0]
    warp_amp = [1.0, 2.0, 4.0]
    z_gain = [0.2, 0.5, 1.0]
    slope_coupling_gain = [0.0, 0.5, 1.0, 2.0]
    curvature_gain = [0.0, 0.5, 1.0, 2.0]
    inplane_z_gain = [0.0, 0.5, 1.0, 2.0]
    mixed_warp_gain = [0.0, 0.5, 1.0, 2.0]
    mixed_xy_gain = [0.0, 0.5, 1.0]
    asymmetry = [0.0, 0.5, 1.0]
    smooth_strength = [0.1, 1.0, 5.0]
    combos = list(product(response_gain, warp_amp, z_gain, slope_coupling_gain, curvature_gain, inplane_z_gain, mixed_warp_gain, mixed_xy_gain, asymmetry, smooth_strength))
    stride = max(1, len(combos) // limit)
    params = []
    for combo in combos[::stride]:
        params.append(ProcessParams(*combo))
        if len(params) >= limit:
            break
    targeted = [
        ProcessParams(3.0, 4.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.1),
        ProcessParams(2.0, 4.0, 0.5, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.1),
        ProcessParams(1.5, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0),
        ProcessParams(2.0, 2.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0),
    ]
    params.extend(targeted)
    return params


def scan_coupling(mesh, B, basis_error: float, params_list: Iterable[ProcessParams]) -> Tuple[List[Dict[str, object]], Dict[str, ProcessParams], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    params_by_id: Dict[int, ProcessParams] = {}
    for i, params in enumerate(params_list):
        response = CouplingStressResponse(f"scan_{i:03d}", mesh, B, params)
        d = diagnostics_for_response(response, basis_error)
        row = {
            "candidate_id": i,
            **params.as_dict(),
            "rho_I_minus_J": d["rho_I_minus_J0"],
            "identity_deviation": d["identity_deviation"],
            "coupling_ratio": d["coupling_ratio"],
            "condition_number": d["condition_number"],
            "initial_residual_norm": d["initial_residual_norm"],
            "category_candidate": "",
        }
        rows.append(row)
        params_by_id[i] = params

    best_overall = max(rows, key=lambda r: r["coupling_ratio"]) if rows else None
    hard_candidates = [r for r in rows if r["rho_I_minus_J"] > 1.2]
    hard = max(hard_candidates, key=lambda r: r["coupling_ratio"]) if hard_candidates else None
    balanced_candidates = [r for r in rows if r["rho_I_minus_J"] > 1.0 and r["condition_number"] < 1e4]
    balanced = max(balanced_candidates, key=lambda r: (r["coupling_ratio"], -abs(r["rho_I_minus_J"] - 1.5))) if balanced_candidates else None
    low_candidates = [r for r in rows if r["rho_I_minus_J"] > 1.0 and r["coupling_ratio"] < 0.10]
    low = min(low_candidates, key=lambda r: r["coupling_ratio"]) if low_candidates else None

    selections = {
        "coupling_best_overall": (best_overall, lambda r: True),
        "coupling_hard": (hard, lambda r: r["rho_I_minus_J"] > 1.2),
        "coupling_balanced": (balanced, lambda r: r["rho_I_minus_J"] > 1.0 and r["condition_number"] < 1e4),
        "coupling_control_low": (low, lambda r: r["rho_I_minus_J"] > 1.0 and r["coupling_ratio"] < 0.10),
    }
    selected: Dict[str, ProcessParams] = {}
    missing = []
    used = set()
    for name, (row, predicate) in selections.items():
        if row is None:
            missing.append(name)
            continue
        cid = int(row["candidate_id"])
        if cid in used:
            alternatives = [r for r in rows if int(r["candidate_id"]) not in used and predicate(r)]
            if alternatives:
                if name == "coupling_control_low":
                    row = min(alternatives, key=lambda r: r["coupling_ratio"])
                else:
                    row = max(alternatives, key=lambda r: r["coupling_ratio"])
                cid = int(row["candidate_id"])
            else:
                missing.append(name)
                continue
        row["category_candidate"] = name
        selected[name] = params_by_id[cid]
        used.add(cid)

    summary = {
        "num_candidates": len(rows),
        "max_coupling_ratio": max((r["coupling_ratio"] for r in rows), default=0.0),
        "max_rho_I_minus_J": max((r["rho_I_minus_J"] for r in rows), default=0.0),
        "num_rho_gt_1": sum(1 for r in rows if r["rho_I_minus_J"] > 1.0),
        "num_coupling_gt_0_15": sum(1 for r in rows if r["coupling_ratio"] > 0.15),
        "num_coupling_gt_0_20": sum(1 for r in rows if r["coupling_ratio"] > 0.20),
        "missing_categories": missing,
        "top_10_by_coupling": sorted(rows, key=lambda r: r["coupling_ratio"], reverse=True)[:10],
    }
    return rows, selected, summary
