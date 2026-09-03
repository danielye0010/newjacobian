"""Jacobian and parameter-scan helpers for Step 3."""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np

from physics_response import C0, PhysicsPlateResponse, ProcessParams, diagnostics_for_response


def finite_difference_jacobian(response: PhysicsPlateResponse, c: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return response.jacobian_fd(c, eps=eps)


def deterministic_parameter_grid(limit: int = 90) -> List[ProcessParams]:
    response_gain = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    warp_amp = [0.5, 1.0, 2.0, 4.0]
    z_gain = [0.0, 0.2, 0.5, 1.0]
    slope_gain = [0.0, 0.1, 0.3]
    asymmetry = [0.0, 0.5, 1.0]
    shrinkage = [(-0.002, -0.002), (-0.005, 0.002), (-0.010, 0.005)]
    smooth_strength = [0.1, 1.0, 5.0]
    all_params: List[ProcessParams] = []
    combos = list(product(response_gain, warp_amp, z_gain, slope_gain, asymmetry, shrinkage, smooth_strength))
    stride = max(1, len(combos) // limit)
    for combo in combos[::stride]:
        rg, wa, zg, sg, asym, shrink, sm = combo
        sx, sy = shrink
        all_params.append(ProcessParams(rg, wa, zg, sg, asym, sx, sy, sm))
        if len(all_params) >= limit:
            break
    required = [
        ProcessParams(0.2, 0.5, 0.0, 0.0, 0.0, -0.002, -0.002, 5.0),
        ProcessParams(1.0, 1.0, 0.2, 0.0, 0.5, -0.005, 0.002, 1.0),
        ProcessParams(2.0, 2.0, 0.5, 0.0, 0.5, -0.010, 0.005, 0.1),
        ProcessParams(3.0, 4.0, 1.0, 0.3, 1.0, -0.010, 0.005, 0.1),
    ]
    all_params.extend(required)
    return all_params


def scan_parameters(mesh, B, basis_error: float, params_list: Iterable[ProcessParams]) -> Tuple[List[Dict[str, object]], Dict[str, ProcessParams], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    params_by_id: Dict[int, ProcessParams] = {}
    for i, params in enumerate(params_list):
        response = PhysicsPlateResponse(f"scan_{i:03d}", mesh, B, params)
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

    def choose(predicate, key):
        candidates = [r for r in rows if predicate(r)]
        if not candidates:
            return None
        return min(candidates, key=key)

    easy = choose(lambda r: r["rho_I_minus_J"] < 0.7, lambda r: (r["coupling_ratio"], abs(r["rho_I_minus_J"] - 0.45)))
    marginal = min(rows, key=lambda r: abs(r["rho_I_minus_J"] - 1.0)) if rows else None
    hard_gain = choose(lambda r: r["rho_I_minus_J"] > 1.2, lambda r: (r["coupling_ratio"], -r["rho_I_minus_J"]))
    hard_coupled = choose(lambda r: r["rho_I_minus_J"] > 1.2 and r["coupling_ratio"] > 0.15, lambda r: (-r["coupling_ratio"], -r["rho_I_minus_J"]))
    if hard_coupled is None:
        hard_coupled = choose(lambda r: r["rho_I_minus_J"] > 1.2, lambda r: (-r["coupling_ratio"], -r["rho_I_minus_J"]))
    hard_nonlinear = choose(lambda r: r["rho_I_minus_J"] > 1.0 and (r["z_gain"] >= 0.5 or r["slope_gain"] >= 0.1), lambda r: (-r["z_gain"] - r["slope_gain"], -r["rho_I_minus_J"]))

    selections = {
        "physics_easy": easy,
        "physics_marginal": marginal,
        "physics_hard_gain": hard_gain,
        "physics_hard_coupled": hard_coupled,
        "physics_hard_nonlinear": hard_nonlinear,
    }
    selected_params: Dict[str, ProcessParams] = {}
    missing = []
    for label, row in selections.items():
        if row is None:
            missing.append(label)
            continue
        row["category_candidate"] = label
        selected_params[label] = params_by_id[int(row["candidate_id"])]
    top_closest = sorted(rows, key=lambda r: (abs(r["rho_I_minus_J"] - 1.2), -r["coupling_ratio"]))[:10]
    return rows, selected_params, {"missing_categories": missing, "top_10_closest_candidates": top_closest}
