# Step 3 Physics Plate Response Diagnostic

This folder contains the fixed-connectivity thin-plate physics-response diagnostic.

The response is generated from current mesh geometry, raw distortion terms, and a graph-Laplacian smoothing solve. It does not use a directly prescribed modal response matrix as the main response.

## Run

```powershell
python experiments\response_jacobian_step3_physics_plate\run_step3_physics_plate.py
```

Outputs are written to:

```text
results\response_jacobian_step3_physics_plate\
```

Main artifacts: `STEP3_PHYSICS_PLATE_REPORT.md`, `summary.csv`, `iteration_history.csv`, `diagnostics.json`, and `parameter_scan.csv`.
