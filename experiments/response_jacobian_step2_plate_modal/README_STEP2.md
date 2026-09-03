# Step 2 Plate Modal Geometry Diagnostic

This folder contains the fixed-connectivity thin-plate modal geometry diagnostic. It verifies the move from synthetic modal-coordinate residuals to an actual plate geometry parameterization:

```text
X(c) = X0 + B c
b(c) = B^T M r(c)
```

No FEM, JAX, or modal mesh/FEM modes are implemented here.

## Files

- `plate_mesh.py`: structured plate grid and lumped area weights.
- `plate_modes.py`: eight analytical low-frequency plate modes and mass orthonormalization.
- `plate_responses.py`: frozen and geometry-dependent response families.
- `plate_solvers.py`: direct inversion, scalar factors, oracle scalar, diagonal factor, full Jacobian GN, and trust-region GN.
- `run_step2_plate_modal.py`: runs all cases and writes report/CSV/JSON outputs.

## Run

From the project root:

```powershell
python experiments\response_jacobian_step2_plate_modal\run_step2_plate_modal.py
```

Outputs are written to:

```text
results\response_jacobian_step2_plate_modal\
```

The main review artifact is `STEP2_PLATE_MODAL_REPORT.md`, supported by `summary.csv`, `iteration_history.csv`, and `diagnostics.json`.
