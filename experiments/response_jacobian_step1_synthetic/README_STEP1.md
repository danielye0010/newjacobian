# Step 1 Synthetic Response-Jacobian Diagnostic

This folder contains the controlled low-dimensional synthetic experiment for validating the response-Jacobian compensation idea before any FEM, JAX, or modal mesh implementation.

## Files

- `response_models.py`: residual model, nonlinear coupling, analytical Jacobian, finite-difference checker, and case definitions.
- `solvers.py`: direct inversion, fixed scalar factors, oracle scalar factor, diagonal modal factor, full Jacobian Gauss-Newton, and trust-region Gauss-Newton.
- `run_step1_synthetic.py`: runs all cases, writes CSV/JSON/Markdown outputs, and prints a compact result table.

## Run

From the project root:

```powershell
python experiments\response_jacobian_step1_synthetic\run_step1_synthetic.py
```

Outputs are written to:

```text
results\response_jacobian_step1_synthetic\
```

The main review artifact is `STEP1_SYNTHETIC_REPORT.md`, supported by `summary.csv`, `iteration_history.csv`, and `diagnostics.json`.
