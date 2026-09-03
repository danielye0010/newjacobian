# Step 6 JAX FEM Response-Jacobian Discovery Suite

This exploratory suite extends the passed Step 6 differentiable FEM validation.
It is intended to support the paper story that geometric compensation is a
response-calibration problem, not that one solver always wins.

Run from the project root:

```powershell
python experiments\step6_jax_differentiable_fem_response_jacobian\discovery_suite\run_discovery_suite.py
```

Outputs are written to:

```text
results\step6_jax_differentiable_fem_response_jacobian\discovery_suite
```

The suite creates CSV tables and a markdown discovery report. It does not
create figures and does not update the manuscript.
