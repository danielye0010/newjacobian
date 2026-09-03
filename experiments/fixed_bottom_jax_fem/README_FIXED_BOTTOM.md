# Fixed-Bottom JAX-FEM Experiment

This independent suite studies response-Jacobian modal compensation when the
compensated input geometry must keep every bottom/first-layer node fixed.

Run from the project root:

```powershell
python -B experiments\fixed_bottom_jax_fem\run_fixed_bottom_jax_fem.py
```

Outputs are written only to:

```text
results\fixed_bottom_jax_fem
```

The response is a differentiable voxel-spring eigenstrain surrogate for stable
printing followed by release-like warpage. It is not calibrated FDM validation
and does not model adhesion failure, delamination, contact loss, or detachment.
