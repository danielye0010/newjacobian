# Complex Geometry Generalization

This package generates complex surrogate geometries from scratch and tests the
response-Jacobian compensation hierarchy with a differentiable FEM forward
model.

Run from the project root:

```powershell
python experiments\step6_jax_differentiable_fem_response_jacobian\complex_geometry_generalization\run_complex_geometry_generalization.py
```

Outputs are written to:

```text
results\step6_jax_differentiable_fem_response_jacobian\complex_geometry_generalization
```

No external CAD/STL files are required. No figures are generated. The result is
a differentiable FEM surrogate generalization check, not real AM/DED validation.
