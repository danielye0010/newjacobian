# Step 6 JAX Differentiable FEM Response-Jacobian Benchmark

This step adds a true JAX-differentiable FEM surrogate for the response-Jacobian modal compensation framework.

Run from the project root:

```powershell
python experiments\step6_jax_differentiable_fem_response_jacobian\run_step6_jax_differentiable_fem_response_jacobian.py
```

The runner writes numerical outputs to:

```text
results\step6_jax_differentiable_fem_response_jacobian
```

No figures are generated. The benchmark uses a small dense plate FEM surrogate so `jax.jacfwd` can differentiate through the forward map

```text
X(c) -> FEM solve -> u(c) -> b(c)
```

and compute the modal response Jacobian directly as `db/dc`.
