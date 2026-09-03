# Step 3B Coupling Stress Test

This experiment refines Step 3 by adding stronger physical-space coupling mechanisms to the mesh-based plate response while keeping fixed connectivity and modal compensation.

It does not implement full FEM, remeshing, or direct modal response matrices.

## Run

```powershell
python experiments\response_jacobian_step3b_coupling_stress\run_step3b_coupling_stress.py
```

Outputs are written to `results\response_jacobian_step3b_coupling_stress\`.
