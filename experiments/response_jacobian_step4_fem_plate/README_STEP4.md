# Step 4 FEM-Like Thermal Plate Diagnostic

This experiment uses a fixed triangular plate mesh and lightweight FEM-like thermal response solves. It does not implement industrial FEM, remeshing, or direct modal response matrices.

Run:

```powershell
python experiments\response_jacobian_step4_fem_plate\run_step4_fem_plate.py
```

Outputs are written to `results\response_jacobian_step4_fem_plate\`.
