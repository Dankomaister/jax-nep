# Virial and stress

Both quantities differentiate the production energy expression with JAX autodiff.
For Cartesian row vectors, apply the dimensionless deformation
`r' = r @ (I + eps)` to all pair vectors, including periodic image vectors.
The total configurational virial is

\[
W_{ab} = -\left.\frac{\partial E}{\partial\epsilon_{ab}}\right|_{\epsilon=0}.
\]

The returned shape is `(3, 3)` with Cartesian indices `(a, b)` in x, y, z order.
There is no Voigt packing or per-atom normalization. For an isolated system this
convention corresponds to `sum_i r[i, a] * force[i, b]`.

Tensile-positive configurational stress is `sigma = -W / V`. Supply the positive
physical volume in Angstrom³: the product of orthorhombic lengths or the absolute
determinant of the row-lattice matrix. Do not multiply volume by the number of
periodic images. Virial is in eV; stress is in eV/Angstrom³. A value in
eV/Angstrom³ can be converted to GPa by multiplying by approximately 160.21766208.
Neither helper includes a kinetic contribution. The configurational pressure
contribution is `trace(W) / (3 * V)`.

After the [quickstart](quickstart.md) setup:

```python
from jax_nep import virial, stress
W = virial(energy_fn, R, neighbor=neighbor)
sigma = stress(energy_fn, R, neighbor=neighbor, volume=box**3)
```

This example uses the quickstart's cubic box. For a matrix box use its determinant.
Both helpers accept JIT compilation and use the matching allocated neighbor state.
They differentiate infinitesimal strain with fixed connectivity; for a finite
change of cell, construct a new factory and allocate neighbors in that cell.

The affine deformation, its cotangent accumulation, and the ANN projection
reduction use scoped float64 arithmetic for robust strain derivatives. Global
JAX x64 mode is not enabled by the library. See [numerical behavior](numerical_behavior.md).
