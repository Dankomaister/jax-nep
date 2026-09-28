# Ensemble model deviation

`nep_model_deviation(models, box, species, *, delta_abs, beta, delta_rel=0.0,
exponent=2.0, skin=1.0, capacity_multiplier=1.25, disable_cell_list=False)`
returns a neighbor provider and a callable. Use at least two independently fitted
models with identical species ordering, cutoffs, descriptor/invariant architecture,
parameter shapes and dtypes, atomic numbers, and ZBL mode. Fixed ZBL tables must
also agree; flexible fitted ZBL values may differ. Incompatible committees fail
at setup. One neighbor state serves the whole committee.

For forces `F` of shape `(M, N, 3)`, the mean and raw deviation are

\[
\bar F_i = M^{-1}\sum_m F_i^{(m)},\qquad
\sigma_i = \sqrt{\frac{\sum_m\|F_i^{(m)}-\bar F_i\|^2}{M-1}}.
\]

This is the [GPUMD sample force deviation](https://gpumd.org/gpumd/input_parameters/active.html),
in eV/Angstrom. The **sample normalization is M-1**, not M.
The returned atomic score is dimensionless:

\[
S_i = [\delta_{\rm abs}^{p} + (\delta_{\rm rel}\|\bar F_i\|)^p]^{1/p},
\qquad \nu_i=\sigma_i/S_i.
\]

- `delta_abs > 0`: absolute force-disagreement tolerance, in eV/Angstrom.
- `delta_rel >= 0`: dimensionless scale relative to the mean force magnitude.
- `exponent >= 1`: the generalized norm exponent `p`, default 2.
- `beta > 0`: explicit dimensionless smooth-max sharpness.

All four settings must be finite scalars. Absolute-only scoring is exactly
`delta_rel=0`, giving `sigma / delta_abs`, for every exponent. There is no separate
relative mode or species-dependent scale.

The `ModelDeviation` NamedTuple has only three fields:

| Field | Shape | Meaning |
|---|---|---|
| `atomic` | `(N,)` | Dimensionless atomic scores |
| `smooth_max` | scalar | `log(sum(exp(beta * atomic))) / beta` |
| `max` | scalar | Exact largest atomic score |

`result.max > 1` means at least one atom exceeds the configured tolerance.
The stable smooth maximum is **unnormalized** and lies between the hard maximum
and `max + log(N)/beta`; equal scores have exactly this size offset. Therefore,
`smooth_max > 1` does not by itself imply any atomic score exceeds 1. Increasing
beta approaches the hard maximum. Model disagreement is an uncertainty heuristic,
not a guaranteed prediction error.

## Example

This executable two-atom example perturbs one model only to demonstrate the API.
For scientific use, replace `models` with independently fitted committee members.

```python
from dataclasses import replace
import jax
import jax.numpy as jnp
from jax_nep import load_model, nep_model_deviation

model = load_model("nep.txt")
models = [model, replace(model, output_weights=model.output_weights * 1.01)]
R = jnp.array([[0., 0., 0.], [1.4, .2, .1]], dtype=jnp.float32)
species = model.map_species(["C", "C"])
provider, deviation_fn = nep_model_deviation(
    models, 12., species, delta_abs=0.1, delta_rel=0.1, exponent=2., beta=10.,
)
neighbor = provider.allocate(R)
neighbor = jax.jit(provider.update)(R, neighbor)
if bool(neighbor.did_buffer_overflow):
    raise RuntimeError("Reallocate neighbors before accepting results")
result = jax.jit(deviation_fn)(R, neighbor=neighbor)
committee_forces = jax.jit(deviation_fn.forces)(R, neighbor=neighbor)
score_gradient = jax.jit(jax.grad(
    lambda r: deviation_fn(r, neighbor=neighbor).smooth_max
))(R)
```

`deviation_fn.forces` returns `(M, N, 3)` forces in eV/Angstrom. Keep the same
[neighbor lifecycle](api.md#neighbor-lifecycle) as single-model inference.
The position gradient differentiates through forces and thus through the NEP
energy's second derivatives. Geometry, basis, and harmonics are shared inside
this differentiable computation; fitted projections, scaling, networks and
flexible ZBL remain model-dependent. The compiled path vectorizes stacked
parameter leaves and contains no Python model loop.

No epsilon is added to raw deviation. Perfect agreement is a norm's
nondifferentiable point; derivatives there are not guaranteed finite. At exactly
zero mean force, the scaling norm uses a zero subgradient (needed also for the
absolute limit); for `p=1` with nonzero relative scale this is a subgradient
convention. The exact maximum is not smooth at crossings of the largest atom.

## Low-level utilities

`jax_nep.active.force_deviation(forces)` exposes raw sigma in eV/Angstrom.
`jax_nep.active.stack_models(models)` validates and stacks fitted parameter leaves
for `jax.vmap`; the resulting model must be mapped over its leading axis.
These helpers do not enlarge the high-level result.

`jax_nep.smooth_max(x, *, beta, axis=None, keepdims=False)` is generic JAX math.
Its caller must supply a finite positive scalar beta; it can be a traced argument.
The active-learning factory validates beta at setup. The utility supports ordinary
JAX reduction axes and differentiates with respect to its array argument.
