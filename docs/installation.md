# Installation

JAX-NEP requires Python 3.12 or later, JAX 0.11.1 or later, JAX-MD 0.2.29
or later, and NumPy 2 or later. The release is validated with JAX 0.11.1 and
JAX-MD 0.2.29. Install from the root of a downloaded or cloned source tree:

```bash
python -m pip install .
```

For editable installation and tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

There is no requirement to install NEP_CPU. Reference arrays and their input
models are included with the tests. The `dev` extra also provides package-building
and mathematical-table maintenance tools.

## Accelerators

Install the JAX build appropriate for your hardware first, following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Driver and accelerator requirements depend on that build. Then install JAX-NEP
into the same environment. Check the devices visible to JAX:

```python
import jax
print(jax.devices())
```

CPU execution is supported; GPU execution is recommended for larger systems.
No cluster-specific environment modules or shell setup are required by this package.
See [numerical validation](numerical_behavior.md#validation) for the full GPU test command.
