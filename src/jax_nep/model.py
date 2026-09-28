"""Immutable NEP4 semantics, independent of neighbor or execution policy."""

from dataclasses import dataclass
from functools import cached_property
import jax
import numpy as np


def _array_key(value):
    if value is None:
        return None
    value = np.asarray(value)
    return value.dtype.str, value.shape, value.tobytes()


def _array_from_key(key):
    if key is None:
        return None
    dtype, shape, data = key
    return np.frombuffer(data, dtype=dtype).reshape(shape)


@jax.tree_util.register_pytree_with_keys_class
@dataclass(frozen=True, eq=False)
class NEPModel:
    """Immutable structure with named, differentiable fitted parameter leaves.

    Loaded leaves are read-only NumPy arrays; JAX transformations naturally
    replace them with device arrays or tracers. Fixed ZBL tables are structural;
    only flexible ZBL tables are fitted leaves.
    """
    species: tuple[str, ...]
    cutoffs: np.ndarray  # species, radial/angular
    orders: tuple[int, int]
    basis_orders: tuple[int, int]
    l_max: int
    invariants: tuple[str, ...]
    input_weights: np.ndarray  # species, descriptor, hidden
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    coefficients: tuple[np.ndarray, np.ndarray]  # central, neighbor, basis, channel
    scale: np.ndarray
    zbl: np.ndarray | None  # symmetric species-pair table, 10 values
    atomic_numbers: np.ndarray
    zbl_mode: str | None = None

    def __post_init__(self):
        # Own immutable static arrays, including models built with dataclasses.replace.
        for name in ("cutoffs", "atomic_numbers"):
            object.__setattr__(self, name, _array_from_key(_array_key(getattr(self, name))))
        if self.zbl_mode != "flexible":
            object.__setattr__(self, "zbl", _array_from_key(_array_key(self.zbl)))

    @property
    def _parameter_names(self):
        names = ("input_weights", "hidden_bias", "output_weights", "output_bias",
                 "coefficients", "scale")
        return names + (("zbl",) if self.zbl_mode == "flexible" else ())

    @cached_property
    def _structure(self):
        # Array bytes give auxiliary data value equality and a stable hash,
        # without putting integer arrays or structural floats in the leaves.
        return (self.species, _array_key(self.cutoffs), self.orders,
                self.basis_orders, self.l_max, self.invariants,
                _array_key(self.atomic_numbers), self.zbl_mode,
                None if self.zbl_mode == "flexible" else _array_key(self.zbl))

    def tree_flatten(self):
        return tuple(getattr(self, n) for n in self._parameter_names), self._structure

    def tree_flatten_with_keys(self):
        return tuple((jax.tree_util.GetAttrKey(n), getattr(self, n))
                     for n in self._parameter_names), self._structure

    @classmethod
    def tree_unflatten(cls, structure, children):
        species, cutoffs, orders, basis, ell, invariants, numbers, mode, fixed_zbl = structure
        model = cls(species, _array_from_key(cutoffs), orders, basis, ell, invariants,
                    *children[:6],
                    children[6] if mode == "flexible" else _array_from_key(fixed_zbl),
                    _array_from_key(numbers), mode)
        object.__setattr__(model, "_structure", structure)
        return model

    @property
    def dimension(self):
        return self.scale.shape[-1]

    @property
    def cutoff(self):
        return float(self.cutoffs.max())

    def map_species(self, symbols):
        mapping = {name: i for i, name in enumerate(self.species)}
        try:
            return np.asarray([mapping[s] for s in symbols], dtype=np.int32)
        except KeyError as error:
            raise ValueError(
                f"Unknown species {error.args[0]!r}; expected {self.species}"
            ) from error

    def validate_types(self, species):
        types = np.asarray(species)
        if (
            types.ndim != 1
            or types.dtype.kind not in "iu"
            or not types.size
            or np.any(types < 0)
            or np.any(types >= len(self.species))
        ):
            raise ValueError("species must be nonempty integer model indices")
        return types.astype(np.int32)
