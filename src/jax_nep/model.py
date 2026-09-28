"""Immutable NEP4 semantics, independent of neighbor or execution policy."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class NEPModel:
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

    @property
    def dimension(self):
        return self.scale.size

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
