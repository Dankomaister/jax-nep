"""Native neighbor ownership and the boundary between connectivity and physics."""

from typing import NamedTuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import partition, custom_partition, dataclasses
from .geometry import Lattice


class PairData(NamedTuple):
    vectors: object
    central_types: object
    neighbor_types: object
    valid: object
    centers: object = None
    n_atoms: int = 0


@dataclasses.dataclass
class NeighborState:
    native: object
    owner: object = dataclasses.static_field()
    angular: object = None

    @property
    def idx(self):
        return self.native.idx

    @property
    def did_buffer_overflow(self):
        return self.native.did_buffer_overflow | (
            False if self.angular is None else self.angular.did_buffer_overflow
        )

    @property
    def error(self):
        if self.owner.backend == "STANDARD":
            code = self.native.error.code
            if self.angular is not None:
                code = code | self.angular.error.code
            return partition.PartitionError(code)
        return partition.PartitionError(
            jnp.asarray(self.did_buffer_overflow, jnp.uint8)
            * int(partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW)
        )

    def update(self, positions):
        return self.owner.update(positions, self)


class NeighborProvider:
    def __init__(
        self,
        box,
        species,
        cutoff,
        *,
        skin=1.0,
        capacity_multiplier=1.25,
        disable_cell_list=False,
        dtype=np.float32,
        representation=partition.Dense,
    ):
        if (
            not np.isfinite(skin)
            or skin < 0
            or not np.isfinite(capacity_multiplier)
            or capacity_multiplier < 1
            or not isinstance(disable_cell_list, (bool, np.bool_))
        ):
            raise ValueError("invalid neighbor settings")
        self.representation = representation
        self.skin = skin
        self.angular_provider = None
        self.types = jnp.asarray(species, dtype=jnp.int32)
        self.lattice = Lattice(box, cutoff + skin, dtype)
        self.backend = (
            "MULTI_IMAGE" if cutoff > self.lattice.minimum_image_radius else "STANDARD"
        )
        if self.backend == "MULTI_IMAGE":
            with jax.default_matmul_precision("highest"):
                self.native = custom_partition.neighbor_list_multi_image(
                    None,
                    self.lattice.cell.T,
                    cutoff + skin,
                    dr_threshold=0,
                    capacity_multiplier=capacity_multiplier,
                    fractional_coordinates=False,
                    format=representation,
                )
            return
        scale = self.lattice.search_scale

        def displacement(a, b, **unused):
            return self.lattice.minimum_image(self.lattice.cartesian((a - b) / scale))

        self.native = partition.neighbor_list(
            displacement,
            scale,
            cutoff + skin,
            dr_threshold=0,
            capacity_multiplier=capacity_multiplier,
            disable_cell_list=disable_cell_list,
            format=representation,
        )

    def validate(self, neighbor):
        if not isinstance(neighbor, NeighborState) or neighbor.owner is not self:
            raise ValueError(
                "neighbor must come from the matching factory; raw/foreign list rejected"
            )

    def search_positions(self, positions):
        if positions.shape != (len(self.types), 3):
            raise ValueError("positions must have shape (number of atoms, 3)")
        if self.backend == "MULTI_IMAGE":
            return self.lattice.wrap(positions)
        return (
            jnp.mod(self.lattice.fractional(positions), 1) * self.lattice.search_scale
        )

    @partial(jax.jit, static_argnums=0)
    def reorder(self, native):
        if self.representation == partition.Dense:
            return native
        if not native.idx[0].size:
            return native
        centers = native.idx[1] if self.backend == "STANDARD" else native.idx[0]
        order = jnp.argsort(centers, stable=True)
        sorted_centers = centers[order]
        edges = jnp.arange(len(centers))
        first = jnp.concatenate(
            (jnp.ones(1, bool), sorted_centers[1:] != sorted_centers[:-1])
        )
        rank = edges - jax.lax.associative_scan(jnp.maximum, jnp.where(first, edges, 0))
        order = order[
            jnp.lexsort((sorted_centers, rank, sorted_centers >= len(self.types)))
        ]
        idx = jnp.stack(native.idx)[:, order]
        if self.backend == "STANDARD":
            return native.set(idx=idx)
        return native.set(idx=tuple(idx), shifts=native.shifts[order])

    def allocate(self, positions, extra_capacity=0):
        with jax.default_matmul_precision("highest"):
            native = self.native.allocate(
                self.search_positions(positions), extra_capacity=extra_capacity
            )
        native = self.reorder(native)
        angular = (
            None
            if self.angular_provider is None
            else self.angular_provider.allocate(positions, extra_capacity)
        )
        return NeighborState(native, self, angular)

    def _rebuild(self, positions, neighbor):
        with jax.default_matmul_precision("highest"):
            native = self.reorder(
                self.native.update(self.search_positions(positions), neighbor.native)
            )
        angular = (
            None
            if neighbor.angular is None
            else self.angular_provider._rebuild(positions, neighbor.angular)
        )
        return NeighborState(native, self, angular)

    def update(self, positions, neighbor):
        self.validate(neighbor)
        current = self.search_positions(positions)
        delta = current - neighbor.native.reference_position
        if self.backend == "STANDARD":
            delta = self.lattice.minimum_image(
                self.lattice.cartesian(delta / self.lattice.search_scale)
            )
        # MULTI_IMAGE shifts refer to wrapped coordinates: a boundary crossing
        # must trigger a rebuild even when the minimum-image motion is small.
        rebuild = jnp.any(jnp.sum(delta * delta, axis=-1) > (self.skin / 2) ** 2)
        return jax.lax.cond(
            rebuild | (self.skin == 0),
            lambda _: self._rebuild(positions, neighbor),
            lambda _: neighbor,
            None,
        )

    def pairs(self, positions, neighbor):
        if self.representation == partition.Dense:
            return self.dense_pairs(positions, neighbor)
        return self.sparse_pairs(positions, neighbor)

    def dense_pairs(self, positions, neighbor):
        self.validate(neighbor)
        idx = neighbor.idx
        n = len(self.types)
        valid = (idx >= 0) & (idx < n)
        gathered = jnp.take(
            positions, jnp.where(valid, idx, n), axis=0, mode="fill", fill_value=0
        )
        delta = gathered - positions[:, None, :]
        if self.backend == "MULTI_IMAGE":
            winding = jnp.floor(self.lattice.fractional(positions))
            shift = (
                neighbor.native.shifts
                + winding[:, None, :]
                - winding[jnp.clip(idx, 0, n - 1)]
            )
            delta += self.lattice.cartesian(shift)
        else:
            delta = self.lattice.minimum_image(delta)
        valid &= jnp.any(delta != 0, axis=-1)
        return PairData(
            delta, self.types[:, None], self.types[jnp.clip(idx, 0, n - 1)], valid
        )

    def sparse_pairs(self, positions, neighbor):
        self.validate(neighbor)
        shifts = None if self.backend == "STANDARD" else neighbor.native.shifts
        return self._sparse_pairs(positions, neighbor.idx, shifts)

    def _sparse_pairs(self, positions, indices, shifts=None):
        n = len(self.types)
        a, b = indices
        center, sender = (b, a) if self.backend == "STANDARD" else (a, b)
        valid = (center >= 0) & (center < n) & (sender >= 0) & (sender < n)
        ci, sj = jnp.clip(center, 0, n - 1), jnp.clip(sender, 0, n - 1)
        indices = jnp.stack(
            (jnp.where(valid, sender, n), jnp.where(valid, center, n)), axis=1
        )
        positions_on_edges = jnp.take(
            positions, indices, axis=0, mode="fill", fill_value=0
        )
        delta = positions_on_edges[:, 0] - positions_on_edges[:, 1]
        if self.backend == "MULTI_IMAGE":
            winding = jnp.floor(self.lattice.fractional(positions))
            delta += self.lattice.cartesian(shifts + winding[ci] - winding[sj])
        else:
            delta = self.lattice.minimum_image(delta)
        valid &= jnp.any(delta != 0, axis=-1)
        return PairData(
            delta, self.types[ci], self.types[sj], valid, jnp.where(valid, center, n), n
        )

    def pair_domains(self, positions, neighbor):
        """Share endpoint gather/scatter across separate STANDARD Sparse domains."""
        self.validate(neighbor)
        if (
            neighbor.angular is not None
            and self.representation == partition.Sparse
            and self.angular_provider.representation == partition.Sparse
            and self.backend == self.angular_provider.backend == "STANDARD"
        ):
            indices = jnp.concatenate((neighbor.idx, neighbor.angular.idx), axis=1)
            pairs = self._sparse_pairs(positions, indices)
            count = neighbor.idx.shape[1]

            def section(start, end):
                return PairData(
                    pairs.vectors[start:end],
                    pairs.central_types[start:end],
                    pairs.neighbor_types[start:end],
                    pairs.valid[start:end],
                    pairs.centers[start:end],
                    pairs.n_atoms,
                )

            return section(0, count), section(count, None)
        return self.pairs(positions, neighbor), self.angular_pairs(positions, neighbor)

    def angular_pairs(self, positions, neighbor):
        return (
            None
            if neighbor.angular is None
            else self.angular_provider.pairs(positions, neighbor.angular)
        )
