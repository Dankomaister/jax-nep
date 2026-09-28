"""Static policy contracts; synthetic geometry and capacity, no model identities."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import partition
from jax_nep.neighbors import NeighborProvider
from jax_nep.policy import ExecutionPlan, PlannedNeighbors
import jax_nep.policy as policy


def model(radial=4.0, angular=4.0):
    # Only physical dimensions are available to the planner: no name/path/ID.
    return SimpleNamespace(cutoff=radial, cutoffs=np.array([[radial, angular]]),
                           orders=(1, 3), basis_orders=(3, 3), l_max=4)


@pytest.mark.parametrize('box,cutoff,expected', [
    (12., 4., 'STANDARD'), (8., 4., 'STANDARD'),
    (8., np.nextafter(4., np.inf), 'MULTI_IMAGE'), (6., 4., 'MULTI_IMAGE'),
    (np.array([[10., 0, 0], [9., 3, 0], [0, 0, 12.]]), 1.5, 'STANDARD'),
    (np.array([[10., 0, 0], [9., 3, 0], [0, 0, 12.]]), 2., 'MULTI_IMAGE'),
])
def test_backend_geometry_only(box, cutoff, expected):
    # In the skew cell the shortest vector is b-a, not a box diagonal.
    for count, skin in [(1, 0.), (7, 1.)]:
        provider = NeighborProvider(box, np.zeros(count, np.int32), cutoff,
                                    skin=skin, disable_cell_list=True)
        assert provider.backend == expected


def test_equal_cutoffs_have_no_child():
    planner = PlannedNeighbors(model(), 20., np.zeros(4, np.int32),
                               disable_cell_list=True)
    assert planner.angular is None
    r = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    assert planner.allocate(r).angular is None


@pytest.mark.parametrize('slots,live,angular_slots,backend,expected,separate', [
    (12, 120, 12, 'STANDARD', partition.Dense, False),
    (12, 12, 12, 'STANDARD', partition.Sparse, False),
    (12, 120, 1, 'STANDARD', partition.Sparse, True),
    (12, 12, 12, 'MULTI_IMAGE', partition.Dense, False),
    (12, 120, 1, 'MULTI_IMAGE', partition.Dense, True),
])
def test_capacity_decisions(monkeypatch, slots, live, angular_slots, backend,
                            expected, separate):
    """Isolate the selector from allocator details with explicit static capacities."""
    class CapacityProvider:
        def __init__(self, box, species, cutoff, **settings):
            self.types = species
            self.backend = backend
            self.representation = settings.get('representation', partition.Dense)
            self.angular_provider = None
            self.cutoff = cutoff
            self.lattice = SimpleNamespace(diagonal=True, corrections=[])

        def allocate(self, positions, extra_capacity=0):
            width = slots if self.cutoff == 4 else angular_slots
            idx = np.full((len(self.types), width), len(self.types), np.int32)
            idx.flat[:min(live, idx.size)] = 0
            child = (None if self.angular_provider is None else
                     self.angular_provider.allocate(positions))
            return SimpleNamespace(idx=idx, owner=self, angular=child)

    monkeypatch.setattr(policy, 'NeighborProvider', CapacityProvider)
    planner = PlannedNeighbors(model(4, 2), 20., np.zeros(16, np.int32))
    r = np.zeros((16, 3), np.float32)
    first, second = planner.allocate(r), planner.allocate(r)
    assert first.owner is second.owner
    assert first.owner.representation == expected
    assert (first.angular is not None) == separate
    if planner.plans:
        assert list(planner.plans) == [ExecutionPlan(separate, expected)]


@pytest.mark.parametrize('box,angular,expected_backend', [
    (30., 4., 'STANDARD'), (30., 1., 'STANDARD'), (7., 1., 'MULTI_IMAGE'),
])
def test_allocated_plan_stable_across_updates(box, angular, expected_backend):
    r = jnp.asarray([[x, y, z] for x in (0., 2., 4.)
                     for y in (0., 2., 4.) for z in (0., 2., 4.)])
    planner = PlannedNeighbors(model(4., angular), box, np.zeros(len(r), np.int32),
                               skin=.4, disable_cell_list=True)
    neighbor = planner.allocate(r)
    owner = neighbor.owner
    child_owner = None if neighbor.angular is None else neighbor.angular.owner
    assert owner.backend == expected_backend
    assert (child_owner is not None) == (angular < 4.)
    if child_owner is not None:
        assert child_owner.backend == 'STANDARD'
    if expected_backend == 'MULTI_IMAGE':
        assert owner.representation == partition.Dense
    before = (owner.representation, neighbor.idx.shape, tuple(planner.plans))
    update = jax.jit(planner.update)
    for displacement in (.01, .3, .8):
        neighbor = update(r + displacement, neighbor)
        assert neighbor.owner is owner
        assert (None if neighbor.angular is None else neighbor.angular.owner) is child_owner
        assert (owner.representation, neighbor.idx.shape, tuple(planner.plans)) == before
        assert not neighbor.did_buffer_overflow


def test_plan_is_immutable_and_identity_free():
    plan = ExecutionPlan(False, partition.Dense)
    with pytest.raises(FrozenInstanceError):
        plan.separate_angular = True
    assert set(inspect.signature(PlannedNeighbors).parameters) == {
        'model', 'box', 'species', 'settings'}
    # The minimal model above deliberately has no filename or identity metadata.
    # Source audit: N occurs only in capacity/index arithmetic, never equality
    # against an atom-count constant. Keep that contract mechanically checked.
    import ast
    tree = ast.parse(inspect.getsource(policy))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(isinstance(x, ast.Name) and x.id == 'n' for x in operands):
                assert not any(isinstance(x, ast.Constant) and isinstance(x.value, int)
                               for x in operands)
