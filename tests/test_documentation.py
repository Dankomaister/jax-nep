"""Execute every public Markdown Python block, not a separately maintained copy."""
from pathlib import Path
import re
import json
import runpy
import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]
TOLERANCES = json.loads((ROOT / 'tests/tolerances.json').read_text())


def prepare_model(tmp_path, name='synthetic_l4'):
    data = np.load(ROOT / 'tests/fixtures' / (name + '.npz'))
    (tmp_path / 'nep.txt').write_text(str(data['model_text']))


def check_workflow(namespace):
    if 'energy' not in namespace:
        return
    for key in ('energy', 'forces', 'local_energies', 'W', 'sigma'):
        assert np.isfinite(namespace[key]).all(), key
    assert namespace['forces'].shape == (2, 3)
    assert namespace['W'].shape == namespace['sigma'].shape == (3, 3)
    # Eager and fused JIT reductions may round differently: reuse the unchanged
    # per-atom reference budget instead of NumPy's float64-oriented default.
    rtol, atol = TOLERANCES['energy']
    n = len(namespace['R'])
    np.testing.assert_allclose(namespace['energy'] / n,
                               namespace['local_energies'].sum() / n,
                               rtol=rtol, atol=atol)
    np.testing.assert_allclose(namespace['sigma'], -namespace['W'] / namespace['box']**3)
    if 'state' in namespace:
        state, neighbor = namespace['state'], namespace['neighbor']
        assert not neighbor.did_buffer_overflow
        force = -jax.grad(lambda x: namespace['energy_fn'](x, neighbor=neighbor))(state.position)
        np.testing.assert_allclose(state.force, force, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_markdown_python(page, tmp_path, monkeypatch):
    blocks = re.findall(r'^```python\n(.*?)^```', page.read_text(), re.M | re.S)
    if not blocks:
        return
    # README exercises an actual fitted carbon model using release fixtures only.
    prepare_model(tmp_path, 'C' if page.name == 'README.md' else 'synthetic_l4')
    monkeypatch.chdir(tmp_path)
    namespace = {}
    if page.name == 'virial_stress.md':
        namespace = runpy.run_path(str(ROOT / 'examples/basic.py'))
    for i, block in enumerate(blocks):
        exec(compile(block, f'{page.name}:block{i+1}', 'exec'), namespace)
    check_workflow(namespace)
    jax.clear_caches()


@pytest.mark.parametrize('name', ['basic.py', 'nve.py'])
def test_example_script(name, tmp_path, monkeypatch):
    prepare_model(tmp_path)
    monkeypatch.chdir(tmp_path)
    check_workflow(runpy.run_path(str(ROOT / 'examples' / name)))
    jax.clear_caches()


def test_documentation_links_and_language():
    for page in PAGES:
        text = page.read_text()
        assert not re.search(r'\b(rewrite|prototype|Codex|campaigns?)\b', text, re.I)
        for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', text):
            if '://' not in link:
                assert (page.parent / link).exists(), (page, link)
