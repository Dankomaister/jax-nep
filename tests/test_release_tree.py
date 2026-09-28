"""The source distribution and public export stay independent of local assets."""
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest(tmp_path):
    target = tmp_path / 'public'
    subprocess.run([sys.executable, str(ROOT / 'tools/make_release_tree.py'), str(target)],
                   check=True)
    for required in ('src/jax_nep/__init__.py', 'README.md', 'LICENSE',
                     'tests/fixtures/C.npz', 'tests/tolerances.json', 'docs/api.md'):
        assert (target / required).is_file()
    forbidden = {'assets', 'benchmarks', 'release', '.git', '.work', '__pycache__'}
    for path in target.rglob('*'):
        assert not forbidden.intersection(path.relative_to(target).parts)
        assert not path.name.startswith('.work-')
        if path.is_file():
            assert path.read_bytes() == (ROOT / path.relative_to(target)).read_bytes()
    assert {p.name for p in (target / 'tools').iterdir()} == {
        'make_release_tree.py', 'extract_tables.py', 'extract_invariants.py'}
    # Refuse to merge with a destination that might already contain local files.
    retry = subprocess.run([sys.executable, str(ROOT / 'tools/make_release_tree.py'),
                            str(target)], capture_output=True)
    assert retry.returncode != 0


def test_package_metadata():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['version'] == '0.1.0'
    assert project['license'] == 'GPL-3.0-or-later'
    assert {x.split('>=')[0] for x in project['dependencies']} == {'jax', 'jax-md', 'numpy'}
    assert 'test' in project['optional-dependencies']
