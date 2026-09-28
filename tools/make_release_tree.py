"""Export an explicit standalone source tree, without Git history or local artifacts.

Usage: python tools/make_release_tree.py /path/to/new/directory
The destination must not exist. The manifest is intentionally an allowlist.
"""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ('.gitignore', '.gitattributes', 'LICENSE', 'README.md', 'pyproject.toml',
         'MANIFEST.in', 'tools/make_release_tree.py', 'tools/extract_tables.py',
         'tools/extract_invariants.py', 'tests/tolerances.json')
PATTERNS = ('src/jax_nep/*.py', 'tests/test_*.py', 'tests/conftest.py',
            'tests/fixtures/*.npz', 'docs/*.md', 'examples/*.py')


def export(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Destination must be new: {destination}')
    files = set(ROOT / name for name in FILES)
    for pattern in PATTERNS:
        matches = set(ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(f'Empty release pattern: {pattern}')
        files.update(matches)
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Release entry must be a regular file: {path}')
    destination.mkdir(parents=True)
    for source in sorted(files):
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(f'Exported {len(files)} files to {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    export(parser.parse_args().destination)
