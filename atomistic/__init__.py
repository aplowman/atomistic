"""`atomistic.__init__.py`"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR.joinpath('data')

print('DATA_DIR: {}'.format(DATA_DIR))
print('is dir? {}'.format(DATA_DIR.is_dir()))
