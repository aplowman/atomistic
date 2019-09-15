"""`atomistic.__init__.py`"""

from pathlib import Path

from atomistic.utils import get_atom_species_jmol_colours


PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR.joinpath('data')

ATOM_JMOL_COLOURS = get_atom_species_jmol_colours(DATA_DIR.joinpath('jmol_colours.txt'))

ENERGY_PER_AREA_UNIT_CONV = 16.02176565  # multiply by this for: eV / Ang^2 to J / m^2
TT_SUPERCELL_TYPE = 'tensile_test_coordinate'
