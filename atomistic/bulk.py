import numpy as np

from atomistic.atomistic import AtomisticStructure
from atomistic.crystal import CrystalBox, CrystalStructure


class BulkCrystal(AtomisticStructure):
    """Class to represent a bulk crystal."""

    def __init__(self, as_params, repeats):
        """Constructor method for BulkCrystal object."""

        super().__init__(**as_params)
        self.repeats = repeats
        self.meta.update({'supercell_type': ['bulk']})

    @classmethod
    def from_crystal_structure(cls, crystal_structure, repeats=None, overlap_tol=1,
                               tile=None):
        """Generate a BulkCrystal object given a `CrystalStructure` object and
        an integer array of column vectors representing the multiplicity of each new
        edge vector.

        Parameters
        ----------
        crystal_structure : CrystalStructure
        repeats : ndarray of int of shape (3, 3), optional
            By default, set to the identity matrix.
        tile : sequence of int of length 3, optional

        """

        if repeats is None:
            repeats = np.eye(3)

        # TODO: validate repeats array (no identical columns).

        cs = CrystalStructure.init_crystal_structures([crystal_structure])[0]

        supercell = np.dot(cs.lattice.unit_cell, repeats)
        crystal_box = CrystalBox(cs, box_vecs=supercell)
        for sites in list(crystal_box.sites.values()):
            sites.basis = None

        as_params = {
            'supercell': supercell,
            'crystals': [crystal_box],
            'overlap_tol': overlap_tol,
            'tile': tile,
        }
        bulk_crystal = cls(as_params, repeats)

        return bulk_crystal
