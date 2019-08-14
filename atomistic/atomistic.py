"""`atomistic.atomistic.py`"""

import warnings
from itertools import combinations

import numpy as np
import mendeleev
import spglib
from vecmaths import rotation, geometry
from gemo import GeometryGroup, Box, Sites

from atomistic.visualise import visualise_structure
from atomistic.utils import get_column_vector
from atomistic.crystal import CrystalStructure


def get_vec_distances(vecs):
    """
    Find the Euclidean distances between all unique column vector
    pairs.

    TODO: add to vecmaths, possibly with `squared=False` parameter for speed.

    Parameters
    ----------
    vecs : ndarray of shape (3, N)

    Returns
    -------
    ndarray of shape (N, )

    """

    # Get indices of unique pairs:
    idx = np.array(list(combinations(range(vecs.shape[1]), 2))).T
    b = vecs.T[idx]
    dist = np.sqrt(np.sum((b[0] - b[1])**2, axis=1))

    return dist


class AtomisticStructureException(Exception):
    pass


class AtomisticStructure(object):
    'Class to represent crystals of atoms'

    # TODO:
    # - store sites in sites dict attribute sites['atoms'] etc
    # - fix methods that deal with `self.crystals` (must use Crystal methods)
    # - fix method `tile_supercell` (`check_overlapping_atoms` will then work)

    def __init__(self, supercell, sites, origin=None, crystals=None,
                 crystal_structures=None, overlap_tol=1, tile=None):
        """Constructor method for AtomisticStructure object."""

        if origin is None:
            origin = np.zeros((3, 1))

        if crystals is None:
            crystals = []

        if crystal_structures is None:
            crystal_structures = []

        self.origin = origin
        self._sites = self._init_sites(sites)
        self.supercell = supercell
        self.meta = {}
        self.crystal_structures = CrystalStructure.init_crystal_structures(
            crystal_structures)
        self.crystals = crystals
        self._overlap_tol = overlap_tol

        # self.check_overlapping_atoms(overlap_tol)

        # Check handedness:
        if self.volume < 0:
            raise ValueError('Supercell does not form a right-handed coordinate system.')

        # if tile:
        #     self.tile_supercell(tile)

    def _init_sites(self, sites):

        allowed_sites = ['atoms', 'lattice_sites', 'interstices']

        if 'atoms' not in sites:
            raise ValueError('`sites` must contain a `Sites` object named "atoms".')

        for name, sites_obj in sites.items():
            if not isinstance(sites_obj, Sites):
                raise ValueError('`sites` must be a dict with `Sites` object values.')
            if name not in allowed_sites:
                raise ValueError('`sites` named "{}" not allowed.'.format(name))
            setattr(self, name, sites_obj)

        return sites

    def translate(self, shift):
        """
        Translate the AtomisticStructure.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """

        shift = get_column_vector(shift)

        self.origin += shift
        for i in self.sites:
            i.translate(shift)

        for crystal in self.crystals:
            crystal.translate(shift)

    def rotate(self, rot_mat):
        """
        Rotate the AtomisticStructure about its origin according to a rotation
        matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to 
            rotate them about a particular axis and angle.

        """

        self.supercell = np.dot(rot_mat, self.supercell)

        for i in self.sites:
            i.rotate(rot_mat, centre=self.origin)

        for crystal in self.crystals:
            crystal.rotate(rot_mat, centre=self.origin)

    def show(self, **kwargs):
        points = {k: v for k, v in self.sites.items()}
        boxes = {'supercell': Box(edge_vectors=self.supercell)}
        for c_idx, c in enumerate(self.crystals):
            boxes.update({
                'crystal {}'.format(c_idx): Box(edge_vectors=c.box_vecs, origin=c.origin)
            })
        gg = GeometryGroup(points=points, boxes=boxes)
        group_points = {
            'atoms': [
                {
                    'label': 'species',
                    'styles': {
                        'fill_colour': {
                            'Zr': 'blue',
                        },
                    },
                },
                {
                    'label': 'species_order',
                    'style': {
                        'outline_colour': {
                            0: 'red',
                            1: 'green',
                        }
                    }
                }
            ],
        }
        style_points = {
            'lattice_sites': {
                'marker_symbol': 'cross',
                'marker_colour': 'gray',
                'marker_size': 5,
            },
            'interstices': {
                'marker_symbol': 'square-open',
                'marker_colour': 'pink',
                'marker_size': 4,
            }
        }
        vis = gg.show(group_points=group_points, style_points=style_points)
        return vis

    def reorient_to_lammps(self):
        """
        Reorient the supercell and its contents to a LAMMPS-compatible
        orientation. Also translate the origin to (0,0,0).

        Returns
        -------
        ndarray of shape (3, 3)
            Rotation matrix used to reorient the supercell and its contents

        """

        # Find rotation matrix which rotates to a LAMMPS compatible orientation
        sup_lmps = rotation.align_xy(self.supercell)
        R = np.dot(sup_lmps, self.supercell_inv)

        # Move the origin to (0,0,0): (I presume this is necessary for LAMMPS?)
        self.translate(-self.origin)

        # Rotate the supercell and its contents by R
        self.rotate(R)

        return R

    def wrap_sites_to_supercell(self, sites='all', dirs=None):
        """
        Wrap sites to within the supercell.

        Parameters
        ----------
        sites : str
            One of "atom", "lattice", "interstice" or "all".
        dirs : list of int, optional
            Supercell direction indices to apply wrapping. Default is None, in
            which case atoms are wrapped in all directions.

        """

        # Validation
        if dirs is not None:
            if len(set(dirs)) != len(dirs):
                raise ValueError('Indices in `dirs` must not be repeated.')

            if len(dirs) not in [1, 2, 3]:
                raise ValueError('`dirs` must be a list of length 1, 2 or 3.')

            for d in dirs:
                if d not in [0, 1, 2]:
                    raise ValueError('`dirs` must be a list whose elements are '
                                     '0, 1 or 2.')

        for name, i in self.sites.items():
            if sites == name or sites == 'all':
                i.basis = self.supercell
                i._coords[:, dirs] -= np.floor(i._coords[:, dirs])
                i.basis = None

    @property
    def sites(self):
        return self._sites

    @property
    def supercell_inv(self):
        return np.linalg.inv(self.supercell)

    @property
    def atom_sites_frac(self):
        return self.atoms.get_coords(new_basis=self.supercell)

    @property
    def spglib_cell(self):
        'Returns a tuple representing valid input for the `spglib` library.'

        cell = (self.supercell.T,
                self.atom_sites_frac.T,
                [mendeleev.element(i).atomic_number for i in self.all_species])
        return cell

    @property
    def num_atoms_per_crystal(self):
        """Computes number of atoms in each crystal, returns a list."""

        if self.crystals is None:
            return None

        num_dict = dict(zip(self.atoms.labels['crystal_idx'].unique_values,
                            self.atoms.labels['crystal_idx'].values_count))
        num_atoms = [value for (key, value) in sorted(num_dict.items())]

        return num_atoms

    @property
    def num_atoms(self):
        """Computes total number of atoms."""
        return len(self.atoms)

    @property
    def num_crystals(self):
        """Returns number of crystals."""
        return len(self.crystals)

    @property
    def reciprocal_supercell(self):
        """Returns the reciprocal supercell as array of column vectors."""

        v = self.supercell
        cross_1 = np.cross(v[:, 1], v[:, 2])
        cross_2 = np.cross(v[:, 0], v[:, 2])
        cross_3 = np.cross(v[:, 0], v[:, 1])

        B = np.zeros((3, 3))
        B[:, 0] = 2 * np.pi * cross_1 / (np.dot(v[:, 0], cross_1))
        B[:, 1] = 2 * np.pi * cross_2 / (np.dot(v[:, 1], cross_2))
        B[:, 2] = 2 * np.pi * cross_3 / (np.dot(v[:, 2], cross_3))

        return B

    def get_kpoint_grid(self, separation):
        """
        Get the MP kpoint grid size for a given kpoint separation.

        Parameters
        ----------
        separation : float or int or ndarray of shape (3, )
            Maximum separation between kpoints, in units of inverse Angstroms.
            If an array, this is the separations in each reciprocal supercell
            direction.

        Returns
        -------
        ndarray of int of shape (3, )
            MP kpoint grid dimensions along each reciprocal supercell
            direction.

        """
        recip_mags = np.linalg.norm(self.reciprocal_supercell, axis=0)
        grid = np.ceil(np.round(recip_mags / (separation * 2 * np.pi), decimals=8))

        return grid.astype(int)

    def get_kpoint_spacing(self, grid):
        """
        Get the kpoint spacing given an MP kpoint grid size.

        Parameters
        ----------
        grid : list of length 3
            Grid size in each of the reciprocal supercell directions.

        Returns
        -------
        ndarray of shape (3, )
            Separation between kpoints in each of the reciprocal supercell
            directions.

        """
        grid = np.array(grid)
        sep = np.linalg.norm(self.reciprocal_supercell, axis=0) / (grid * 2 * np.pi)

        return sep

    @property
    def centroid(self):
        'Get supercell centroid.'
        return geometry.get_box_corners(self.supercell, origin=self.origin).mean(2).T

    def tile_supercell(self, tiles):
        """
        Tile supercell and its sites by some integer factors in each supercell 
        direction.

        Parameters
        ----------
        tiles : tuple or list of length 3 or ndarray of size 3
            Number of repeats in each supercell direction.

        """
        invalid_msg = ('`tiles` must be a tuple or list of three integers '
                       'greater than 0.')

        if isinstance(tiles, np.ndarray):
            tiles = np.squeeze(tiles).tolist()

        if len(tiles) != 3:
            raise ValueError(invalid_msg)

        for t in tiles:
            if not isinstance(t, int) or t < 1:
                raise ValueError(invalid_msg)

        tl_atm, tl_atm_lb = self.get_tiled_sites(
            self.atom_sites, self.atom_labels, tiles)

        self.atom_sites = tl_atm
        self.atom_labels = tl_atm_lb

        if self.lattice_sites is not None:
            tl_lat, tl_lat_lb = self.get_tiled_sites(
                self.lattice_sites, self.lattice_labels, tiles)

            self.lattice_sites = tl_lat
            self.lattice_labels = tl_lat_lb

        if self.interstice_sites is not None:
            tl_int, tl_int_lb = self.get_tiled_sites(
                self.interstice_sites, self.interstice_labels, tiles)

            self.interstice_sites = tl_int
            self.interstice_labels = tl_int_lb

        self.supercell *= tiles

    def get_tiled_sites(self, sites, site_labels, tiles):
        """
        Get sites (atoms, lattice, interstice) and site labels tiled by some
        integer factors in each supercell direction.

        Sites are tiled in the positive supercell directions.

        Parameters
        ----------
        tiles : tuple or list of length 3 or ndarray of size 3
            Number of repeats in each supercell direction.

        Returns
        -------
        sites_tiled : ndarray
        labels_tiled : dict

        """

        invalid_msg = ('`tiles` must be a tuple or list of three integers '
                       'greater than 0.')

        if isinstance(tiles, np.ndarray):
            tiles = np.squeeze(tiles).tolist()

        if len(tiles) != 3:
            raise ValueError(invalid_msg)

        sites_tiled = np.copy(sites)
        labels_tiled = {k: tuple(np.copy(i) for i in v)
                        for k, v in site_labels.items()}

        for t_idx, t in enumerate(tiles):

            if t == 1:
                continue

            if not isinstance(t, int) or t < 1:
                raise ValueError(invalid_msg)

            v = self.supercell[:, t_idx:t_idx + 1]
            v_range = v * np.arange(1, t)

            all_t = v_range.T[:, :, np.newaxis]

            sites_stack = all_t + sites_tiled
            add_sites = np.hstack(sites_stack)
            sites_tiled = np.hstack([sites_tiled, add_sites])

            labels_tiled_new = {}
            for k, v in labels_tiled.items():

                add_label_idx = np.tile(v[1], t - 1)
                new_label_idx = np.concatenate((v[1], add_label_idx))
                labels_tiled_new.update({
                    k: (v[0], new_label_idx)
                })

            labels_tiled = labels_tiled_new

        return sites_tiled, labels_tiled

    def get_interatomic_dist(self, periodic=True):
        """
        Find the distances between unique atom pairs across the whole
        structure.

        Parameters
        ----------
        periodic : bool
            If True, the atom sites are first tiled in each supercell direction
            to ensure that distances between periodic cells are considered.
            Currently, this is crude, and so produces interatomic distances
            between like atoms (i.e. of one supercell vector length).

        Returns
        ------
        ndarray of shape (N,)

        TODO:
        -   Improve consideration of periodicity. Maybe instead have a function
            `get_min_interatomic_dist` which gets the minimum distances of each
            atom and every other atom, given periodicity.

        """
        if periodic:
            atms = self.get_tiled_sites(
                self.atom_sites, self.atom_labels, [2, 2, 2])[0]
        else:
            atms = self.atom_sites

        return get_vec_distances(atms)

    def check_overlapping_atoms(self, tol=None):
        """
        Checks if any atoms are overlapping within a tolerance.

        Parameters
        ----------
        tol : float, optional
            Distance below which atoms are considered to be overlapping. By,
            default uses the value assigned on object initialisation as
            `_overlap_tol`.

        Raises
        ------
        AtomisticStructureException
            If any atoms are found to overlap within `tol`.

        """
        if tol is None:
            tol = self._overlap_tol

        dist = self.get_interatomic_dist()
        if np.any(dist < tol):
            raise AtomisticStructureException('Found overlapping atoms. '
                                              'Minimum separation: '
                                              '{:.3f}'.format(np.min(dist)))

    def get_sym_ops(self):
        return spglib.get_symmetry(self.spglib_cell)

    def shift_sites(self, shift, wrap=False):
        """
        Perform a rigid shift on all sites, in fractional supercell coordinates.

        Parameters
        ----------
        shift : list or tuple of length three or ndarry of size (3,) of float
            Fractional supercell coordinates to translate all atoms by.
        wrap : bool
            If True, wrap atoms to within the supercell edges after shift.
        """

        shift = get_column_vector(shift)
        shift_std = np.dot(self.supercell, shift)

        for i in self.sites.values():
            i += shift_std

        if wrap:
            self.wrap_sites_to_supercell(sites='all')

    def add_vac(self, thickness, dir_idx, position=1):
        """
        Extend the supercell in a given direction.

        Supercell vector given by direction index `dir_idx` is extended such
        that it's component in the direction normal to the other two supercell
        vectors is a particular `thickness`.

        Parameters
        ----------
        thickness : float
            Thickness of vacuum to add
        dir_idx : int 0, 1 or 2
            Supercell direction in which to add vacuum
        position : float
            Fractional coordinate along supercell vector given by `dir_idx` at
            which to add the vacuum. By default, adds vacuum to the far face of
            the supercell, such that atom Cartesian coordinates are not
            affected. Must be between 0 (inclusive) and 1 (inclusive).
        """

        # TODO: validate it does what we want. Maybe revert back to calling it
        # `add_surface_vac`.

        warnings.warn('!! Untested function... !!')

        if dir_idx not in [0, 1, 2]:
            raise ValueError('`dir_idx` must be 0, 1 or 2.')

        if position < 0 or position > 1:
            raise ValueError('`position` must be between 0 (inclusive) and 1 '
                             '(inclusive).')

        non_dir_idx = [i for i in [0, 1, 2] if i != dir_idx]
        v1v2 = self.supercell[:, non_dir_idx]
        v3 = self.supercell[:, dir_idx]

        n = np.cross(v1v2[:, 0], v1v2[:, 1])
        n_unit = n / np.linalg.norm(n)
        v3_mag = np.linalg.norm(v3)
        v3_unit = v3 / v3_mag
        d = thickness / np.dot(n_unit, v3_unit)

        v3_mag_new = v3_mag + d
        v3_new = v3_unit * v3_mag_new

        self.supercell[:, dir_idx] = v3_new

        asf = self.atom_sites_frac
        shift_idx = np.where(asf[dir_idx] > position)[0]

        self.atom_sites[:, shift_idx] += (n_unit * thickness)

    def check_atomic_environment(self, checks_list):
        """Invoke checks of the atomic environment."""

        allowed_checks = {
            'atoms_overlap': self.check_overlapping_atoms,
        }

        for chk, func in allowed_checks.items():
            if chk in checks_list:
                func()

    @property
    def volume(self):
        """Get the volume of the supercell."""
        vol = np.dot(np.cross(self.supercell[:, 0],
                              self.supercell[:, 1]),
                     self.supercell[:, 2])
        return vol
