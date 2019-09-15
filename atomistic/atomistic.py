"""`atomistic.atomistic.py`"""

import warnings
from itertools import combinations
import copy
from pprint import pprint

import numpy as np
import mendeleev
import spglib
from vecmaths import rotation, geometry
from vecmaths.utils import snap_arr
from gemo import GeometryGroup, Box, Sites
from gemo.camera import OrthographicCamera

from atomistic import ATOM_JMOL_COLOURS
from atomistic.utils import get_column_vector
from atomistic.crystal import CrystalStructure


def get_vec_squared_distances(vecs):
    'Find the squared distances between all column vectors.'

    # Get indices of unique pairs:
    idx = np.array(list(combinations(range(vecs.shape[1]), 2))).T
    b = vecs.T[idx]
    dist_sq = np.sum((b[0] - b[1])**2, axis=1)

    return dist_sq


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

    return np.sqrt(get_vec_squared_distances(vecs))


class AtomisticStructureException(Exception):
    pass


class AtomisticStructure(object):
    'Class to represent crystals of atoms'

    atoms = None

    def __init__(self, supercell, sites=None, origin=None, crystals=None,
                 overlap_tol=None, tile=None):
        """Constructor method for AtomisticStructure object."""

        if origin is None:
            origin = np.zeros((3, 1))

        if crystals is None:
            crystals = []

        if sites is None:
            sites = {}

        self.origin = origin
        self._init_sites(sites, crystals)
        self.supercell = supercell
        self.meta = {}

        if overlap_tol:
            self.check_overlapping_atoms(overlap_tol)

        # Check handedness:
        if self.volume < 0:
            raise ValueError('Supercell does not form a right-handed coordinate system.')

        if tile:
            self.tile(tile)

    # @property
    # def supercell(self):
    #     return snap_arr(self._supercell, 0, tol=1e-12)

    # @supercell.setter
    # def supercell(self, supercell):
    #     self._supercell = supercell

    def _init_sites(self, sites, crystals):
        'Merge crystal-less sites with crystal sites and add attributes.'

        # Merge crystal sites:
        all_sites = {}
        if crystals:
            for name in crystals[0].sites.keys():
                combined = Sites.concatenate([i.sites[name] for i in crystals])
                crystal_idx = np.concatenate([[idx] * len(i.sites[name])
                                              for idx, i in enumerate(crystals)])
                combined.add_labels(crystal_idx=crystal_idx)
                all_sites.update({name: combined})
            for i in crystals:
                i.atomistic_structure = self
                i._sites = None

        # Merge crystal-less sites:
        for name, sites_obj in sites.items():
            if not isinstance(sites_obj, Sites):
                raise ValueError('`sites` must be a dict with `Sites` object values.')

            sites_obj_copy = sites_obj.copy()
            crystal_idx = np.array([[-1] * len(sites_obj_copy)])
            sites_obj_copy.add_labels(crystal_idx=crystal_idx)

            if name in all_sites:
                all_sites[name] += sites_obj_copy
            else:
                all_sites.update({name: sites_obj_copy})

        # Add attributes:
        for name, sites_obj in all_sites.items():
            sites_obj.parent_visual_handlers.append(self.refresh_visual)
            setattr(self, name, sites_obj)

        self._sites = all_sites
        self.crystals = crystals

        if crystals:
            for i in crystals:
                i._init_sites(i.sites)

    def translate(self, shift):
        """
        Translate the AtomisticStructure.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """

        shift = get_column_vector(shift)
        self.origin += shift

        for i in self.sites.values():
            i.filter(crystal_idx=-1).translate(shift)

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

        for i in self.sites.values():
            i.filter(crystal_idx=-1).rotate(rot_mat, centre=self.origin)

        for crystal in self.crystals:
            crystal.rotate(rot_mat, centre=self.origin)

    def get_geometry_group(self):
        'Get the GeometryGroup object for visualisation.'
        points = {k: v for k, v in self.sites.items()}
        boxes = {'supercell': Box(edge_vectors=self.supercell, origin=self.origin)}
        for c_idx, c in enumerate(self.crystals):
            boxes.update({
                'crystal {}'.format(c_idx): Box(edge_vectors=c.box_vecs, origin=c.origin)
            })
        gg = GeometryGroup(points=points, boxes=boxes)

        return gg

    def _set_default_visual_args(self, visual_args):
        if 'group_points' not in visual_args:
            uniq_species = self.atoms.labels['species'].unique_values
            group_points = {
                'atoms': [
                    {
                        'label': 'species',
                        'styles': {
                            'fill_colour': {
                                species: 'rgb({}, {}, {})'.format(
                                    *ATOM_JMOL_COLOURS[species])
                                for species in uniq_species
                            },
                        },
                    },
                ],
            }
            if 'species_order' in self.sites['atoms'].labels:
                group_points['atoms'].append({
                    'label': 'species_order',
                    'styles': {}
                })
            if self.crystals:
                group_points['atoms'].append({
                    'label': 'crystal_idx',
                    'styles': {},
                })
            visual_args.update({'group_points': group_points})

        if 'style_points' not in visual_args:

            style_points = {
                'lattice_sites': {
                    'marker_symbol': 'cross',
                    'marker_size': 5,
                    'fill_colour': 'gray',
                },
                'interstices': {
                    'marker_symbol': 'square-open',
                    'marker_size': 4,
                    'fill_colour': 'pink',
                }
            }
            visual_args.update({'style_points': style_points})

        return visual_args

    def show(self, layout_args=None, **visual_args):
        gg = self.get_geometry_group()
        visual_args = self._set_default_visual_args(visual_args)
        return gg.show(**visual_args, layout_args=layout_args)

    def show_projection(self, look_at, up, width=None, height=None, depth=None,
                        camera_translate=None, **visual_args):

        geom_group = self.get_geometry_group()
        camera = OrthographicCamera.from_bounding_box(
            geom_group,
            look_at=look_at,
            up=up,
            width=width,
            height=height,
            depth=depth,
            camera_translate=camera_translate,
        )
        geom_proj = geom_group.project(camera)
        visual_args = visual_args or self._get_default_visual_args()
        return geom_proj.show(**visual_args)

    def preview_projection(self, look_at, up, width=None, height=None, depth=None,
                           camera_translate=None, **visual_args):
        geom_group = self.get_geometry_group()
        camera = OrthographicCamera.from_bounding_box(
            geom_group,
            look_at=look_at,
            up=up,
            width=width,
            height=height,
            depth=depth,
            camera_translate=camera_translate,
        )
        geom_proj = geom_group.project(camera)
        visual_args = visual_args or self._get_default_visual_args()
        return geom_proj.preview(**visual_args)

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
                i._coords[dirs] -= np.floor(i._coords[dirs])
                i.basis = None  # reset to standard basis

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

    def tile(self, tiles):
        """Tile by some integer factors in each supercell direction.

        Parameters
        ----------
        tiles : tuple or list of length 3 or ndarray of size 3
            Number of repeats in each supercell direction.

        """
        self._sites = self.get_tiled_sites(tiles)
        self.supercell *= tiles

    def get_tiled_sites(self, tiles):
        """Get sites, tiled by some integer factors in each supercell direction.

        Parameters
        ----------
        tiles : tuple or list of length 3 or ndarray of size 3
            Number of repeats in each supercell direction.

        """

        grid = [list(range(0, i)) for i in tiles]
        origin_coords = np.vstack(np.meshgrid(*grid)).reshape((3, -1))
        origin_sites = Sites(coords=origin_coords, basis=self.supercell)
        origin_sites.basis = None
        tiled_sites = {k: origin_sites.tile(v) for k, v in self.sites.items()}

        return tiled_sites

    def get_squared_interatomic_dist(self, periodic=True):
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
            atoms = self.get_tiled_sites([2, 2, 2])['atoms']._coords
        else:
            atoms = self.atoms._coords

        dist_sq = get_vec_squared_distances(atoms)
        return dist_sq

    def check_overlapping_atoms(self, tol=1):
        """
        Checks if any atoms are overlapping within a tolerance.

        Parameters
        ----------
        tol : float
            Distance below which atoms are considered to be overlapping.

        Raises
        ------
        AtomisticStructureException
            If any atoms are found to overlap within `tol`.

        """

        dist_sq = self.get_squared_interatomic_dist()
        if np.any(dist_sq < tol**2):
            min_dist = np.sqrt(np.min(dist_sq))
            msg = 'Found overlapping atoms. Minimum separation: {:.3f}'.format(min_dist)
            raise AtomisticStructureException(msg)

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

    def refresh_visual(self, obj):
        # print('AS.refresh_visual invoked by object: {}'.format(obj))
        pass
