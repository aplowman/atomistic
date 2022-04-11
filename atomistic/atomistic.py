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

from progress.bar import Bar

from atomistic import ATOM_JMOL_COLOURS
from atomistic.utils import get_column_vector
from atomistic.crystal import CrystalStructure
from atomistic.voronoi import VoronoiTessellation, in_hull


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


def get_volumetric_data_grid(grid_size, supercell, grid_values=None, periodic=False):
    """Get uniform grid indices.

    Returns
    -------
    ndarray of shape (3, N)        

    """
    if grid_values is not None:
        if grid_values.shape[0] != np.prod(grid_size):
            msg = ('The outer shape of `grid_values` should be equal to the product of '
                   'the `grid_size`.')
            raise ValueError(msg)

    # Get cartesian coordinates of grid:
    A, B, C = np.meshgrid(*[np.arange(i) for i in grid_size])
    Af = A / grid_size[0]
    Bf = B / grid_size[1]
    Cf = C / grid_size[2]
    grid_frac = np.vstack([Af.flatten(), Bf.flatten(), Cf.flatten()])
    grid = supercell @ grid_frac

    if not periodic:
        if grid_values is not None:
            return (grid, grid_values)
        else:
            return grid

    else:

        x, y, z = np.meshgrid(*[[-1, 0, 1]] * 3)
        sup_trans = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        translation = np.dot(supercell, sup_trans)
        grid_periodic = np.concatenate(translation.T[:, :, None] + grid, axis=1)

        if grid_values is not None:
            grid_values_periodic = np.tile(grid_values, (27,))
            return (grid_periodic, grid_values_periodic)
        else:
            return grid_periodic


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

        self.tessellation = None            # Set in `set_voronoi_tessellation`
        self.atom_site_geometries = None    # Set in `set_atom_site_geometries`
        self.volumetric_data = {}           # Updated in `add_volumetric_data`
        self.binned_volumetric_data = {}    # Updated in `bin_volumetric_data` with method "grid"

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

        lines = {}
        if self.tessellation:
            include_atoms = None
            show_vertices = False
            show_ridges = True
            points.update(self.tessellation.get_geometry_group_points(
                include_atoms, show_vertices, show_ridges))
            lines.update(self.tessellation.get_geometry_group_lines(
                include_atoms, show_vertices, show_ridges))

        if self.volumetric_data:
            for name, dat in self.volumetric_data.items():
                # Get cartesian coordinates of grid:
                grid = get_volumetric_data_grid(dat['grid_size'], self.supercell)

                # Sub sample; max 1000 points:
                step = int(grid.shape[1] / 1000) or 1
                grid = grid[:, ::step]
                grid_sites = Sites(grid)
                points.update({name: grid_sites})

                # Add periodic grid too:
                grid_per = get_volumetric_data_grid(
                    dat['grid_size'], self.supercell, periodic=True)
                step_per = int(grid_per.shape[1] / 1000) or 1
                grid_per = grid_per[:, ::step_per]
                grid_per_sites = Sites(grid_per)
                points.update({name + '_periodic': grid_per_sites})

        gg = GeometryGroup(points=points, boxes=boxes, lines=lines)

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

            visual_args.update({
                'style_points': {
                    'lattice_sites': {
                        'marker_symbol': 'cross',
                        'marker_size': 5,
                        'fill_colour': 'gray',
                    },
                    'interstices': {
                        'marker_symbol': 'square-open',
                        'marker_size': 4,
                        'fill_colour': 'pink',
                    },
                },
            })

        if self.volumetric_data:
            for k, v in self.volumetric_data.items():

                # Sub sample; max 1000 points:
                total_len = np.product(v['grid_size'])
                step = int(total_len / 1000) or 1
                dat = v['data'][::step]
                visual_args['style_points'].update({
                    k: {'fill_colour': dat, 'marker_size': 2, },
                })

                _, grid_per_dat = get_volumetric_data_grid(
                    v['grid_size'], self.supercell, grid_values=v['data'], periodic=True)
                step_per = int(grid_per_dat.shape[0] / 1000) or 1
                dat_per = grid_per_dat[::step_per]
                visual_args['style_points'].update({
                    (k + '_periodic'): {'fill_colour': dat_per, 'marker_size': 2, },
                })

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

    def set_voronoi_tessellation(self):
        'Perform a Voronoi tessellation of the atoms within the periodic supercell.'
        if not self.tessellation:
            self.tessellation = VoronoiTessellation(self.supercell, self.atoms.coords)

    def _get_atom_neighbours(self):

        neighbours = []
        for atom_idx in range(len(self.atoms)):

            facets_idx = self.tessellation.point_facets[atom_idx]
            fac_pts = self.tessellation.facet_points[facets_idx]
            fac_pts_per = self.tessellation.facet_points_periodic[facets_idx]
            fac_pts_ext = self.tessellation.ridge_points_external_idx[facets_idx]
            neigh_atom_idx = fac_pts_per[fac_pts != atom_idx]
            is_external = (fac_pts[fac_pts != atom_idx] == -1)

            neighbours.append({
                'atom_idx': neigh_atom_idx,
                'is_external': is_external,
                'external_atom_idx': fac_pts_ext,
                'area': self.tessellation.facet_areas[facets_idx],
                'length': self.tessellation.neighbour_distances[facets_idx],
                'vectors': self.tessellation.neighbour_vectors[facets_idx],
            })

        return neighbours

    def set_atom_site_geometries(self):
        if not self.atom_site_geometries:
            self.set_voronoi_tessellation()
            neighbours = self._get_atom_neighbours()
            self.atom_site_geometries = {
                'volume': self.tessellation.point_volumes,
                'neighbours': neighbours,
            }

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

    def same_atoms(self, structure):
        'Check if the supercell and atoms and the same as in another AtomisticStructure.'

        # Supercell:
        if not np.allclose(self.supercell, structure.supercell):
            return False

        # Atoms:
        atoms_srt_idx = np.lexsort(np.round(self.atoms.coords, decimals=9))
        atoms = self.atoms.coords[:, atoms_srt_idx]

        struct_atoms_srt_idx = np.lexsort(np.round(structure.atoms.coords, decimals=9))
        struct_atoms = structure.atoms.coords[:, struct_atoms_srt_idx]

        # Wrap atoms to within the supercell:
        atoms_sup = np.linalg.inv(self.supercell) @ atoms
        struct_atoms_sup = np.linalg.inv(structure.supercell) @ struct_atoms

        boundary_idx = structure.boundary_idx
        struct_atoms_sup[boundary_idx] -= np.floor(struct_atoms_sup[boundary_idx])
        atoms_sup[boundary_idx] -= np.floor(atoms_sup[boundary_idx])

        bad_atoms_idx = np.where(np.abs(atoms_sup - struct_atoms_sup) > 1e-6)[1]

        if not np.allclose(struct_atoms_sup, atoms_sup):
            return False

        return True

    def add_volumetric_data(self, name, grid_size, data):
        """Add a uniform grid of volumetric data.

        Parameters
        ----------
        name : str
        grid_size : tuple of length three
            The size of the grid, along the three supercell directions.
        data : ndarray of outer shape `grid_size`
            Data to assign to the grid points. The outer shape must be equal to
            `grid_size`. The grid is assumed to occupy the whole supercell.

        """

        if name in self.volumetric_data:
            raise ValueError(f'Volumetric data by the name "{name}" is already assigned.')

        if list(data.shape[:len(grid_size)]) != list(grid_size):
            raise ValueError(f'`data` must be an ndarray whose outer shape is equal to '
                             f'`grid_size`.')

        self.volumetric_data.update({
            name: {
                'grid_size': grid_size,
                'data': data.flatten(),
            }
        })

    def set_bond_regions(self, method='polyhedron'):

        if not self.tessellation:
            self.set_voronoi_tessellation()

        self.tessellation.set_bond_regions(method=method)

    def bin_volumetric_data(self, name, method, **kwargs):
        """Bin scalar volumetric data over the supercell.

        If method="bonds_polyhedron", this adds the bin values to the tessellation
        object's `bond_regions` attribute; if method="atoms_voronoi", this adds a list to
        the tessellation object's binned_volumetric_data attribute; if method="grid", this
        adds a list to the AtomisticStructure.binned_volumetric_data attribute.

        Parameters
        ----------
        name : str
            Name of volumetric data to bin.
        method : str
            Method by which to define bins. If "bonds_polyhedron", use the bond regions
            as defined with `set_bond_regions`, using `method=polyhedron`. If
            "atoms_voronoi", use the Voronoi regions of the atoms. If "grid", define a
            uniform grid over the supercell and bin into each grid cell.
        kwargs : dict
            grid_size : list of int
                If `method` is "grid", specify the grid_size as an additional keyword
                parameter here.

        """

        if not self.tessellation:
            self.set_voronoi_tessellation()

        # TODO: add another method: "grid" for binning over a uniform grid of a given resolution.

        if name not in self.volumetric_data:
            raise ValueError(f'No volumetric data named "{name}" exists.')

        allowed_methods = ['bonds_polyhedron', 'atoms_voronoi', 'grid']
        if method not in allowed_methods:
            raise ValueError(f'`method` must be one of: {allowed_methods}.')

        vol_grid_size = self.volumetric_data[name]['grid_size']
        vol_grid_data = self.volumetric_data[name]['data']

        # Get cartesian coordinates of grid:
        grid_per, grid_vals_per = get_volumetric_data_grid(
            vol_grid_size,
            self.supercell,
            grid_values=vol_grid_data,
            periodic=True
        )

        if 'grid_idx_in_volume_count' not in self.volumetric_data[name]:
            # No binning has previously been performed of this data:
            self.volumetric_data[name].update({'grid_idx_in_volume_count': {}})

        elif method in self.volumetric_data[name]['grid_idx_in_volume_count']:
            if (
                method != 'grid' or
                method == 'grid' and (
                    tuple(kwargs['grid_size']) in
                    self.volumetric_data[name]['grid_idx_in_volume_count']['grid']
                )
            ):
                # This binning has already been done:
                msg = (f'Volumetric data "{name}" has already been binned over the '
                       f'supercell using method "{method}" and kwargs: {kwargs}.')
                raise ValueError(msg)

        # Use this to check a sensible binning. grid_idx_in_volume_count counts how
        # many times each grid point has been found to be within a bond region. Note
        # that we might expect this to be an array of ones. However, since the grid is
        # periodic (3x3x3 supercells) and the bond regions barely overlap the central
        # supercell, there will be a large number of zeros in this array. Also, we
        # would expect a relatively small number of twos in this array, where a given
        # grid point exists at the boundary between two bond regions.
        grid_idx_in_volume_count = np.zeros((grid_per.shape[1],), dtype=int)

        # Get indices of volumetric data that is contained within each sub-volume:
        if method == 'bonds_polyhedron':

            if (
                not self.tessellation.bond_regions or
                'polyhedron' not in self.tessellation.bond_regions
            ):
                msg = 'Bond regions have not been defined using the method "polyhedron".'
                raise ValueError(msg)

            num_regs = len(self.tessellation.bond_regions['polyhedron'])
            bar = Bar('Processing', max=num_regs)
            bar.check_tty = False

            for reg_idx, reg in enumerate(self.tessellation.bond_regions['polyhedron']):

                grid_idx_in_volume_bool = in_hull(grid_per.T, reg['vertices'].T)
                grid_idx_in_volume = np.where(grid_idx_in_volume_bool)[0]
                bin_value = np.sum(grid_vals_per[grid_idx_in_volume_bool])

                if 'binned_volumetric_data' not in reg:
                    reg.update({'binned_volumetric_data': {}})

                if name not in reg['binned_volumetric_data']:
                    reg['binned_volumetric_data'].update({name: {}})

                reg['binned_volumetric_data'][name].update({
                    'bin_value': bin_value,
                    'grid_idx_in_volume': grid_idx_in_volume,
                })

                grid_idx_in_volume_count += grid_idx_in_volume_bool
                bar.next()

            bar.finish()

        elif method == 'atoms_voronoi':

            region_verts = self.tessellation.point_vertices
            num_atoms = len(region_verts)
            bar = Bar('Processing', max=num_atoms)
            bar.check_tty = False

            if name not in self.tessellation.binned_volumetric_data:
                self.tessellation.binned_volumetric_data.update({name: []})

            for atom_idx, verts_idx in enumerate(region_verts):
                verts = self.tessellation.vertices[verts_idx]
                grid_idx_in_volume_bool = in_hull(grid_per.T, verts)
                grid_idx_in_volume = np.where(grid_idx_in_volume_bool)[0]
                bin_value = np.sum(grid_vals_per[grid_idx_in_volume_bool])

                self.tessellation.binned_volumetric_data[name].append({
                    'bin_value': bin_value,
                    'grid_idx_in_volume': grid_idx_in_volume,
                })

                grid_idx_in_volume_count += grid_idx_in_volume_bool
                bar.next()

            bar.finish()

        elif method == 'grid':

            grid_cell_dims = self.supercell / kwargs['grid_size']
            grid_cell_origins = get_volumetric_data_grid(
                grid_size=kwargs['grid_size'],
                supercell=self.supercell,
            )
            bar = Bar('Processing', max=grid_cell_origins.shape[1])
            bar.check_tty = False

            if name not in self.binned_volumetric_data:
                self.binned_volumetric_data.update({name: {}})

            grid_size_tuple = tuple(kwargs['grid_size'])
            if grid_size_tuple not in self.binned_volumetric_data[name]:
                self.binned_volumetric_data[name].update({grid_size_tuple: []})

            for origin in grid_cell_origins.T:
                verts = geometry.get_box_corners(grid_cell_dims, origin[:, None])[0]
                centre = np.mean(verts, axis=1)

                grid_idx_in_volume_bool = in_hull(grid_per.T, verts.T)
                grid_idx_in_volume = np.where(grid_idx_in_volume_bool)[0]
                bin_value = np.sum(grid_vals_per[grid_idx_in_volume_bool])

                self.binned_volumetric_data[name][grid_size_tuple].append({
                    'grid_cell_origin': origin,
                    'grid_cell_centre': centre,
                    'bin_value': bin_value,
                    'grid_idx_in_volume': grid_idx_in_volume,
                })

                grid_idx_in_volume_count += grid_idx_in_volume_bool
                bar.next()

            grid_idx_in_volume_count = {
                **self.volumetric_data[name]['grid_idx_in_volume_count'].get('grid', {}),
                grid_size_tuple: grid_idx_in_volume_count,
            }
            bar.finish()

        self.volumetric_data[name]['grid_idx_in_volume_count'].update({
            method: grid_idx_in_volume_count,
        })


class AtomisticSimulation(object):
    'Class to store the results of an atomistic simulation on an AtomisticStructure.'

    def __init__(self, structure, data=None, atom_displacements=None,
                 supercell_displacements=None):
        """
        Parameters
        ----------
        structure : AtomisticStructure
            Initial structure at the start of the simulation.
        data : dict of (str: ndarray of outer shape (N,)), optional
            Additional data to associated with each relaxation step. Each key must be an
            ndarray of outer shape (N,) for N relaxation steps.            
        atom_displacements : ndarray of shape (N, M, 3), optional
            Stack of arrays of column vectors representing the displacements of the atoms
            from their original positions at each relaxation step. `N` is the number of
            relaxation steps and `M` is the number of atoms.
        supercell_displacements : ndarray of shape (N, 3, 3), optional
            Stack of arrays of three column vectors representing the displacements of
            the supercell edge vectors from their original positions at each relaxation
            step. `N` is the number of relaxation steps.

        """

        self.structure = structure
        self.atom_displacements = atom_displacements
        self.supercell_displacements = supercell_displacements
        self.data = data or {}

        self._validate()

    @property
    def num_steps(self):
        return self.atom_displacements.shape[0]

    def _validate(self):

        # Check `ndarray`s:
        if self.atom_displacements is None:
            self.atom_displacements = np.zeros((1, 3, len(self.structure.atoms)))
        elif not isinstance(self.atom_displacements, np.ndarray):
            msg = '`atom_displacements` must be a Numpy ndarray.'
            raise ValueError(msg)

        num_steps = self.atom_displacements.shape[0]
        num_atoms = self.atom_displacements.shape[2]

        if self.supercell_displacements is None:
            self.supercell_displacements = np.zeros((num_steps, 3, 3))
        elif not isinstance(self.supercell_displacements, np.ndarray):
            msg = '`supercell_displacements` must be a Numpy ndarray.'
            raise ValueError(msg)

        # Check same number of steps/atoms:
        if self.supercell_displacements.shape[0] != num_steps:
            msg = ('Inconsistent number of optimisation steps: `atom_displacements` has '
                   '{} but `supercell_displacements has {}.')
            raise ValueError(msg.format(num_steps, self.supercell_displacements.shape[0]))

        msg = 'Data value for key "{}" must be an ndarray or list of outer shape ({},)'
        for k, v in self.data.items():
            if isinstance(v, (np.ndarray, list)):
                if len(v) != num_steps:
                    msg += ' but has length {}.'
                    raise ValueError(msg.format(k, num_steps, len(v)))
            else:
                msg += '.'
                raise ValueError(msg.format(k, num_steps))

    def get_step(self, opt_step=-1, atom_site_geometries=True):
        'Get the AtomisticStructure and data from a given step in the relaxation process.'

        structure = copy.deepcopy(self.structure)
        structure.atom_site_geometries = None
        structure.tessellation = None

        structure.atoms += self.atom_displacements[opt_step]
        structure.supercell += self.supercell_displacements[opt_step]

        data = {k: v[opt_step] for k, v in self.data.items()}

        step = {
            'structure': structure,
            'data': data
        }

        if atom_site_geometries:
            structure.set_atom_site_geometries()
            step.update({
                'atom_site_geometries': structure.atom_site_geometries,
            })

        return step
