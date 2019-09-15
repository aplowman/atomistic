"""`atomistic.bicrystal.py`"""

import warnings
import fractions
import json
import copy
from functools import partial
from pprint import pprint
from pathlib import Path

import spglib
import numpy as np
from spatial_sites import Sites

from atomistic import mathsutils, TT_SUPERCELL_TYPE
from atomistic.atomistic import AtomisticStructure
from atomistic.crystal import CrystalBox
from atomistic.utils import fractions_to_common_denom, zeropad


def distance_from_origin(vecs, stretch_origin, stretch_direction):
    # TODO: generalise this nomenclature further?
    return np.einsum('jk,jl->k', vecs - stretch_origin, stretch_direction)


def stretch_box(edge_vectors, box_origin, stretch_func, stretch_origin, stretch_direction):
    """Stretch a box along a given direction.

    Parameters
    ----------
    edge_vectors : ndarray of column vectors
    box_origin : 3-vector
    stretch_func : callable
        Function that returns the change in coordinate for a given coordinate.
    stretch_origin : 3-vector
        Points at the `stretch_origin` are not modified.
    stretch_direction 3-vector

    Returns
    -------
    tuple of (vecs_stretch, org_stretch)

    """

    vecs_full = edge_vectors + box_origin
    vecs_dist = distance_from_origin(vecs_full, stretch_origin, stretch_direction)
    org_dist = distance_from_origin(box_origin, stretch_origin, stretch_direction)

    vecs_dx = stretch_func(vecs_dist)
    org_dx = stretch_func(org_dist)

    org_stretch = box_origin + (org_dx * stretch_direction)
    vecs_stretch = vecs_full + (vecs_dx * stretch_direction) - org_stretch

    return (vecs_stretch, org_stretch)


def stretch_sites(sites, stretch_func, stretch_origin, stretch_direction):
    """
    TODO: must `stretch_direction` be a unit vector?
    """

    dist = distance_from_origin(sites._coords, stretch_origin, stretch_direction)
    sites += stretch_func(dist) * stretch_direction


def requires_base_structure(func):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], GammaSurfaceCoordinate):
            gamma_surface = args[0].gamma_surface
        else:
            gamma_surface = args[0]
        if not hasattr(gamma_surface, '_base_structure'):
            msg = ('This method/property requires the `base_structure` attribute to be '
                   'set on the `GammaSurface`.')
            raise ValueError(msg)
        return func(*args, **kwargs)
    return wrapper


class Bicrystal(AtomisticStructure):
    """
    Class to represent a bicrystal supercell.

    Attributes
    ----------
    atoms_gb_dist : ndarray of shape (N,)
        Perpendicular distances of each atom from the origin boundary plane

    TODO: Finish docstrings.

    """

    def __init__(self, as_params=None, maintain_inv_sym=False, reorient=False,
                 boundary_vac=None, relative_shift=None, wrap=True, non_boundary_idx=None,
                 rot_mat=None, overlap_tol=1):

        # Call parent constructor
        super().__init__(**as_params)

        # Set meta info:
        self.meta = {'supercell_type': ['bicrystal']}

        # Non-boundary (column) index of `box_csl` and grain arrays
        gb_idx = [0, 1, 2]
        gb_idx.remove(non_boundary_idx)

        # Boundary normal vector:
        n = np.cross(self.supercell[:, gb_idx[0]],
                     self.supercell[:, gb_idx[1]])[:, np.newaxis]
        n_unit = n / np.linalg.norm(n)

        # Non-boundary supercell unit vector
        u = self.supercell[:, non_boundary_idx][:, None]
        u_unit = u / np.linalg.norm(u)

        # Set instance Bicrystal-specific attributes:
        self.maintain_inv_sym = maintain_inv_sym
        self.n_unit = n_unit
        self.u_unit = u_unit
        self.non_boundary_idx = non_boundary_idx
        self.boundary_idx = gb_idx
        self.boundary_vac = 0
        self.boundary_vac_type = None
        self.relative_shift = [0, 0]
        self.rot_mat = rot_mat

        # Invoke additional methods:
        if reorient:
            self.reorient_to_xy()

        if boundary_vac is not None:
            for bvac in boundary_vac:
                self.apply_boundary_vac(**bvac)

        if relative_shift is not None:
            self.apply_relative_shift(**relative_shift)

        if wrap:
            self.wrap_sites_to_supercell()

        if overlap_tol:
            # Delay overlap check until after relative shift and boundary vac application.
            self.check_overlapping_atoms(overlap_tol)

    @classmethod
    def from_atoms(cls, atoms, species, supercell, **kwargs):
        """Generate a Bicrystal from atoms and a supercell definition by assuming the two
        crystals are of equal size."""

        if 'non_boundary_idx' not in kwargs:
            msg = ('Specify which column of the supercell represents the out-of-boundary'
                   ' direction using the `non_boundary_idx` argument.')
            raise ValueError(msg)

        non_boundary_idx = kwargs['non_boundary_idx']

        crystal_1_box = np.copy(supercell)
        crystal_1_box[:, non_boundary_idx] /= -2

        # Generate crystal 0 at the "top".
        crystal_0_box = np.copy(supercell)
        crystal_0_box[:, non_boundary_idx] /= +2

        origin = np.copy(crystal_0_box[:, non_boundary_idx])

        atoms_sup = np.linalg.inv(supercell) @ atoms
        atoms_0_idx = np.where(atoms_sup[non_boundary_idx] >= 0.5)[0]
        atoms_1_idx = np.where(atoms_sup[non_boundary_idx] < 0.5)[0]

        sites_0 = {
            'atoms': Sites(
                coords=atoms[:, atoms_0_idx],
                vector_direction='col',
                labels={
                    'species': species[atoms_0_idx],
                },
            ),
        }
        sites_1 = {
            'atoms': Sites(
                coords=atoms[:, atoms_1_idx],
                vector_direction='col',
                labels={
                    'species': species[atoms_1_idx],
                },
            ),
        }

        crystal_0 = CrystalBox(
            box_vecs=crystal_0_box,
            sites=sites_0,
            origin=origin,
        )
        crystal_1 = CrystalBox(
            box_vecs=crystal_1_box,
            sites=sites_1,
            origin=origin,
        )

        as_params = {
            'supercell': supercell,
            'crystals': [crystal_0, crystal_1],
        }

        return cls(as_params, **kwargs)

    @property
    def bicrystal_thickness(self):
        """Get bicrystal thickness in grain boundary normal direction."""
        return np.einsum('ij,ij', self.non_boundary_vec, self.n_unit)

    @property
    def boundary_area(self):
        """Get the grain boundary area, in square Angstroms."""
        return np.linalg.norm(np.cross(self.boundary_vecs[:, 0],
                                       self.boundary_vecs[:, 1]))

    @property
    def boundary_vecs(self):
        """Get the supercell vectors defining the boundary plane."""
        return np.vstack([
            self.supercell[:, self.boundary_idx[0]],
            self.supercell[:, self.boundary_idx[1]]
        ]).T

    @property
    def boundary_vecs_magnitude(self):
        'Get the magnitudes of the boundary vectors.'
        return np.linalg.norm(self.boundary_vecs, axis=0)

    @property
    def non_boundary_vec(self):
        'Get the supercell vector in the out-of-boundary direction.'
        return self.supercell[:, self.non_boundary_idx, None]

    @property
    def non_boundary_vec_magnitude(self):
        'Get the magnitude of the non-boundary vector.'
        return np.linalg.norm(self.non_boundary_vec)

    def distance_from_gb(self, points):
        """
        Computes the distance from each in an array of column vector to the origin grain
        boundary plane.

        # TODO: does this assume supercell origin is at (0,0,0)?

        """
        return np.einsum('jk,jl->k', points, self.n_unit)

    def reorient_to_xy(self):
        """
        Reorient the supercell to a LAMMPS-compatible orientation in such a way that the
        boundary plane is in the xy plane.

        """

        # Reorient so the boundary plane is in xy
        if self.non_boundary_idx != 2:

            # Ensure non-boundary supercell vector is last vector, whilst
            # maintaining handedness of the supercell coordinate system.
            self.supercell = np.roll(self.supercell, (2 - self.non_boundary_idx), axis=1)
            self.non_boundary_idx = 2
            self.boundary_idx = [0, 1]

        rot_mat = super().reorient_to_lammps()

        # Reorient objects which are CSLBicrystal specific
        self.n_unit = np.dot(rot_mat, self.n_unit[:, 0])[:, None]
        self.u_unit = np.dot(rot_mat, self.u_unit[:, 0])[:, None]

        return rot_mat

    def apply_boundary_vac(self, thickness, func, wrap=False, **kwargs):
        """
        Apply vacuum to the Bicrystal in the direction normal to the grain boundary,
        distributed according to one of three functions.

        Parameters
        ----------
        thickness : float
            The length by which the supercell will be extended in the GB normal direction.
            This is not necessarily a supercell vector direction.
        func : str
            One of "sigmoid", "flat" or "linear". Describes how the vacuum should be
            distributed across the supercell in the GB normal direction.
        kwargs : dict
            Additional arguments to pass to the function.

        """

        # Validation
        allowed_func = [
            'sigmoid',
            'flat',
            'linear',
        ]

        if func not in allowed_func:
            msg = ('"{}" is not an allowed function name to use for applying grain '
                   'boundary vacuum.'.format(func))
            raise ValueError(msg)

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type and TT_SUPERCELL_TYPE not in sup_type:
            msg = 'Cannot apply boundary vacuum to a bulk_bicrystal.'
            raise NotImplementedError(msg)

        # For convenience:
        vt = thickness
        bt = self.bicrystal_thickness
        nu = self.n_unit
        uu = self.u_unit

        if func in ['sigmoid', 'flat']:

            if func == 'sigmoid':
                b = kwargs.get('sharpness', 1)
            elif func == 'flat':
                b = 1000

            func_args = {
                'a': thickness,
                'b': b,
                'width': bt,
            }

            if self.maintain_inv_sym:
                func_cll = mathsutils.double_sigmoid
            else:
                func_cll = mathsutils.single_sigmoid

        elif func == 'linear':

            func_args = {
                'm': vt / bt,
            }
            func_cll = mathsutils.linear

        # Callable that returns dx for a given x
        f = partial(func_cll, **func_args)

        stretch_params = {
            'stretch_func': f,
            'stretch_origin': self.origin,
            'stretch_direction': self.n_unit,
        }

        for sites in self.sites.values():
            stretch_sites(sites, **stretch_params)

        # Store info in the meta dict regarding this change in site positions:
        bv_dict = {
            'thickness': vt,
            'func': func,
            'kwargs': kwargs,
        }

        for crystal in self.crystals:
            new_vecs, new_org = stretch_box(crystal.box_vecs, crystal.origin,
                                            **stretch_params)
            crystal.box_vecs = new_vecs
            crystal.origin = new_org

        # Apply vacuum to the supercell
        if self.maintain_inv_sym:
            new_vecs, _ = stretch_box(self.supercell, self.origin, **stretch_params)
            self.supercell = new_vecs

        else:
            vac_add = (uu * vt) / np.einsum('ij,ij->', nu, uu)
            self.supercell[:, self.non_boundary_idx, None] += vac_add

        # Add new meta information:
        bv_meta = self.meta.get('boundary_vac')
        if bv_meta is None:
            self.meta.update({
                'boundary_vac': [bv_dict]
            })
        else:
            bv_meta.append(bv_dict)

        new_vac_thick = self.boundary_vac + vt

        if self.boundary_vac != 0:
            warnings.warn('`boundary_vac` is already non-zero ({}). '
                          'Summing to new value: {}'.format(
                              self.boundary_vac, new_vac_thick))

        new_vac_type = func
        if self.boundary_vac_type is not None:
            new_vac_type = self.boundary_vac_type + '+' + new_vac_type

        # Update attributes:
        self.boundary_vac = new_vac_thick
        self.boundary_vac_type = new_vac_type

        if wrap:
            self.wrap_sites_to_supercell()

    def apply_relative_shift(self, shift, crystal_idx, wrap=False):
        """
        Apply in-boundary-plane shifts to the specified grain to enable an exploration of
        the grain boundary's microscopic degrees of freedom.

        Parameters
        ----------
        shift : ndarray of size two
            A two-element array whose elements are the relative shift in of one crystal
            relative to the other in fractional coordinates of the boundary area.
        crystal_idx : int
            Which crystal to shift. Zero-indexed.
        wrap : bool, optional
            If True, sites (atoms, lattice sites, etc) within the supercell are wrapped to
            the supercell volume after shifting. By default, False. 

        """

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            msg = 'Cannot apply relative shift to a bulk_bicrystal.'
            raise NotImplementedError(msg)

        elif all([i not in sup_type for i in ['bicrystal', 'surface_bicrystal']]):
            msg = 'Cannot apply relative shift to this supercell type.'
            raise NotImplementedError(msg)

        shift_frac_gb = np.array(shift).squeeze()

        if np.any(shift_frac_gb < -1) or np.any(shift_frac_gb > 1):
            raise ValueError('Elements of `shift` should be between -1 and 1.')

        shift_frac = np.zeros(3)
        shift_frac[self.boundary_idx] = shift_frac_gb
        shift_std = self.supercell @ shift_frac[:, None]

        self.crystals[crystal_idx].translate(shift_std)

        if self.relative_shift != [0, 0]:
            warnings.warn('`relative_shift` is already non-zero. Adding new value.')
        self.relative_shift = [i + j for i,
                               j in zip(shift_frac_gb.tolist(), self.relative_shift)]

        if self.maintain_inv_sym:
            # Modify out-of-boundary supercell vector
            self.supercell[:, self.non_boundary_idx, None] += (2 * shift_std)

        if wrap:
            self.wrap_sites_to_supercell()

    def wrap_sites_to_supercell(self, sites='all'):
        """Wrap atoms to within the boundary plane as defined by the supercell."""

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError('Cannot wrap atoms within a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal',
                                              'surface_bicrystal']]):
            raise NotImplementedError('Cannot wrap atoms within this supercell type.')

        super().wrap_sites_to_supercell(sites=sites, dirs=self.boundary_idx)

    def check_inv_symmetry(self):
        """
        Check atoms exhibit inversion symmetry through the two crystal centres, if
        `self.maintain_inv_sym` is True.

        """

        if self.maintain_inv_sym:

            sym_ops = spglib.get_symmetry(self.spglib_cell)
            sym_rots = sym_ops['rotations']
            sym_trans = sym_ops['translations']
            inv_sym_rot = -np.eye(3, dtype=int)
            inv_sym = np.where(np.all(sym_rots == inv_sym_rot, axis=(1, 2)))[0]
            if len(inv_sym) == 0:
                raise ValueError('The bicrystal does not have inversion '
                                 'symmetry.')

    def check_atomic_environment(self, checks_list):
        """Invoke checks of the atomic environment."""

        super().check_atomic_environment(checks_list)

        allowed_checks = {
            'bicrystal_inversion_symmetry': self.check_inv_symmetry,
        }

        for chk, func in allowed_checks.items():
            if chk in checks_list:
                func()

    def swap_crystal_sites(self):
        'Swap sites between crystals.'
        self.crystals[0].translate(-self.non_boundary_vec / 2)
        self.crystals[1].translate(self.non_boundary_vec / 2)


class GammaSurface(object):

    def __init__(self, base_structure, shifts, expansions, data=None, fitted_data=None):
        """

        Parameters
        ----------
        base_structure : Bicrystal
            Base bicrystal whose gamma surface is to be (or was) investigated.
        shifts : ndarray or list of shape (N, 2)
            The set of relative shifts to apply in the directions parallel to the boundary
            vectors of `base_structure`.
        expansions : ndarray of list of shape (N,)
            The set of expansions to apply in the out-of-boundary direction of
            `base_structure`.
        data : dict, optional
            Dict whose keys are strings and labels are ndarrays of outer dimension (N, )
        fitted_data : dict, optional
            Dict whose keys are strings that must exist in `data`, and whose values are
            sub-dicts whose keys are strings and whose values are ndarrays.

        """

        shifts, expansions, data = self._validate(shifts, expansions, data)
        self.base_structure = base_structure
        self.shifts = shifts
        self.expansions = expansions
        self.data = data
        self.fitted_data = self._validate_fitted_data(fitted_data)

        self._absolute_shifts = None

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            if not np.allclose(self.shifts, other.shifts):
                return False
            if not np.allclose(self.expansions, other.expansions):
                return False

            if set(self.data.keys()) != set(other.data.keys()):
                return False

            for k, v in self.data.items():
                if not np.allclose(v, other.data[k]):
                    return False

            if set(self.fitted_data.keys()) != set(other.fitted_data.keys()):
                return False

            for k, v in self.fitted_data.items():
                if set(v.keys()) != set(other.fitted_data[k].keys()):
                    return False
                for fit_key, fit_val in v.items():
                    if not np.allclose(fit_val, other.fitted_data[k][fit_key]):
                        return False

            return True

        else:
            return NotImplemented

    def _validate_fitted_data(self, fitted_data):
        if not fitted_data:
            fitted_data = {}

        for k, v in fitted_data.items():
            # Check keys:
            if k not in self.data:
                msg = 'Fitted data key "{}" is not in data.'
                raise ValueError(msg.format(k))

            # Check sub dict values have consistent lengths and cast to ndarray
            length = None
            for fit_key, fit_val in v.items():
                v[fit_key] = np.array(fit_val)
                if length is None:
                    length = v[fit_key].shape[0]
                elif v[fit_key].shape[0] != length:
                    msg = ('Fit key "{}" of fitted data "{}" does not have the same '
                           'outer shape as its neighbouring keys.')
                    raise ValueError(msg)

        return fitted_data

    @classmethod
    def from_json_file(cls, base_structure, path):
        """Load a gamma surface from a base structure and a JSON file."""

        with Path(path).open() as handle:
            contents = json.load(handle)

        return cls(base_structure, **contents)

    @classmethod
    def from_grid(cls, base_structure, grid, expansions=0):
        """Generate a gamma surface from a base structure and a grid specification at a
        given expansion.

        Parameters
        ----------
        base_structure : Bicrystal
        grid : list of length two
            Number of relative shifts in each boundary vector direction.
        expansions : number or (list or ndarray) of numbers, optional
            Expansion(s) for all shifts in the grid. By default, 0, meaning a single grid
            is added. If a list or ndarray is supplied, multiple grids are added, one for
            each expansion value.

        """

        gamma_surface = cls(base_structure, None, None)
        gamma_surface.add_grid(grid, expansions)

        return gamma_surface

    def add_grid(self, grid, expansions=0):
        """Add a grid of shifts at a given expansion.

        Parameters
        ----------
        grid : list of length two
            Number of relative shifts in each boundary vector direction.
        expansions : number or (list or ndarray) of numbers, optional
            Expansion(s) for all shifts in the grid. By default, 0, meaning a single grid
            is added. If a list or ndarray is supplied, multiple grids are added, one for
            each expansion value.

        """

        if self.data:
            msg = 'Cannot currently add a grid to a gamma surface with existing `data`.'
            raise NotImplementedError(msg)

        try:
            _ = iter(expansions)
        except TypeError:
            expansions = [expansions]

        x, y = np.meshgrid(*[np.arange(i + 1) / i for i in grid])
        shifts = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

        for i in expansions:
            self.add_coordinates(shifts, [i] * shifts.shape[0])

    def add_coordinate(self, shift, expansion, data=None):
        'Add a single coordinate to the gamma surface.'

        if not data:
            data = {}
        data = {name: [val] for name, val in data.items()}
        self.add_coordinates([shift], [expansion], data)

    def add_coordinates(self, shifts, expansions, data=None):
        """Add more coordinates to the gamma surface.

        Parameters
        ----------
        shifts : ndarray or list of shape (N, 2)
        expansions : ndarray of list of shape (N,)

        """

        shifts, expansions, data = self._validate(shifts, expansions, data)

        self.expansions = np.append(self.expansions, expansions)
        self.shifts = np.vstack([self.shifts, shifts])

        if set(data.keys()) != set(self.data.keys()):
            msg = 'New data names must match existing data names.'
            raise ValueError(msg)

        for data_name, data_item in data.items():
            self.data[data_name] = np.concatenate(
                [self.data[data_name], data_item], axis=0)

    def to_dict(self):
        'Generate a JSON-compatible dict of this GammaSurface (except the structure).'
        out = {
            'shifts': self.shifts.tolist(),
            'expansions': self.expansions.tolist(),
            'data': {k: v.tolist() for k, v in self.data.items()},
            'fitted_data': {
                k: {fit_key: fit_val.tolist() for fit_key, fit_val in v.items()}
                for k, v in self.fitted_data.items()},
        }
        return out

    def to_json_file(self, path):
        'Generate a JSON file of this GammaSurface (except the structure).'
        dct = self.to_dict()
        with Path(path).open('w') as handle:
            json.dump(dct, handle, indent=4, sort_keys=True)

        return path

    @property
    @requires_base_structure
    def base_structure(self):
        return self._base_structure

    @base_structure.setter
    def base_structure(self, base_structure):
        if base_structure is not None:
            self._base_structure = self._validate_structure(base_structure)

    @property
    def shifts(self):
        return self.shift_numerators / self.shift_denominators

    @shifts.setter
    def shifts(self, shifts):

        if shifts.size:
            shift_frac = np.array([[fractions.Fraction(i).limit_denominator()
                                    for i in j] for j in shifts])

            x_nums, x_denom = fractions_to_common_denom(shift_frac[:, 0])
            y_nums, y_denom = fractions_to_common_denom(shift_frac[:, 1])

            self._shift_numerators = np.hstack([x_nums[:, None], y_nums[:, None]])
            self._shift_denominators = np.hstack([x_denom, y_denom])

            uniq, inverse = np.unique(self.shift_numerators, axis=0, return_inverse=True)
            self._unique_shift_numerators = {
                tuple(shift): np.where(inverse == idx)[0]
                for idx, shift in enumerate(uniq)
            }

        else:
            self._shift_numerators = np.empty((0, 2))
            self._shift_denominators = np.empty((0, 2))
            self._unique_shift_numerators = {}

    @property
    def shift_numerators(self):
        return self._shift_numerators

    @property
    def shift_denominators(self):
        return self._shift_denominators

    @property
    def unique_shift_numerators(self):
        'Get a dict of unique shifts numerators and their indices.'
        return self._unique_shift_numerators

    def get_shift_idx(self, shift):
        'Get the indices of coordinates that have a given shift.'

        shift = np.array(shift).squeeze().tolist()
        err_msg = ('Shift {} does not exist in this gamma surface.')

        # Get the numerators of the shift (given the grid denominator)
        shift_nums = []
        for i_idx, i in enumerate(shift):
            i_frac = fractions.Fraction(i).limit_denominator()
            if self.shift_denominators[i_idx] % i_frac.denominator != 0:
                raise ValueError(err_msg.format(shift))
            factor = int(self.shift_denominators[i_idx] / i_frac.denominator)
            numerator = i_frac.numerator * factor
            shift_nums.append(numerator)

        # Get the indices of coordinates:
        shift_tup = tuple(shift_nums)
        try:
            idx = self.unique_shift_numerators[shift_tup]
            return idx
        except IndexError:
            raise ValueError(err_msg)

    def get_expansion_idx(self, expansion):
        'Get the indices of coordinates that have a given expansion.'

        idx = np.where(np.isclose(self.expansions, expansion))[0]
        if not idx.size:
            msg = 'No coordinates with expansion close to {}.'.format(expansion)
            raise ValueError(msg)

        return idx

    def _validate(self, shifts, expansions, data):

        if shifts is None:
            shifts = np.empty((0, 2))
        if expansions is None:
            expansions = np.empty((0, ))
        if data is None:
            data = {}

        shifts = np.array(shifts)
        expansions = np.array(expansions)

        if shifts.ndim != 2 or shifts.shape[1] != 2:
            msg = '`shifts` must have shape (N, 2), but has shape: {}'
            raise ValueError(msg.format(shifts.shape))

        if expansions.ndim != 1:
            msg = '`expansions` must have shape (N,), but has shape: {}'
            raise ValueError(msg.format(expansions.shape))

        if shifts.shape[0] != expansions.shape[0]:
            msg = ('`shifts` and `expansions` must have the same outer shape, but have '
                   'shapes of {} and {}, respectively.')
            raise ValueError(msg.format(shifts.shape, expansions.shape))

        for data_name, data_item in data.items():
            data[data_name] = np.array(data_item)
            if data[data_name].shape[0] != shifts.shape[0]:
                msg = ('Data named "{}" must have outer dimension of length {}, but has '
                       'shape {}.')
                raise ValueError(msg.format(data_name, shifts.shape[0]))

        return shifts, expansions, data

    def _validate_structure(self, structure):
        if not isinstance(structure, Bicrystal):
            raise ValueError('base_structure` must be a Bicrystal object.')
        if structure.relative_shift != [0, 0]:
            raise ValueError('`base_structure` must have no pre-existing relative shift.')
        if structure.boundary_vac != 0:
            msg = '`base_structure` must have no pre-existing boundary vacuum.'
            raise ValueError(msg)
        return structure

    def __len__(self):
        return self.shifts.shape[0]

    @property
    @requires_base_structure
    def absolute_shifts(self):
        if self._absolute_shifts is None:
            self._absolute_shifts = self.shifts * self.base_structure.boundary_vecs_magnitude
        return self._absolute_shifts

    def get_coordinates(self, shift=None, expansion=None):
        'Get coordinates with a given shift and/or expansion.'

        if shift is None and expansion is None:
            msg = 'Specify at least one of `shift` and `expansion`.'
            raise ValueError(msg)

        if shift is not None:
            idx = self.get_shift_idx(shift)
            if expansion is not None:
                idx = np.intersect1d(idx, self.get_expansion_idx(expansion))
        elif expansion is not None:
            idx = self.get_expansion_idx(expansion)

        coords = [self.get_coordinate_by_index(i) for i in idx]

        return coords

    def get_coordinate_by_index(self, index):
        'Get a coordinate by index.'
        return GammaSurfaceCoordinate(self, index)

    def all_coordinates(self):
        'Generate all coordinates.'

        for index in range(len(self)):
            yield GammaSurfaceCoordinate(self, index)

    def add_fit(self, data_name, fit_size, shift=None):
        """

        Parameters
        ----------
        data_name : str
            Key in `data` to fit
        fit_size : int
            Number of expansion data points at each shift to include in the fit
        shift : list or length 2
            Do the fit for just a single shift.

        """

        if data_name not in self.data:
            msg = 'Data name "{}" does not exists. Existing data names are: {}.'
            raise ValueError(msg.format(data_name, list(self.data.keys())))

        fit = {
            'first_index': [],
            'minimum': [],
            'fit_in_range': [],
            'fit_coefficients': [],
        }

        # Fit at each unique relative shift:
        uniq, inverse = np.unique(self.shift_numerators, axis=0, return_inverse=True)
        for uniq_idx, _ in enumerate(uniq):

            idx = np.where(inverse == uniq_idx)[0]
            if len(idx) < fit_size:
                continue

            fitted_data = self._fit_expansions(self.expansions[idx],
                                               self.data[data_name][idx])

            fit['first_index'].append(idx[0])
            fit['minimum'].append(fitted_data['minimum'])
            fit['fit_coefficients'].append(fitted_data['fit_coefficients'])
            fit['fit_in_range'].append(fitted_data['fit_in_range'])

        fit = {k: np.array(v) for k, v in fit.items()}

        self.fitted_data.update({
            data_name: fit
        })

    def _fit_expansions(self, expansions, data):
        'Do quadratic fit on data with expansions.'

        poly_coeff = np.polyfit(expansions, data, 2)
        p1d = np.poly1d(poly_coeff)
        grad = np.polyder(p1d)
        min_exp = -grad[0] / grad[1]
        min_dat = p1d(min_exp)

        out = {
            'minimum': [min_exp, min_dat],
            'fit_coefficients': poly_coeff,
            'fit_in_range': (min(expansions) < min_exp) and (min_exp < max(expansions)),
        }

        return out

    @requires_base_structure
    def get_surface_grids(self, fractional=False, grid=True):

        x, y = np.meshgrid(*[np.arange(i + 1) / i for i in self.shift_denominators])

        if not fractional:
            x *= self.base_structure.boundary_vecs_magnitude[0]
            y *= self.base_structure.boundary_vecs_magnitude[1]

        z = np.zeros_like(x) * np.nan

        if not grid:
            x = x[0]
            y = y[:, 0]

        return x, y, z

    def get_fit_plot_data(self, data_name, shift, exp_range=None):
        """Get data for plotting fits for a given shifts.

        Returns
        -------
        list of dict
            TODO
            A list of three dicts are returned. They contain the x and y coordinates for
            the, respectively, the following plot traces:
                - The discrete data that has been fitted
                - A 
        """

        coords = self.get_coordinates(shift=shift)
        coords_srt = sorted(coords, key=lambda x: x.expansion)
        dat_x = [i.expansion for i in coords_srt]
        dat_y = [i.data[data_name] for i in coords_srt]
        fitting_dat = {
            'x': dat_x,
            'y': dat_y,
        }

        if not exp_range:
            dat_x_range = max(dat_x) - min(dat_x)
            exp_range = [
                min(dat_x) - 0.1 * dat_x_range,
                max(dat_x) + 0.1 * dat_x_range,
            ]

        fit_data = self.get_fit_data(data_name, shift)
        p1d = np.poly1d(fit_data['fit_coefficients'])
        fit_dat_x = np.linspace(*exp_range)
        fit_dat_y = p1d(fit_dat_x)
        fitted_curve = {
            'x': fit_dat_x,
            'y': fit_dat_y,
        }

        minimum_dat = {
            'x': [fit_data['minimum'][0]],
            'y': [fit_data['minimum'][1]],
        }

        out = {
            'data': fitting_dat,
            'fitted_data': fitted_curve,
            'minimum': minimum_dat,
        }

        return out

    def get_fit_data(self, data_name, shift):
        'Get fit data for a given shift.'

        fitted_data = self.fitted_data[data_name]
        first_idx = sorted(self.get_shift_idx(shift))[0]
        fit_idx = np.where(fitted_data['first_index'] == first_idx)[0]
        out = {
            'fit_coefficients': fitted_data['fit_coefficients'][fit_idx][0],
            'fit_in_range': fitted_data['fit_in_range'][fit_idx][0],
            'minimum': fitted_data['minimum'][fit_idx][0],
        }
        return out

    def get_surface_plot_data(self, data_name, expansion, fractional=False,
                              xy_as_grid=True):

        x, y, z = self.get_surface_grids(fractional, grid=xy_as_grid)

        for idx in self.get_expansion_idx(expansion):
            coord = self.get_coordinate_by_index(idx)
            z[tuple(coord.shift_numerator[::-1])] = self.data[data_name][idx]

        out = {
            'x': x,
            'y': y,
            'z': z,
        }

        return out

    def get_xy_plot_data(self, fractional=False):

        x, y, _ = self.get_surface_grids(fractional, grid=True)
        out = {
            'x': x.flatten(),
            'y': y.flatten(),
        }

        return out

    def get_fitted_surface_plot_data(self, data_name, expansion=False, fractional=False,
                                     xy_as_grid=True):

        x, y, z = self.get_surface_grids(fractional, grid=xy_as_grid)

        fitted_data = self.fitted_data[data_name]
        for idx, coord_idx in enumerate(fitted_data['first_index']):
            coord = self.get_coordinate_by_index(coord_idx)
            minimum = fitted_data['minimum'][idx]
            z_idx = tuple(coord.shift_numerator[::-1])
            z[z_idx] = minimum[0] if expansion else minimum[1]

        out = {
            'x': x,
            'y': y,
            'z': z,
        }

        return out

    def get_minimum_fitted_shift(self, data_name):

        min_idx = np.argmin(self.fitted_data[data_name]['minimum'][:, 1])
        first_idx = self.fitted_data[data_name]['first_index'][min_idx]
        shift = self.shifts[first_idx]
        coords = self.get_coordinates(shift=shift)
        return coords


class GammaSurfaceCoordinate(object):

    def __init__(self, gamma_surface, index):

        if index >= len(gamma_surface):
            msg = ('No gamma surface coordinate exists with index: {}. Number of '
                   'coordinates is: {}.')
            raise ValueError(msg.format(index, len(gamma_surface)))

        self.gamma_surface = gamma_surface
        self.index = index
        self._structure = None

    @property
    def shift(self):
        return self.gamma_surface.shifts[self.index]

    @property
    def shift_numerator(self):
        return self.gamma_surface.shift_numerators[self.index]

    @property
    def shift_denominator(self):
        return self.gamma_surface.shift_denominator[self.index]

    @property
    def expansion(self):
        return self.gamma_surface.expansions[self.index]

    @property
    def absolute_shift(self):
        self.gamma_surface.absolute_shifts[self.index]

    @property
    def data(self):
        return {k: v[self.index] for k, v in self.gamma_surface.data.items()}

    @property
    @requires_base_structure
    def structure(self):
        if not self._structure:
            structure = copy.deepcopy(self.gamma_surface.base_structure)
            structure.apply_relative_shift(self.gamma_surface.shifts[self.index], 0)
            structure.apply_boundary_vac(
                self.gamma_surface.expansions[self.index], 'sigmoid')
            self._structure = structure
        return self._structure

    @property
    def shift_fmt(self):
        all_nums = self.gamma_surface.shift_numerators
        max_nums = np.max(all_nums, axis=0)
        nums = [zeropad(i, j) for i, j in zip(all_nums[self.index], max_nums)]
        denoms = self.gamma_surface.shift_denominators
        out = '{0:}.{2:}_{1:}.{3:}'.format(*nums, *denoms)
        return out

    @property
    def expansion_fmt(self):
        return '{:+.3f}'.format(self.expansion)

    @property
    def coordinate_fmt(self):
        return '{}__{}'.format(self.shift_fmt, self.expansion_fmt)

    def __repr__(self):
        out = '{}(shift={!r}, expansion={!r}'.format(
            self.__class__.__name__,
            self.shift,
            self.expansion,
        )
        for k, v in self.data.items():
            out += ', {}={!r}'.format(k, v)
        out += ')'

        return out
