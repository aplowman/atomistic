"""`atomistic.bicrystal.py`"""

import warnings
from functools import partial

import spglib
import numpy as np

from atomistic import mathsutils
from atomistic.atomistic import AtomisticStructure
from atomistic.crystal import CrystalBox


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

    @property
    def bicrystal_thickness(self):
        """Get bicrystal thickness in grain boundary normal direction."""
        sup_nb = self.supercell[:, self.non_boundary_idx][:, None]
        return np.einsum('ij,ij', sup_nb, self.n_unit)

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

    def distance_from_gb(self, points):
        """
        Computes the distance from each in an array of column vector to the
        origin grain boundary plane.

        # TODO: does this assume supercell origin is at (0,0,0)?

        """
        return np.einsum('jk,jl->k', points, self.n_unit)

    def reorient_to_xy(self):
        """
        Reorient the supercell to a LAMMPS-compatible orientation in such a
        way that the boundary plane is in the xy plane.

        """

        # Reorient so the boundary plane is in xy
        if self.non_boundary_idx != 2:

            # Ensure non-boundary supercell vector is last vector, whilst
            # maintaining handedness of the supercell coordinate system.
            self.supercell = np.roll(self.supercell,
                                     (2 - self.non_boundary_idx), axis=1)

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

        if 'bulk_bicrystal' in sup_type:
            msg = 'Cannot apply boundary vacuum to a bulk_bicrystal.'
            raise NotImplementedError(msg)

        elif all([i not in sup_type for i in ['bicrystal', 'surface_bicrystal']]):
            msg = 'Cannot apply boundary vacuum to this supercell type.'
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
        Apply in-boundary-plane shifts to the specified grain to enable an
        exploration of the grain boundary's microscopic degrees of freedom.

        Parameters
        ----------
        shift : ndarray of size two
            A two-element array whose elements are the relative shift in of one
            crystal relative to the other in fractional coordinates s of the
            boundary area.
        crystal_idx : int
            Which crystal to shift. Zero-indexed.
        wrap : bool, optional
            If True, sites (atoms, lattice sites, etc) within the supercell are
            wrapped to the supercell volume after shifting. By default, False. 

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
        """
        Wrap atoms to within the boundary plane as defined by the supercell.

        """
        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot wrap atoms within a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal',
                                              'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot wrap atoms within this supercell type.')

        super().wrap_sites_to_supercell(sites=sites, dirs=self.boundary_idx)

    def check_inv_symmetry(self):
        """
        Check atoms exhibit inversion symmetry through the two crystal centres,
        if `self.maintain_inv_sym` is True.

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
