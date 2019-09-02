"""`atomistic.api.bicrystal.py`"""

import copy

import numpy as np
from vecmaths.vectors import vecpair_angle
from vecmaths.rotation import axang2rotmat

from atomistic.atomistic import AtomisticStructure
from atomistic.bicrystal import Bicrystal
from atomistic.crystal import CrystalBox, CrystalStructure


CSL_FROM_PARAMS_GB_TYPES = {
    'tilt_A': np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]),
    'tilt_B': np.array([
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]),
    'twist': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]),
    'mixed_A': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1]
    ])
}


def bicrystal_from_csl_vectors(crystal_structure, csl_vecs, box_csl=None,
                               gb_type=None, gb_size=None, edge_conditions=None,
                               overlap_tol=None, reorient=True, wrap=True,
                               maintain_inv_sym=False, boundary_vac=None,
                               relative_shift=None):
    """
    Parameters
    ----------
    crystal_structure : dict or CrystalStructure
    csl_vecs : list of length 2 of ndarray of shape (3, 3)
        List of two arrays of three column vectors representing CSL vectors
        in the lattice basis. The two CSL unit cells defined here rotate onto
        each other by the CSL rotation angle. The rotation axis is taken as the
        third vector, which must therefore be the same for both CSL unit cells.
    box_csl : ndarray of shape (3, 3), optional
        The linear combination of CSL unit vectors defined in `csl_vecs` used
        to construct each half of the bicrystal. The first two columns
        represent vectors defining the boundary plane. The third column
        represents a vector in the out-of-boundary direction. Only one of
        `box_csl` and `gb_type` may be specified.
    gb_type : str, optional
        Default is None. Must be one of 'tilt_A', 'tilt_B', 'twist' or
        'mixed_A'. Only one of `box_csl` and `gb_type` may be specified.
    gb_size : ndarray of shape (3,) of int, optional
        If `gb_type` is specified, the unit grain vectors associated with that
        `gb_type` are scaled by these integers. Default is None, in which case
        it is set to np.array([1, 1, 1]).
    edge_conditions : list of list of str
        Edge conditions for each grain in the bicrystal. See `CrystalBox` for
        details.
    maintain_inv_sym : bool, optional
        If True, the supercell atoms will be checked for inversion symmetry
        through the centres of both crystals. This check will be repeated
        following methods which manipulate atom positions. In this way, the two
        grain boundaries in the bicrystal are ensured to be identical.
    reorient : bool, optional
        If True, after construction of the boundary, reorient_to_lammps() is
        invoked. Default is True.
    boundary_vac_args : dict, optional
        If not None, after construction of the boundary, apply_boundary_vac()
        is invoked with this dict as keyword arguments. Default is None.
    boundary_vac_flat_args : dict, optional
        If not None, after construction of the boundary,
        apply_boundary_vac_flat() is invoked with this dict as keyword
        arguments. Default is None.
    boundary_vac_linear_args : dict, optional
        If not None, after construction of the boundary,
        apply_boundary_vac_linear() is invoked with this dict as keyword
        arguments. Default is None.
    relative_shift_args : dict, optional
        If not None, after construction of the boundary, apply_relative_shift()
        is invoked with this dict as keyword arguments. Default is None.
    wrap : bool, optional
        If True, after construction of the boundary, wrap_atoms_to_supercell()
        is invoked. Default is True.

    Notes
    -----
    Algorithm proceeds as follows:
    1.  Apply given linear combinations of given CSL unit vectors to form grain
        vectors of the bicrystal.
    2.  Multiply the out-of-boundary vector of the second grain by -1, such
        that rotation of the second grain by the CSL rotation angle will form a
        bicrystal of two grains.
    3.  Check grain A is formed from a right-hand basis - since we want the
        supercell vectors to be formed from a right-hand basis. If not, for
        both grain A and B, swap the first and second vectors to do this.
    4.  Fill the two grains with atoms
    5.  Rotate B onto A??

    TODO:
    -   Sort out lattice sites in apply_boundary_vac() & apply_relative_shift()
    -   Rename wrap_atoms_to_supercell to wrap_to_supercell and apply wrapping
        to lattice sites, and crystal boxes as well.

    """

    # Generate CrystalStructure if necessary:
    cs = CrystalStructure.init_crystal_structures([crystal_structure])[0]
    box_csl = np.array(box_csl)
    csl_vecs = np.array(csl_vecs)

    # print('csl_vecs: {}'.format(csl_vecs))

    if np.all(csl_vecs[0][:, 2] != csl_vecs[1][:, 2]):
        raise ValueError('Third vectors in `csl_vecs[0]` and csl_vecs[1] '
                         'represent the CSL rotation axis and must '
                         'therefore be equal.')

    if box_csl is not None and gb_type is not None:
        raise ValueError('Only one of `box_csl` and `gb_type` may be '
                         'specified.')

    if box_csl is None and gb_type is None:
        raise ValueError('Exactly one of `box_csl` and `gb_type` must be '
                         'specified.')

    if gb_type is not None:

        if gb_size is None:
            gb_size = np.array([1, 1, 1])

        if gb_type not in CSL_FROM_PARAMS_GB_TYPES:
            raise ValueError(
                'Invalid `gb_type`: {}. Must be one of {}'.format(
                    gb_type, list(CSL_FROM_PARAMS_GB_TYPES.keys())))

        box_csl = CSL_FROM_PARAMS_GB_TYPES.get(gb_type) * gb_size

    lat_vecs = cs.lattice.unit_cell
    rot_ax_std = np.dot(lat_vecs, csl_vecs[0][:, 2:3])
    csl_vecs_std = [np.dot(lat_vecs, c) for c in csl_vecs]

    # Non-boundary (column) index of `box_csl` and grain arrays:
    non_boundary_idx = 2

    # Enforce a rule that out of boundary grain vector has to be
    # (a multiple of) a single CSL unit vector. This reduces the
    # potential "skewness" of the supercell.
    if np.count_nonzero(box_csl[:, non_boundary_idx]) > 1:
        raise ValueError('The out of boundary vector, `box_csl[:, {}]`'
                         ' must have exactly one non-zero '
                         'element.'.format(non_boundary_idx))

    # Scale grains in lattice basis
    grn_a_lat = np.dot(csl_vecs[0], box_csl)
    grn_b_lat = np.dot(csl_vecs[1], box_csl)
    grn_b_lat[:, non_boundary_idx] *= -1

    # Get grain vectors in standard Cartesian basis
    grn_a_std = np.dot(lat_vecs, grn_a_lat)
    grn_b_std = np.dot(lat_vecs, grn_b_lat)

    # Get rotation matrix for rotating grain B onto grain A
    if np.all(csl_vecs[0] == csl_vecs[1]):
        rot_angles = [0, 0, 0]
        rot_mat = np.eye(3)

    else:
        rot_angles = vecpair_angle(*csl_vecs_std, axis=0)
        # print('rot_angles: {}'.format(np.rad2deg(rot_angles)))
        # print('rot_ax_std: {}'.format(rot_ax_std))
        if not np.isclose(*rot_angles[0:2]):
            raise ValueError('Non-equivalent rotation angles found between CSL'
                             ' vectors.')

        rot_mat = axang2rotmat(rot_ax_std[:, 0], rot_angles[0])

    grn_vols = [np.dot(np.cross(g[:, 0], g[:, 1]), g[:, 2])
                for g in (grn_a_std, grn_b_std)]

    # print('grn_vols: {}'.format(grn_vols))

    # Check grain volumes are the same:
    if not np.isclose(*np.abs(grn_vols)):
        raise ValueError('Grain A and B have different volumes.')

    # Check if grain A forms a right-handed coordinate system:
    if grn_vols[0] < 0:
        # Swap boundary vectors to make a right-handed coordinate system:
        grn_a_lat[:, [0, 1]] = grn_a_lat[:, [1, 0]]
        grn_b_lat[:, [0, 1]] = grn_b_lat[:, [1, 0]]
        grn_a_std[:, [0, 1]] = grn_a_std[:, [1, 0]]
        grn_b_std[:, [0, 1]] = grn_b_std[:, [1, 0]]
        box_csl[0, 1] = box_csl[1, 0]

    # Specify bounding box edge conditions for including atoms:
    if edge_conditions is None:
        edge_conditions = [
            ['10', '10', '10'],
            ['10', '10', '10']
        ]
        edge_conditions[1][non_boundary_idx] = '01'

    # print('grn_a_std: \n{}'.format(grn_a_std))
    # print('grn_b_std: \n{}'.format(grn_b_std))

    # Make two crystal boxes:
    crys_a = CrystalBox(cs, grn_a_std, edge_conditions=edge_conditions[0])
    crys_b = CrystalBox(cs, grn_b_std, edge_conditions=edge_conditions[1])

    for sites in list(crys_a.sites.values()) + list(crys_b.sites.values()):
        sites.basis = None

    # Rotate crystal B onto A:
    crys_b.rotate(rot_mat)

    # Shift crystals to form a supercell at the origin
    zero_shift = -crys_b.box_vecs[:, non_boundary_idx][:, None]
    crys_a.translate(zero_shift)
    crys_b.translate(zero_shift)

    # Define the supercell:
    sup_std = np.copy(crys_a.box_vecs)
    sup_std[:, non_boundary_idx] = (crys_a.box_vecs[:, non_boundary_idx] -
                                    crys_b.box_vecs[:, non_boundary_idx])

    # AtomisticStructure parameters
    as_params = {
        'supercell': sup_std,
        'crystals': [crys_a, crys_b],
    }

    # Bicrystal parameters
    bc_params = {
        'as_params': as_params,
        'maintain_inv_sym': maintain_inv_sym,
        'reorient': reorient,
        'boundary_vac': boundary_vac,
        'relative_shift': relative_shift,
        'wrap': wrap,
        'non_boundary_idx': 2,
        'rot_mat': rot_mat,
        'overlap_tol': overlap_tol,
    }

    return Bicrystal(**bc_params)


def bulk_bicrystal_from_csl_vectors(crystal_structure, csl_vecs,
                                    box_csl=None, gb_type=None,
                                    gb_size=None, edge_conditions=None,
                                    overlap_tol=1, reorient=True):
    """
    Parameters
    ----------
    csl_vecs: ndarray of int of shape (3, 3)

    """

    bc_params = {
        'crystal_structure': crystal_structure,
        'csl_vecs': [csl_vecs, csl_vecs],
        'box_csl': box_csl,
        'gb_type': gb_type,
        'gb_size': gb_size,
        'edge_conditions': edge_conditions,
        'overlap_tol': overlap_tol,
        'reorient': reorient,
        'wrap': False,
    }

    bicrys = bicrystal_from_csl_vectors(**bc_params)
    bicrys.meta['supercell_type'] = ['bulk', 'bulk_bicrystal']
    return bicrys


def surface_bicrystal_from_csl_vectors(crystal_structure, csl_vecs,
                                       box_csl=None, gb_type=None,
                                       gb_size=None, edge_conditions=None,
                                       overlap_tol=1,
                                       reorient=True, wrap=True,
                                       maintain_inv_sym=False,
                                       boundary_vac=None,
                                       relative_shift=None,
                                       surface_idx=0):
    """
    Parameters
    ----------
    csl_vecs: list of length 2 of ndarray of int of shape (3, 3)

    """

    bc_params = {
        'crystal_structure': crystal_structure,
        'csl_vecs': csl_vecs,
        'box_csl': box_csl,
        'gb_type': gb_type,
        'gb_size': gb_size,
        'edge_conditions': edge_conditions,
        'reorient': reorient,
        'wrap': wrap,
        'maintain_inv_sym': maintain_inv_sym,
        'boundary_vac': boundary_vac,
        'relative_shift': relative_shift,
    }

    bicrys = bicrystal_from_csl_vectors(**bc_params)

    # Remove atoms from removed crystal
    for sites in bicrys.sites.values():
        sites.remove(crystal_idx=(1 - surface_idx))

    # Now do overlap check:
    bicrys.check_overlapping_atoms(overlap_tol)

    bicrys.meta['supercell_type'] = ['surface', 'surface_bicrystal']
    return bicrys
