"""`atomistic.crystal.py`"""

import copy

import numpy as np
from beautifultable import BeautifulTable
from vecmaths.geometry import get_box_corners
from vecmaths.utils import snap_arr
from bravais import BravaisLattice
from spatial_sites import Sites

from atomistic.utils import get_column_vector, check_indices
from atomistic.visualise import visualise_structure

REPR_INDENT = 4


def get_bounding_box(box, bound_vecs=None, padding=0):
    """
    Find bounding boxes around parallelopipeds.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3)
        Array defining N parallelograms, each specified by three 3D-column
        vectors.
    bound_vecs : ndarray of shape (3, 3), optional
        Array defining the vectors of which the computed bounding box edge
        vectors should be integer multiples. Default is identity matrix of
        shape (3, 3).
    padding : int
        Integer number of additional `bound_vecs` to include in the bounding
        box in each direction as padding around the box. Note that as currently
        implemented, this adds a total of (2 * padding) bound vecs magnitude to
        each of the bounding box edge vectors.

    Returns
    -------
    dict of (str : ndarray)
        `bound_box` is an ndarray of shape (N, 3, 3) defining bounding box edge
        vectors as three 3D-column vectors for each input parallelogram.

        `bound_box_origin` is an ndarray of shape (3, N) defining the origins
        of the bounding boxes as 3D-column vectors.

        `bound_box_bv` is an ndarray of shape (3, N) defining as 3D-column
        vectors the multiples of bound_vecs which form the bounding box.

        `bound_box_origin_bv` is an ndarray with shape (3, N) defining as
        3D-column vectors the origins of the bouding boxes in the `bound_vecs`
        basis.

    TODO:
    -   Allow compute non-integer bounding box (i.e. bounding box just
        determined by directions of `bound_vecs`, not magnitudes.)

    """

    if bound_vecs is None:
        bound_vecs = np.eye(3)

    # Transformation matrix to `bound_vecs` basis:
    bound_vecs_inv = np.linalg.inv(bound_vecs)

    corners = get_box_corners(box)
    corners_bound = bound_vecs_inv @ corners

    tol = 1e-12
    mins = snap_arr(
        np.min(corners_bound, axis=2)[:, :, np.newaxis], 0, tol)
    maxs = snap_arr(
        np.max(corners_bound, axis=2)[:, :, np.newaxis], 0, tol)

    mins_floor = np.floor(mins) - padding
    maxs_ceil = np.ceil(maxs)

    bound_box_origin = np.concatenate(bound_vecs @ mins_floor, axis=1)
    bound_box_bv = np.concatenate(
        (maxs_ceil - mins_floor + padding).astype(int), axis=1)
    bound_box = snap_arr(
        bound_box_bv.T[:, np.newaxis] * bound_vecs[np.newaxis], 0, tol)
    bound_box_origin_bv = np.concatenate(mins_floor.astype(int), axis=1)

    out = {
        'bound_box': bound_box,
        'bound_box_origin': bound_box_origin,
        'bound_box_bv': bound_box_bv,
        'bound_box_origin_bv': bound_box_origin_bv
    }

    return out


class CrystalStructure(object):
    """Class to represent a crystal structure with a lattice and atomic motif.

    Attributes
    ----------
    lattice: BravaisLattice
    motif: dict
    lat_sites_std : ndarray of shape (3, N)
    lat_sites_frac : ndarray of shape (3, N)
    atom_sites_std : ndarray of shape (3, M)
    atom_sites_frac : ndarray of shape (3, M)

    TODO: finish/correct docstring

    """

    def __init__(self, lattice, motif):
        """Instantiate a CrystalStructure object.

        Parameters
        ----------
        lattice : BravaisLattice or dict
            If a dict, a BravaisLattice object is created, using the following
            keys:
                lattice_system : str
                lattice_parameters : dict
                    dict containing lattice parameters: a, b, c, alpha/α,
                    beta/β, gamma/α.
                centring_type : str, optional

        motif : dict or AtomicMotif
            TODO: redo
            Dict representing the atomic motif of the crystal structure. The
            following keys must exist:
                atom_sites : ndarray of shape (3, N)
                    Array of column vectors representing positions of the atoms
                    associated with each lattice site. Given in fractional
                    coordinates of the lattice unit cell.
                species : ndarray or list of length P of str
                    Species names associated with each atom site.
                species_idx : ndarray or list of length N of int
                    Array which maps each atom site to a chemical symbol in
                    `species`.

        """

        self._lattice = self._init_lattice(lattice)
        self._motif = self._init_motif(motif)
        self._sites = self._init_sites()

    def __setattr__(self, name, value):
        'Overridden method to prevent reassigning sites attributes.'

        if getattr(self, '_sites', None) and name in self._sites:
            msg = 'Cannot set attribute "{}"'.format(name)
            raise AttributeError(msg)

        # Set all other attributes as normal:
        super().__setattr__(name, value)

    def _init_lattice(self, lattice):
        """Generate a BravaisLattice object if only a parametrisation is
        passed."""

        if not isinstance(lattice, BravaisLattice):
            kwargs = {
                k: v for k, v in lattice.items()
                if k not in 'lattice_parameters'
            }
            kwargs.update(lattice['lattice_parameters'])
            lattice = BravaisLattice(**kwargs)

        return lattice

    def _init_motif(self, motif):
        """Generate an AtomicMotif object if only a parametrisation is
        passed."""

        if not isinstance(motif, AtomicMotif):
            motif = AtomicMotif(**motif)

        return motif

    def _init_sites(self):
        'Tile Sites (atoms, interstices) in the atomic motif'

        repeat_labels = {
            'atoms': {
                'species': 'species_order',
            }
        }

        sites_dict = {}
        for site_name, site_obj in self.motif.sites.items():

            rep_lab = repeat_labels.get(site_name, {})
            tiled_sites = self.lattice_sites.tile(site_obj, rep_lab)

            setattr(self, site_name, tiled_sites)
            sites_dict.update({
                site_name: tiled_sites
            })

        return sites_dict

    @property
    def lattice(self):
        return self._lattice

    @property
    def motif(self):
        return self._motif

    @property
    def sites(self):
        return self._sites

    @property
    def lattice_sites(self):
        """Alias to the BravaisLattice lattice_sites Sites object."""
        return self.lattice.lattice_sites

    # @property
    # def atom_sites_frac(self):
    #     return np.dot(np.linalg.inv(self.lattice.unit_cell), self.atom_sites)

    # @property
    # def species(self):
    #     return self.atom_labels['species'][0]

    # @property
    # def species_idx(self):
    #     return self.atom_labels['species'][1]

    # @property
    # def all_species(self):
    #     return self.species[self.species_idx]

    def visualise(self, **kwargs):
        visualise_structure(self, **kwargs)

    def __repr__(self):

        return ('CrystalStructure(\n'
                '\t' + self.lattice.__repr__() + '\n'
                '\t' + '{!r}'.format(self.motif) + '\n'
                ')')

    def __str__(self):

        atoms_str = BeautifulTable()
        atoms_str.numeric_precision = 4
        atoms_str.intersection_char = ''
        column_headers = ['Number', 'x', 'y', 'z']

        for i in self.atom_labels.keys():
            column_headers.append(i)

        atoms_str.column_headers = column_headers
        atom_sites_frac = self.atom_sites_frac

        for atom_idx in range(atom_sites_frac.shape[1]):

            row = [
                atom_idx,
                *(atom_sites_frac[:, atom_idx]),
                *[v[0][v[1]][atom_idx] for _, v in self.atom_labels.items()]
            ]
            atoms_str.append_row(row)

        ret = ('{!s}-{!s} Bravais lattice + {!s}-atom motif\n\n'
               'Lattice parameters:\n'
               'a = {!s}\nb = {!s}\nc = {!s}\n'
               'α = {!s}°\nβ = {!s}°\nγ = {!s}°\n'
               '\nLattice vectors = \n{!s}\n'
               '\nLattice sites (fractional) = \n{!s}\n'
               '\nLattice sites (Cartesian) = \n{!s}\n'
               '\nAtoms (fractional coordinates of '
               'unit cell) = \n{!s}\n').format(
            self.lattice.lattice_system,
            self.lattice.centring_type,
            self.motif['atoms']['sites'].shape[1],
            self.lattice.a, self.lattice.b,
            self.lattice.c, self.lattice.α,
            self.lattice.β, self.lattice.γ,
            self.lattice.unit_cell,
            self.lattice.lattice_sites_frac,
            self.lattice.lattice_sites,
            atoms_str)

        return ret


class Crystal(object):
    """Class to represent a bounded volume filled with a given
    CrystalStructure.

    """

    origin = None
    lattice_sites = None
    interstice_sites = None
    box_vecs = None
    atom_labels = None

    def translate(self, shift):
        """
        Translate the crystal.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """

        shift = get_column_vector(shift)
        self.origin += shift
        self.atom_sites += shift

        if self.lattice_sites is not None:
            self.lattice_sites += shift

        if self.interstice_sites is not None:
            self.interstice_sites += shift

    def rotate(self, rot_mat):
        """
        Rotate the crystal about its origin according to a rotation matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to
            rotate them about a particular axis and angle.

        """

        origin = np.copy(self.origin)
        self.translate(-origin)

        self.atom_sites = np.dot(rot_mat, self.atom_sites)

        if self.lattice_sites is not None:
            self.lattice_sites = np.dot(rot_mat, self.lattice_sites)

        if self.interstice_sites is not None:
            self.interstice_sites = np.dot(rot_mat, self.interstice_sites)

        self.translate(origin)

    @property
    def atom_sites_frac(self):
        return np.dot(np.linalg.inv(self.box_vecs.vecs), self.atom_sites)

    @property
    def lattice_sites_frac(self):
        if self.lattice_sites is not None:
            return np.dot(np.linalg.inv(self.box_vecs.vecs), self.lattice_sites)
        else:
            return None

    @property
    def interstice_sites_frac(self):
        if self.interstice_sites is not None:
            return np.dot(np.linalg.inv(self.box_vecs.vecs),
                          self.interstice_sites)
        else:
            return None

    @property
    def species(self):
        return self.atom_labels['species'][0]

    @property
    def species_idx(self):
        return self.atom_labels['species'][1]

    @property
    def all_species(self):
        return self.species[self.species_idx]


class CrystalBox(Crystal):
    """
    Class to represent a parallelepiped filled with a CrystalStructure

    Attributes
    ----------
    crystal_structure : CrystalStructure
    box_vecs : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the
        parallelepiped to fill with crystal.
    bounding_box : dict of str : ndarray
        Dict of arrays defining the bounding box used to generate an array of
        candidate lattice and atom sites.
    lat_sites_std : ndarray of shape (3, N)
        Array of column vectors representing the lattice sites in Cartesian
        coordinates.
    lat_sites_frac : ndarray of shape (3, N)
        Array of column vectors representing the lattice sites in fractional
        coordinates of the crystal box.
    atom_sites_std : ndarray of shape (3, M)
        Array of column vectors representing the atoms in Cartesian
        coordinates.
    atom_sites_frac : ndarray of shape (3, M)
        Array of column vectors representing the atoms in fractional
        coordinates of the crystal box.
    species_idx : ndarray of shape (M)
        Identifies the species for each of the M atoms.
    species : list of str
        List of element symbols for species present in the crystal.

    TODO: correct docstring

    """

    def translate(self, shift):
        """
        Translate the crystal box.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """
        shift = get_column_vector(shift)
        super().translate(shift)

        self.bounding_box['bound_box_origin'] += shift

    def rotate(self, rot_mat):
        """
        Rotate the crystal box about its origin according to a rotation matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to
            rotate them about a particular axis and angle.

        """

        super().rotate(rot_mat)

        self.box_vecs = np.dot(rot_mat, self.box_vecs)
        self.bounding_box['bound_box'][0] = np.dot(
            rot_mat, self.bounding_box['bound_box'][0])

    def _find_valid_sites(self, sites_frac, labels, unit_cell_origins, lat_vecs,
                          box_vecs_inv, edge_conditions):

        tol = 1e-14

        sts_frac_rs = sites_frac.T.reshape((-1, 3, 1))
        sts_lat = np.concatenate(unit_cell_origins + sts_frac_rs, axis=1)
        sts_std = np.dot(lat_vecs, sts_lat)
        sts_box = np.dot(box_vecs_inv, sts_std)
        sts_box = snap_arr(sts_box, 0, tol)
        sts_box = snap_arr(sts_box, 1, tol)

        # Form a boolean edge condition array based on `edge_condtions`.
        # Start by allowing all sites:
        cnd = np.ones(sts_box.shape[1], dtype=bool)

        for dir_idx, pt in enumerate(edge_conditions):

            if pt[0] == '1':
                cnd = np.logical_and(cnd, sts_box[dir_idx] >= 0)
            elif pt[0] == '0':
                cnd = np.logical_and(cnd, sts_box[dir_idx] > 0)

            if pt[1] == '1':
                cnd = np.logical_and(cnd, sts_box[dir_idx] <= 1)
            elif pt[1] == '0':
                cnd = np.logical_and(cnd, sts_box[dir_idx] < 1)

        in_idx = np.where(cnd)[0]
        sts_box_in = sts_box[:, in_idx]
        sts_std_in = sts_std[:, in_idx]
        sts_std_in = snap_arr(sts_std_in, 0, tol)

        labels_in = {}
        for k, v in labels.items():
            labels_in.update({
                k: (v[0],
                    np.repeat(v[1], unit_cell_origins.shape[1])[in_idx])
            })

        return (sts_std_in, sts_box_in, labels_in)

    def __init__(self, crystal_structure=None, box_vecs=None, edge_conditions=None,
                 origin=None):
        """
        Fill a parallelopiped with atoms belonging to a given crystal
        structure.

        Parameters
        ----------
        crystal_structure : CrystalStructure
        box_vecs : ndarray of shape (3, 3)
            Array of column vectors representing the edge vectors of the
            parallelopiped to fill with crystal.
        edge_conditions : list of str, optional
            Determines if atom and lattice sites on the edges of the `box_vecs`
            parallelopiped should be included. It is a list of three
            two-character strings, each being a `1` or `0`. These refer to
            whether atoms are included (`1`) or not (`0`) for the near and far
            boundary along the dimension given by the position in the list. The
            near boundary is the boundary of the crystal box which intercepts
            the crystal box origin. Default is None, in which case it will be
            set to ['10', '10', '10']. For a given component, say x, the
            strings are decoded in the following way:
                '00': 0 <  x <  1
                '01': 0 <  x <= 1
                '10': 0 <= x <  1
                '11': 0 <= x <= 1
        origin : list or ndarray of size 3, optional
            Origin of the crystal box. By default, set to [0, 0, 0].

        Notes
        -----
        Algorithm proceeds as follows:
        1.  Form a bounding parallelopiped around the parallelopiped defined by
            `box_vecs`, whose edge vectors are parallel to the lattice vectors.
        2.  Find all sites within and on the edges/corners of that bounding
            box.
        3.  Transform sites to the box basis.
        4.  Find valid sites, which have vector components in the interval
            [0, 1] in the box basis, where the interval may be (half-)closed
            /open depending on the specified edge conditions.

        """

        if edge_conditions is None:
            edge_conditions = ['10', '10', '10']

        # Convenience:
        cs = crystal_structure
        lat_vecs = cs.lattice.unit_cell

        # Get the bounding box of box_vecs whose vectors are parallel to the
        # crystal lattice. Use padding to catch edge atoms which aren't on
        # lattice sites.
        bounding_box = get_bounding_box(
            box_vecs, bound_vecs=lat_vecs, padding=1)
        box_vecs_inv = np.linalg.inv(box_vecs)

        bb = bounding_box['bound_box'][0]
        bb_org = bounding_box['bound_box_origin'][:, 0]
        bb_bv = bounding_box['bound_box_bv'][:, 0]
        bb_org_bv = bounding_box['bound_box_origin_bv'][:, 0]

        # Get all lattice sites within the bounding box, as column vectors:
        grid = [range(bb_org_bv[i], bb_org_bv[i] + bb_bv[i] + 1)
                for i in [0, 1, 2]]
        unit_cell_origins = np.vstack(np.meshgrid(*grid)).reshape((3, -1))

        com_params = [
            unit_cell_origins,
            lat_vecs,
            box_vecs_inv,
            edge_conditions,
        ]

        (lat_std, lat_box,
            lat_labs) = self._find_valid_sites(cs.lattice_sites_frac,
                                               cs.lattice_labels, *com_params)

        (at_std, at_box,
            at_labs) = self._find_valid_sites(cs.atom_sites_frac,
                                              cs.atom_labels, *com_params)

        int_std, int_box, int_labs = None, None, None
        if cs.interstice_sites is not None:
            (int_std, int_box,
                int_labs) = self._find_valid_sites(cs.interstice_sites_frac,
                                                   cs.interstice_labels,
                                                   *com_params)

        self.lattice_sites = lat_std
        self.lattice_labels = lat_labs

        self.atom_sites = at_std
        self.atom_labels = at_labs

        self.interstice_sites = int_std
        self.interstice_labels = int_labs

        self.bounding_box = bounding_box
        self.crystal_structure = cs
        self.box_vecs = box_vecs
        self.origin = np.zeros((3, 1))

        if origin is not None:
            self.translate(origin)

    def visualise(self, **kwargs):
        visualise_structure(self, **kwargs)


class AtomicMotif(object):
    """Class to represent an atomic motif that is to be applied at each lattice
    site of a Bravais lattice to form a crystal structure."""

    def __init__(self, **sites):

        req_sites = ['atoms']

        self._sites = {}

        for sites_name, sites_data in sites.items():
            if sites_name in self._sites:
                msg = ('Multiple sites with the same name were found in the '
                       'AtomicMotif.')
                raise ValueError(msg)
            sites_obj = self._init_sites(sites_name, sites_data)
            setattr(self, sites_name, sites_obj)
            self._sites.update({
                sites_name: sites_obj
            })

        for i in req_sites:
            if i not in sites:
                msg = ('Sites with name "{}" must be specified for the '
                       'AtomicMotif')
                raise ValueError(msg.format(i))

    def __setattr__(self, name, value):
        """Overridden method to prevent reassigning sites attributes."""

        if getattr(self, '_sites', None) and name in self._sites:
            msg = 'Cannot set attribute "{}"'.format(name)
            raise AttributeError(msg)

        # Set all other attributes as normal:
        super().__setattr__(name, value)

    def _init_sites(self, sites_name, sites_data):
        """Instantiate Sites if parametrisations are passed instead of
        Sites objects themselves."""

        allowed_sites_labels = {
            'atoms': ['species', 'bulk_coord_num'],
            'interstices': ['bulk_name', 'is_occupied']
        }

        sites_obj = sites_data
        if not isinstance(sites_obj, Sites):
            sites_obj = Sites(coords=sites_data['coords'],
                              labels=sites_data['labels'],
                              cast_to_float=True)

        if not sites_name in allowed_sites_labels:
            msg = ('Site name "{}" is not a valid AtomicMotif site name. '
                   'Valid names are: {}')
            raise ValueError(
                msg.format(sites_name, list(allowed_sites_labels.keys()))
            )

        for lab_name in sites_obj.labels:
            if not lab_name in allowed_sites_labels[sites_name]:
                msg = ('Label name "{}" is not a valid label name for '
                       'AtomicMotif sites called "{}".')
                raise ValueError(msg.format(lab_name, sites_name))

        return sites_obj

    @property
    def sites(self):
        return self._sites

    @property
    def atom_sites(self):
        return self.sites['atom_sites']

    def __repr__(self):

        arg_fmt = ' ' * REPR_INDENT
        out = (
            '{0}(\n'
            '{1}sites={2!r},\n'
            ')'.format(
                self.__class__.__name__,
                arg_fmt,
                self.sites
            )
        )
        return out
