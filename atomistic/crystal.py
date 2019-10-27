"""`atomistic.crystal.py`"""

from pprint import pprint
import copy

import numpy as np
from beautifultable import BeautifulTable
from vecmaths.geometry import get_box_corners
from vecmaths.utils import snap_arr
from bravais import BravaisLattice
from spatial_sites import Sites
from spatial_sites.utils import repr_dict
from gemo import GeometryGroup, Box

from atomistic.utils import get_column_vector, check_indices

REPR_INDENT = 4


def get_bounding_box(box, bound_vecs=None, padding=0):
    """
    Find bounding boxes around parallelepipeds.

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
    mins = snap_arr(np.min(corners_bound, axis=2)[:, :, np.newaxis], 0, tol)
    maxs = snap_arr(np.max(corners_bound, axis=2)[:, :, np.newaxis], 0, tol)

    mins_floor = np.floor(mins) - padding
    maxs_ceil = np.ceil(maxs)

    bound_box_origin = np.concatenate(bound_vecs @ mins_floor, axis=1)
    bound_box_bv = np.concatenate((maxs_ceil - mins_floor + padding).astype(int), axis=1)
    bound_box = snap_arr(bound_box_bv.T[:, np.newaxis] * bound_vecs[np.newaxis], 0, tol)
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

        lat_sites = self.lattice.lattice_sites
        sites_dict = {'lattice_sites': lat_sites}
        for site_name, site_obj in self.motif.sites.items():

            rep_lab = repeat_labels.get(site_name, {})
            tiled_sites = lat_sites.tile(site_obj, rep_lab)

            setattr(self, site_name, tiled_sites)
            sites_dict.update({
                site_name: tiled_sites
            })

        return sites_dict

    @classmethod
    def init_crystal_structures(cls, crystal_structures):
        """Instantiate crystal structures if parametrisations are passed instead of
        CrystalStructure objects themselves.

        Parameters
        ----------
        crystal_structures : list of (dict or CrystalStructure)
            If a dict, must have keys:
                lattice : dict or BravaisLattice
                motif : dict

        Returns
        -------
        cs_objects : list of CrystalStructure

        """

        cs_objects = []
        for i in crystal_structures:
            if not isinstance(i, CrystalStructure):
                i = cls(**i)
            cs_objects.append(i)

        return cs_objects

    @property
    def lattice(self):
        return self._lattice

    @property
    def motif(self):
        return self._motif

    @property
    def sites(self):
        return self._sites

    def show(self, **kwargs):
        gg = GeometryGroup(
            points={'atoms': self.atoms.basis @ self.atoms},
            boxes={'unit cell': Box(edge_vectors=self.lattice.unit_cell)},
        )
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
                    'styles': {
                        'outline_colour': {
                            0: 'green',
                            1: 'purple',
                        },
                    },
                },
            ],
        }
        return gg.show(group_points=group_points)

    def __repr__(self):

        arg_fmt = ' ' * REPR_INDENT

        lat = '{!r}'.format(self.lattice).replace('\n', '\n' + arg_fmt)
        motif = '{!r}'.format(self.motif).replace('\n', '\n' + arg_fmt)
        sites = repr_dict(self.sites, REPR_INDENT)
        out = (
            '{0}(\n'
            '{1}lattice={2},\n'
            '{1}motif={3},\n'
            '{1}sites={4},\n'
            ')'.format(self.__class__.__name__, arg_fmt, lat, motif, sites)
        )
        return out


class Crystal(object):
    """Class to represent a bounded volume filled with a given
    CrystalStructure.

    """

    origin = None
    lattice_sites = None
    interstice_sites = None
    box_vecs = None
    atom_labels = None
    _sites = None
    atomistic_structure = None

    @property
    def crystal_idx(self):
        if self.atomistic_structure:
            for idx, i in enumerate(self.atomistic_structure.crystals):
                if i is self:
                    return idx
        else:
            return None

    @property
    def sites(self):
        if self.atomistic_structure:
            sites = {k: v.filter(crystal_idx=self.crystal_idx)
                     for k, v in self.atomistic_structure.sites.items()}
        else:
            sites = self._sites
        return sites

    def translate(self, shift):
        """
        Translate the crystal.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """

        shift = get_column_vector(shift)
        self.origin += shift

        for i in self.sites.values():
            i.translate(shift)

    def rotate(self, rot_mat, centre=None):
        """
        Rotate the crystal about a point according to a rotation matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to
            rotate them about a particular axis and angle.

        """

        if centre is None:
            centre = self.origin

        self.origin = rot_mat @ (self.origin - centre) + centre

        for i in self.sites.values():
            i.rotate(rot_mat, centre=centre)


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

        if self.bounding_box is not None:
            self.bounding_box['bound_box_origin'] += shift

    def rotate(self, rot_mat, centre=None):
        """
        Rotate the crystal box about a point according to a rotation matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to
            rotate them about a particular axis and angle.

        """
        if centre is None:
            centre = self.origin

        super().rotate(rot_mat, centre)

        self.box_vecs = rot_mat @ self.box_vecs

        # self.bounding_box['bound_box'][0] = np.dot(
        #     rot_mat, self.bounding_box['bound_box'][0]) # TODO

    def __init__(self, crystal_structure=None, box_vecs=None, edge_conditions=None,
                 origin=None, sites=None):
        """
        Fill a parallelepiped with atoms belonging to a given crystal structure.

        Parameters
        ----------
        crystal_structure : CrystalStructure
        box_vecs : ndarray of shape (3, 3)
            Array of column vectors representing the edge vectors of the parallelepiped
            to fill with crystal.
        edge_conditions : list of str, optional
            Determines if atom and lattice sites on the edges of the `box_vecs`
            parallelepiped should be included. It is a list of three two-character
            strings, each being a `1` or `0`. These refer to whether atoms are included
            (`1`) or not (`0`) for the near and far boundary along the dimension given by
            the position in the list. The near boundary is the boundary of the crystal box
            which intercepts the crystal box origin. Default is None, in which case it
            will be set to ['10', '10', '10']. For a given component, say x, the strings
            are decoded in the following way:
                '00': 0 <  x <  1
                '01': 0 <  x <= 1
                '10': 0 <= x <  1
                '11': 0 <= x <= 1
        origin : list or ndarray of size 3, optional
            Origin of the crystal box. By default, set to [0, 0, 0].

        Notes
        -----
        Algorithm proceeds as follows:
        1.  Form a bounding parallelepiped around the parallelepiped defined by
            `box_vecs`, whose edge vectors are parallel to the lattice vectors.
        2.  Find all sites within and on the edges/corners of that bounding box.
        3.  Transform sites to the box basis.
        4.  Find valid sites, which have vector components in the interval [0, 1] in the
            box basis, where the interval may be (half-)closed/open depending on the
            specified edge conditions.

        """

        msg = ('Specify exactly one of: `crystal_structure` and `sites`.')
        if crystal_structure is None and sites is None:
            raise ValueError(msg)
        if crystal_structure is not None and sites is not None:
            raise ValueError(msg)

        if crystal_structure:

            if edge_conditions is None:
                edge_conditions = ['10', '10', '10']

            # Generate CrystalStructure if necessary:
            cs = CrystalStructure.init_crystal_structures([crystal_structure])[0]

            # Get the bounding box of box_vecs whose vectors are parallel to the crystal
            # lattice. Use padding to catch edge atoms which aren't on lattice sites:
            bounding_box = get_bounding_box(
                box_vecs,
                bound_vecs=cs.lattice.unit_cell,
                padding=1
            )
            bb_bv = bounding_box['bound_box_bv'][:, 0]
            bb_org_bv = bounding_box['bound_box_origin_bv'][:, 0]

            # Get all lattice sites within the bounding box, as column vectors:
            grid = [range(bb_org_bv[i], bb_org_bv[i] + bb_bv[i] + 1) for i in [0, 1, 2]]
            unit_cell_origins = np.vstack(np.meshgrid(*grid)).reshape((3, -1))

            origin_sites = Sites(coords=unit_cell_origins,
                                 vector_direction='col',
                                 basis=cs.atoms.basis)

            sites = {k: origin_sites.tile(v) for k, v in cs.sites.items()}
            for i in sites.values():

                i.basis = box_vecs
                i.snap_coords([0, 1], tol=1e-14)

                for j in range(3):
                    edge_con = edge_conditions[j]
                    remove_arr = np.zeros(len(i), dtype=bool)
                    if edge_con[0] == '0':
                        remove_arr = np.logical_or(remove_arr, i._coords[j] <= 0)
                    elif edge_con[0] == '1':
                        remove_arr = np.logical_or(remove_arr, i._coords[j] < 0)
                    if edge_con[1] == '0':
                        remove_arr = np.logical_or(remove_arr, i._coords[j] >= 1)
                    elif edge_con[1] == '1':
                        remove_arr = np.logical_or(remove_arr, i._coords[j] > 1)
                    i.remove(remove_arr)

            origin = np.zeros((3, 1))

        else:
            bounding_box = None
            cs = None

        self._sites = self._init_sites(sites)
        self.box_vecs = box_vecs
        self.bounding_box = bounding_box
        self.crystal_structure = cs
        self.origin = get_column_vector(origin)
        self.edge_conditions = edge_conditions

        if crystal_structure:
            # Only translate sites if they were generated in the constructor.
            if origin is not None:
                self.translate(origin)

    def _init_sites(self, sites):

        req_sites = ['atoms']
        allowed_sites_labels = {
            'atoms': ['species', 'species_order', 'bulk_coord_num', 'crystal_idx'],
            'lattice_sites': ['crystal_idx'],
            'interstices': ['bulk_name', 'is_occupied', 'crystal_idx'],
        }

        out = {}
        for sites_name, sites_data in sites.items():

            if sites_name in out:
                msg = ('Multiple sites with the same name were found.')
                raise ValueError(msg)

            sites_obj = sites_data
            if not isinstance(sites_obj, Sites):
                sites_obj = Sites(coords=sites_data['coords'],
                                  labels=sites_data['labels'])

            if not sites_name in allowed_sites_labels:
                msg = ('Site name "{}" is not a valid CrystalBox site name. '
                       'Valid names are: {}')
                raise ValueError(
                    msg.format(sites_name, list(allowed_sites_labels.keys())))

            for lab_name in sites_obj.labels:
                if not lab_name in allowed_sites_labels[sites_name]:
                    msg = ('Label name "{}" is not a valid label name for '
                           'CrystalBox sites called "{}".')
                    raise ValueError(msg.format(lab_name, sites_name))

            setattr(self, sites_name, sites_obj)
            out.update({sites_name: sites_obj})

        for i in req_sites:
            if i not in out:
                msg = 'Sites with name "{}" must be specified for the CrystalBox'
                raise ValueError(msg.format(i))

        return out

    @property
    def centroid(self):
        return get_box_corners(self.box_vecs, origin=self.origin).mean(2).T

    def show(self, **kwargs):
        gg = GeometryGroup(
            points={
                'atoms': self.atoms.basis @ self.atoms,
                'lattice_sites': self.lattice_sites.basis @ self.lattice_sites,
            },
            boxes={
                'unit cell': Box(edge_vectors=self.crystal_structure.lattice.unit_cell),
                'box_vecs': Box(edge_vectors=self.box_vecs, origin=self.origin),
            }
        )
        return gg.show(
            group_points={
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
                ]
            }
        )


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

    def __repr__(self):

        arg_fmt = ' ' * REPR_INDENT
        sites = repr_dict(self.sites, REPR_INDENT)

        out = (
            '{0}(\n'
            '{1}sites={2},\n'
            ')'.format(
                self.__class__.__name__,
                arg_fmt,
                sites
            )
        )
        return out
