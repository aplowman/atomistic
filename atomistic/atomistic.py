"""`atomistic.atomistic.py`"""

import warnings
from itertools import combinations

import numpy as np
import mendeleev
import spglib
from vecmaths import rotation, geometry

from atomistic.visualise import visualise_structure
from atomistic.utils import get_column_vector


def get_box_centre(box, origin=None):
    """
    Find the centre of a parallelepiped.

    TODO: add to vecmaths

    Parameters
    ----------
    box : ndarray of shape (3, 3)
        Array of edge vectors defining a parallelepiped.
    origin : ndarray of shape (3, 1)
        Origin of the parallelepiped.

    Returns
    -------
    ndarray of shape (3, N)

    """

    return geometry.get_box_corners(box, origin=origin).mean(2).T


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
    """
    Class to represent crystals of atoms

    Attributes
    ----------
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the atom positions.
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing supercell edge vectors.
    lattice_sites : ndarray of shape (3, M), optional
        Array of column vectors representing lattice site positions.
    crystals : list of dict of (str : ndarray or int), optional
        Each dict contains at least these keys:
            `crystal` : ndarray of shape (3, 3)
                Array of column vectors representing the crystal edge vectors.
            `origin` : ndarray of shape (3, 1)
                Column vector specifying the origin of this crystal.
        Additional keys are:
            'cs_idx': int
                Index of `crystal_structures`, defining to which
                CrystalStructure this crystal belongs.
            `cs_orientation`: ndarray of shape (3, 3)
                Rotation matrix which rotates the CrystalStructure lattice
                unit cell from the initialised BravaisLattice object to some
                other desired orientation.
            'cs_origin': list of float or int
                Origin of the CrystalStructure unit cell in multiples of the
                CrystalStructure unit cell vectors. For integer values, this
                will not affect the atomic structure of the crystal. To
                produce a rigid translation of the atoms within the crystal,
                non-integer values can be used.

    crystal_structures : list of CrystalStructure, optional
    crystal_idx : ndarray of shape (N,), optional
        Defines to which crystal each atom belongs.
    lat_crystal_idx : ndarray of shape (M,), optional
        Defines to which crystal each lattice site belongs
    species_idx : ndarray of shape (N,), optional
        Defines to which species each atom belongs, indexed within the atom's
        crystal_structure. For atom index `i`, this indexes
        `crystal_structures[
            crystals[crystal_idx[i]]['cs_idx']]['species_set']`
        Either specify (`all_species` and `all_species_idx`) or (`species_idx`
        and `motif_idx`), but not both.
    motif_idx : ndarray of shape (N,), optional
        Defines to which motif atom each atom belongs, indexed within the
        atom's crystal_structure. For atom index `i`, this indexes
        `crystal_structures[
            crystals[crystal_idx[i]]['cs_idx']]['species_motif']`
        Either specify (`all_species` and `all_species_idx`) or (`species_idx`
        and `motif_idx`), but not both.
    all_species : ndarray of str, optional
        1D array of strings representing the distinct species. Either specify
        (`all_species` and `all_species_idx`) or (`species_idx` and
        `motif_idx`), but not both.
    all_species_idx : ndarray of shape (N, ), optional
        Defines to which species each atom belongs, indexed over the whole
        AtomisticStructure. This indexes `all_species`. Either specify
        (`all_species` and `all_species_idx`) or (`species_idx` and
        `motif_idx`), but not both.

    atom_sites_frac
    num_atoms_per_crystal
    num_atoms
    num_crystals
    reciprocal_supercell

    Methods
    -------
    todo

    TODO:
    -   Re-write docstrings.
    -   Consolidate atom/lattice/interstice into a list of Sites objects.

    """

    def __init__(self, supercell, atom_sites, atom_labels, origin=None,
                 lattice_sites=None, lattice_labels=None, interstice_sites=None,
                 interstice_labels=None, crystals=None, crystal_structures=None,
                 overlap_tol=1, tile=None):
        """Constructor method for AtomisticStructure object."""

        if origin is None:
            origin = np.zeros((3, 1))

        self.origin = origin

        self.atom_sites = atom_sites
        self.atom_labels = atom_labels
        self.supercell = supercell
        self.meta = {}

        self.lattice_sites = lattice_sites
        self.lattice_labels = lattice_labels
        self.interstice_sites = interstice_sites
        self.interstice_labels = interstice_labels
        self.crystals = crystals
        self.crystal_structures = crystal_structures
        self._overlap_tol = overlap_tol

        self.check_overlapping_atoms(overlap_tol)

        # Check handedness:
        if self.volume < 0:
            raise ValueError('Supercell does not form a right - handed '
                             'coordinate system.')

        if tile:
            self.tile_supercell(tile)

    def translate(self, shift):
        """
        Translate the AtomisticStructure.

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

        if self.crystals is not None:
            for c_idx in range(len(self.crystals)):
                self.crystals[c_idx]['origin'] += shift

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

        origin = np.copy(self.origin)
        self.translate(-origin)

        self.supercell = np.dot(rot_mat, self.supercell)
        self.atom_sites = np.dot(rot_mat, self.atom_sites)

        if self.lattice_sites is not None:
            self.lattice_sites = np.dot(rot_mat, self.lattice_sites)

        if self.interstice_sites is not None:
            self.interstice_sites = np.dot(rot_mat, self.interstice_sites)

        if self.crystals is not None:

            for c_idx in range(len(self.crystals)):

                c = self.crystals[c_idx]

                c['crystal'] = np.dot(rot_mat, c['crystal'])
                c['origin'] = np.dot(rot_mat, c['origin'])

                if 'cs_orientation' in c.keys():
                    c['cs_orientation'] = np.dot(rot_mat, c['cs_orientation'])

        self.translate(origin)

    def visualise(self, **kwargs):
        visualise_structure(self, **kwargs)

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
                    raise ValueError('`dirs` must be a list whose elements are'
                                     '0, 1 or 2.')

        allowed_sites_str = [
            'atom',
            'lattice',
            'interstice',
            'all',
        ]

        if not isinstance(sites, str) or sites not in allowed_sites_str:
            raise ValueError('`sites` must be a string and one of: "atom", '
                             '"lattice", "interstice" or "all".')

        if sites == 'all':
            sites_arr = [
                self.atom_sites,
                self.lattice_sites,
                self.interstice_sites
            ]
        elif sites == 'atom':
            sites_arr = [self.atom_sites]
        elif sites == 'lattice':
            sites_arr = [self.lattice_sites]
        elif sites == 'interstice':
            sites_arr = [self.interstice_sites]

        for s_idx in range(len(sites_arr)):

            s = sites_arr[s_idx]
            if s is None:
                continue

            # Get sites in supercell basis:
            s_sup = np.dot(self.supercell_inv, s)

            # Wrap atoms:
            s_sup_wrp = np.copy(s_sup)
            s_sup_wrp[dirs] -= np.floor(s_sup_wrp[dirs])

            # Snap to 0:
            # s_sup_wrp = vectors.snap_arr_to_val(s_sup_wrp, 0, 1e-12)

            # Convert back to Cartesian basis
            s_std_wrp = np.dot(self.supercell, s_sup_wrp)

            # Update attributes:
            sites_arr[s_idx][:] = s_std_wrp

    @property
    def supercell_inv(self):
        return np.linalg.inv(self.supercell)

    @property
    def atom_sites_frac(self):
        return np.dot(self.supercell_inv, self.atom_sites)

    @property
    def lattice_sites_frac(self):
        if self.lattice_sites is not None:
            return np.dot(self.supercell_inv, self.lattice_sites)
        else:
            return None

    @property
    def interstice_sites_frac(self):
        if self.interstice_sites is not None:
            return np.dot(self.supercell_inv, self.interstice_sites)
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
        """Get the species of each atom as a string array."""
        return self.species[self.species_idx]

    @property
    def spglib_cell(self):
        """Returns a tuple representing valid input for the spglib library."""

        cell = (self.supercell.T,
                self.atom_sites_frac.T,
                [mendeleev.element(i).atomic_number for i in self.all_species])
        return cell

    @property
    def num_atoms_per_crystal(self):
        """Computes number of atoms in each crystal, returns a list."""

        if self.crystals is None:
            return None

        na = []
        for c_idx in range(len(self.crystals)):
            crystal_idx_tup = self.atom_labels['crystal_idx']
            crystal_idx = crystal_idx_tup[0][crystal_idx_tup[1]]
            na.append(np.where(crystal_idx == c_idx)[0].shape[0])

        return na

    @property
    def num_atoms(self):
        """Computes total number of atoms."""
        return self.atom_sites.shape[1]

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

        recip = self.reciprocal_supercell
        grid = np.ceil(np.round(
            np.linalg.norm(recip, axis=0) / (separation * 2 * np.pi),
            decimals=8)
        ).astype(int)

        return grid

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

        recip = self.reciprocal_supercell
        seps = np.linalg.norm(recip, axis=0) / (np.array(grid) * 2 * np.pi)

        return seps

    @property
    def crystal_centres(self):
        """Get the midpoints of each crystal in the structure."""

        return [get_box_centre(c['crystal'], origin=c['origin'])
                for c in self.crystals]

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

    def shift_atoms(self, shift, wrap=False):
        """
        Perform a rigid shift on all atoms, in fractional supercell coordinates.

        Parameters
        ----------
        shift : list or tuple of length three or ndarry of shape (3,) of float
            Fractional supercell coordinates to translate all atoms by.
        wrap : bool
            If True, wrap atoms to within the supercell edges after shift.
        """

        shift = np.array(shift)[:, np.newaxis]
        shift_std = np.dot(self.supercell, shift)
        self.atom_sites += shift_std

        if wrap:
            self.wrap_sites_to_supercell(sites='atom')

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
        sup = self.supercell
        return np.dot(np.cross(sup[:, 0], sup[:, 1]), sup[:, 2])
