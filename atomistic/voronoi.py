"""`atomistic.voronoi.py`"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull

from gemo import Box, GeometryGroup, Sites


def get_tiled_sites(box, sites, tiles):
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

    """

    invalid_msg = ('`tiles` must be a tuple or list of three integers '
                   'greater than 0.')

    if isinstance(tiles, np.ndarray):
        tiles = np.squeeze(tiles).tolist()

    if len(tiles) != 3:
        raise ValueError(invalid_msg)

    sites_tiled = np.copy(sites)

    for t_idx, t in enumerate(tiles):

        if t == 1:
            continue

        if not isinstance(t, int) or t < 1:
            raise ValueError(invalid_msg)

        v = box[:, t_idx:t_idx + 1]
        v_range = v * np.arange(1, t)

        all_t = v_range.T[:, :, np.newaxis]

        sites_stack = all_t + sites_tiled
        add_sites = np.hstack(sites_stack)
        sites_tiled = np.hstack([sites_tiled, add_sites])

    return sites_tiled


def tile_supercell(box, atoms, tiles):
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

    atoms = get_tiled_sites(box, atoms, tiles)
    box = box * tiles

    return (box, atoms)


def area_polygon_3d(verts, normal):
    """Find the signed area of a polygon in 3D space.

    Parameters
    ----------
    verts : ndarray of shape (3, N)
        Array of column vectors describing the vertices of the polygon.
    normal: ndarray of shape (3, 1)
        Normal column vector, determining the sign.

    Returns
    -------
    area : float

    """
    srt_idx = order_coplanar_points(verts, normal)
    #print('srt_idx: \n{}\n'.format(srt_idx))

    verts = verts[:, srt_idx]
    #print('verts (srt): \n{}\n'.format(verts))

    verts_a = verts
    verts_b = np.roll(verts, -1, axis=1)

    #print('verts_a: \n{}\n'.format(verts_a))
    #print('verts_b: \n{}\n'.format(verts_b))

    verts_cross = np.cross(verts_a, verts_b, axis=0)
    verts_sum = np.sum(verts_cross, axis=1)

    #print('verts_cross: \n{}\n'.format(verts_cross))
    #print('verts_sum: \n{}\n'.format(verts_sum))

    area = np.einsum('ij,i->j', normal, verts_sum) / 2
    return area[0]


def order_coplanar_points(points, normal, anticlockwise=True):
    """
        `points` is an array of column three-vectors
        `normal` is a column three-vector, the normal vector of the plane on which all `points` lie.

        Returns the ordered indices of points according to an anticlockwise direction when looking in the opposite direction to `normal`

    """

    # Normalise `normal` to a unit vector:
    normal = normal / np.linalg.norm(normal)

    # Compute the centroid:
    centroid = np.mean(points, axis=1)

    # Get the direction vectors from each point to the centroid
    p = points - centroid[:, np.newaxis]

    # Use the first point as the reference point
    # Find cross product of each point with the reference point
    crs = np.cross(p[:, 0:1], p, axis=0)

    # Find the scalar triple product of the point pairs and the normal vector
    stp = np.einsum('ij,ik->k', normal, crs)

    # Find the dot product of the point pairs
    dot = np.einsum('ij,ik->k', p[:, 0:1], p)

    # Find signed angles from reference point to each other point
    ang = np.arctan2(stp, dot)
    ang_order = np.argsort(ang)

    if not anticlockwise:
        ang_order = ang_order[::-1]

    return ang_order


def split_1d_list(lst, lengths):
    """Split a 1D list into a list of lists, whose lengths are given in lengths"""

    if len(lst) != sum(lengths):
        raise ValueError(
            'Length of `lst` is not equal to the sum of `lengths`.')

    a_cumlen = np.cumsum([0] + lengths)
    lst_split = [lst[a_cumlen[i]:a_cumlen[i + 1]]
                 for i, _ in enumerate(lengths)]

    return lst_split


def collapse_to_bigger_column(arr):
    """Compress a 2D array (N, 2) into a 1D array (N,), where only the largest value from
    each row is kept."""

    out = -np.ones(arr.shape[0], dtype=int)

    big_left_idx = arr[:, 0] > arr[:, 1]
    big_right_idx = np.logical_not(big_left_idx)

    out[big_left_idx] = arr[big_left_idx, 0]
    out[big_right_idx] = arr[big_right_idx, 1]

    return out


def get_box_xyz(box, origin=None, faces=False):
    """
    Get coordinates of paths which trace the edges of parallelepipeds
    defined by edge vectors and origins. Useful for plotting parallelepipeds.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelepipeds, each as three 3D column vectors which
        define the edges of the parallelepipeds.
    origin : ndarray of shape (3, N) or (3,)
        Array defining the N origins of N parallelepipeds as 3D column vectors.
    faces : bool, optional
        If False, returns an array of shape (N, 3, 30) where the coordinates of
        a path tracing the edges of each of N parallelepipeds are returned as
        column 30 vectors.

        If True, returns a dict where the coordinates for
        each face is a key value pair. Keys are like `face01a`, where the
        numbers refer to the column indices of the vectors in the plane of the
        face to plot, the `a` faces intersect the origin and the `b` faces are
        parallel to the `a` faces. Values are arrays of shape (N, 3, 5), which
        define the coordinates of a given face as five 3D column vectors for
        each of the N input parallelepipeds.

    Returns
    -------
    ndarray of shape (N, 3, 30) or dict of str : ndarray of shape (N, 3, 5)
    (see `faces` parameter).

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    N = box.shape[0]

    if origin is None:
        origin = np.zeros((3, N), dtype=box.dtype)

    elif origin.ndim == 1:
        origin = origin[:, np.newaxis]

    if origin.shape[1] != box.shape[0]:
        raise ValueError('If `origin` is specified, there must be an origin '
                         'specified for each box.')

    c = get_box_corners(box, origin=origin)

    face01a = c[:, :, [0, 1, 4, 2, 0]]
    face01b = c[:, :, [3, 5, 7, 6, 3]]
    face02a = c[:, :, [0, 1, 5, 3, 0]]
    face02b = c[:, :, [2, 4, 7, 6, 2]]
    face12a = c[:, :, [0, 2, 6, 3, 0]]
    face12b = c[:, :, [1, 4, 7, 5, 1]]

    coords = [face01a, face01b, face02a, face02b, face12a, face12b]

    if not faces:
        xyz = np.concatenate(coords, axis=2)

    else:
        faceNames = ['face01a', 'face01b', 'face02a',
                     'face02b', 'face12a', 'face12b']
        xyz = dict(zip(faceNames, coords))

    return xyz


class VoronoiTessellation(object):
    'Perform a Voronoi tessellation on a set of (periodic) atoms.'

    def __init__(self, box, atoms):

        tes = self._get_tessellation(box, atoms)

        self.box = box
        self.points = atoms

        self.vertices = tes['vertices']

        self.point_volumes = tes['point_volumes']
        self.point_volumes_fractional = tes['point_volumes_fractional']
        self.point_vertices = tes['point_vertices']
        self.point_facets = tes['point_facets']
        self.point_number_facets = tes['point_number_facets']

        self.facet_vertices = tes['facet_vertices']
        self.facet_points = tes['facet_points']
        self.facet_points_periodic = tes['facet_points_periodic']
        self.facet_areas = tes['facet_areas']

        self.neighbour_distances = tes['neighbour_distances']
        self.neighbour_vectors = tes['neighbour_vectors']

        self.ridge_points_external_idx = tes['ridge_points_external_idx']
        self.external_points = tes['external_points']

    def _get_tessellation(self, box, atoms):

        natoms = atoms.shape[1]
        box_t, atoms_t = tile_supercell(box, atoms, (3, 3, 3))
        mid_box_idx = 14
        mid_atoms_idx_range = np.array([mid_box_idx - 1, mid_box_idx]) * natoms
        mid_atoms_idx = np.arange(*mid_atoms_idx_range)

        # print('mid_atoms_idx_range: {}'.format(mid_atoms_idx_range))

        atoms_t = atoms_t.T
        vor = Voronoi(atoms_t)

        # print('atoms_t.shape: {}'.format(atoms_t.shape))

        # Get the ridge indices that correspond to points in mid_atoms_idx:
        ridge_points_mid_atoms = np.logical_and(
            vor.ridge_points >= mid_atoms_idx[0],
            vor.ridge_points <= mid_atoms_idx[-1]
        )

        ridge_idx = np.where(np.any(ridge_points_mid_atoms, axis=1))[0]

        ridge_points_internal_map = ridge_points_mid_atoms[ridge_idx]

        ridge_points_external_map = np.logical_not(ridge_points_internal_map)

        # print('ridge_points_internal_map: \n{}\n'.format(ridge_points_internal_map))
        # print('ridge_points_internal_map.shape: {}'.format(
        # ridge_points_internal_map.shape))

        # print('ridge_points_external_map: \n{}\n'.format(ridge_points_external_map))
        # print('ridge_points_external_map.shape: {}'.format(
        #     ridge_points_external_map.shape))

        base_ridge_points = vor.ridge_points[ridge_idx]
        # print('base_ridge_points: \n{}\n'.format(base_ridge_points))
        # print('base_ridge_points.shape: {}'.format(base_ridge_points.shape))
        # print('atoms_t: \n{}\n'.format(atoms_t))

        neighbour_vecs = (atoms_t[base_ridge_points[:, 0]] -
                          atoms_t[base_ridge_points[:, 1]])
        region_distances = np.sqrt(np.sum(neighbour_vecs ** 2, axis=1))

        #print('region_distances: \n{}\n'.format(region_distances))

        base_ridge_points_external = np.copy(base_ridge_points)
        base_ridge_points_external[ridge_points_internal_map] = -1
        # print('base_ridge_points_external: \n{}\n'.format(base_ridge_points_external))

        base_ridge_points_external = collapse_to_bigger_column(
            base_ridge_points_external)
        # print('base_ridge_points_external: \n{}\n'.format(base_ridge_points_external))

        ridge_points_ext_uq, uq_inv = np.unique(
            base_ridge_points_external, return_inverse=True)

        ridge_points_ext_uq = ridge_points_ext_uq[1:]
        uq_inv -= 1

        # print('ridge_points_ext_uq: \n{}\n'.format(ridge_points_ext_uq))
        # print('uq_inv: \n{}\n'.format(uq_inv))

        ridge_points_external_idx = uq_inv

        ext_atoms = atoms_t[ridge_points_ext_uq]
        ext_atoms -= box[:, 0]
        ext_atoms -= box[:, 1]
        ext_atoms -= box[:, 2]

        # print('ext_atoms: \n{}\n'.format(ext_atoms))

        point_idx_wrap = np.array(list(range(natoms)) * 27)
        # print('point_idx_wrap: \n{}\n'.format(point_idx_wrap))

        base_ridge_points_mapped_periodic = point_idx_wrap[base_ridge_points]

        # print('base_ridge_points_mapped_periodic: \n{}\n'.format(
        #     base_ridge_points_mapped_periodic))

        base_ridge_points_mapped = np.copy(base_ridge_points_mapped_periodic)
        base_ridge_points_mapped[ridge_points_external_map] = -1

        # base_ridge_points_mapped_external = np.copy(
        # base_ridge_points_mapped_periodic)
        # base_ridge_points_mapped_external[ridge_points_internal_map] = -1

        # print('base_ridge_points_mapped: \n{}\n'.format(base_ridge_points_mapped))
        # print('base_ridge_points_mapped_external: \n{}\n'.format(
        # base_ridge_points_mapped_external))

        external_ridge_points = base_ridge_points[ridge_points_external_map]
        # print('external_ridge_points: \n{}\n'.format(external_ridge_points))

        vor.vertices -= box[:, 0]
        vor.vertices -= box[:, 1]
        vor.vertices -= box[:, 2]

        # Get which region each base atom belongs
        base_point_region = np.array(vor.point_region[mid_atoms_idx])
        base_point_region_srt_idx = np.argsort(base_point_region)

        # Get vertices for each base region:
        base_regions = [vor.regions[i] for i in base_point_region]
        base_regions_len = [len(i) for i in base_regions]
        base_regions_flat = np.concatenate(base_regions)
        base_regions_split = split_1d_list(base_regions_flat, base_regions_len)

        base_regions_uniq = np.unique(base_regions_flat)

        new_idx = [None] * (max(base_regions_uniq) + 1)
        for i_idx, i in enumerate(base_regions_uniq):
            new_idx[i] = i_idx

        new_idx = np.array(new_idx)

        base_regions_new_flat = list(new_idx[base_regions_flat])
        base_regions_new = split_1d_list(base_regions_new_flat, base_regions_len)
        base_regions_new_srt = [base_regions_new[i] for i in base_point_region_srt_idx]
        base_facets = [list(new_idx[vor.ridge_vertices[i]]) for i in ridge_idx]
        base_vertices = vor.vertices[base_regions_uniq]

        #print('base_facets: \n{}\n'.format(base_facets))
        #print('base_vertices: \n{}\n'.format(base_vertices))

        # Find area of each facet:
        facet_areas = []
        for i_idx, i in enumerate(base_facets):

            facet_vertices = base_vertices[i]
            facet_norm = neighbour_vecs[i_idx]
            area_i = area_polygon_3d(facet_vertices.T, facet_norm[:, None])
            facet_areas.append(area_i)

        facet_areas = np.array(facet_areas)

        # Find volume of each region:
        point_vols = np.array(
            [ConvexHull(base_vertices[i]).volume for i in base_regions_new])

        vol_sum = np.sum(point_vols)
        box_vol = np.dot(np.cross(box[:, 0], box[:, 1]), box[:, 2])

        if not np.isclose(vol_sum, box_vol):
            msg = 'Sum of Voronoi volumes ({}) is not equal to the box volume ({}).'
            print(msg.format(vol_sum, box_vol))
            # raise ValueError(msg.format(vol_sum, box_vol))

        # Find which facets border each point:
        point_facets = []
        for i in range(natoms):
            w = np.where(base_ridge_points_mapped == i)[0]
            point_facets.append(w)

        ret = {
            # Array of column vectors representing Voronoi vertices:
            'vertices': base_vertices,

            # Volume of each atom, according to Voronoi tessellation:
            'point_volumes': point_vols,
            'point_volumes_fractional': point_vols / box_vol,

            # The `vertices` indices which form each region:
            'point_vertices': base_regions_new,
            'point_facets': point_facets,
            'point_number_facets': [len(i) for i in point_facets],

            # The indices of `vertices` which form a facet:
            'facet_vertices': base_facets,
            'facet_points': base_ridge_points_mapped,
            'facet_points_periodic': base_ridge_points_mapped_periodic,

            'facet_areas': facet_areas,

            'neighbour_distances': region_distances,
            'neighbour_vectors': neighbour_vecs,

            'ridge_points_external_idx': ridge_points_external_idx,
            'external_points': ext_atoms,
        }

        return ret

    def get_geometry_group_points(self, include_atoms, show_vertices, show_ridges,
                                  show_atoms=False):

        include_atoms = self._validate_include_atoms_arg(include_atoms)
        points = {}
        if show_atoms:
            points.update({
                'atoms': Sites(
                    self.points,
                    labels={
                        'volume': self.point_volumes,
                    }
                ),
            })

        if show_vertices:
            # Atom vertices for given atoms:
            for atom_idx in include_atoms:
                atom_vert_idx = np.array(self.point_vertices)[atom_idx]
                atom_verts = self.vertices[atom_vert_idx].T
                points.update({'vertices_{}'.format(atom_idx): Sites(atom_verts)})

        return points

    def get_geometry_group_lines(self, include_atoms, show_vertices, show_ridges):

        include_atoms = self._validate_include_atoms_arg(include_atoms)

        lines = {}
        if show_ridges:
            all_facet_atom_idx = []
            for facet_idx, i in enumerate(self.facet_points):

                # TODO: part of this can be done without looping since `facet_points` is
                # an (N, 2) array.

                inc_facet = False
                facet_atom_idx = None

                for j in i:  # loop over two atom indices that form this facet
                    if j in include_atoms:
                        inc_facet = True
                        facet_atom_idx = j
                        if j in all_facet_atom_idx:
                            pass
                        else:
                            all_facet_atom_idx.append(j)
                        break

                if inc_facet:
                    facet_vert_idx = self.facet_vertices[facet_idx]
                    facet_verts = self.vertices[facet_vert_idx].T  # shape (3, N)
                    facet_verts_roll = np.roll(facet_verts, -1, axis=1)
                    facet_lines = np.concatenate(
                        [facet_verts[:, None].T, facet_verts_roll[:, None].T],
                        axis=1
                    ).swapaxes(1, 2)  # shape (N, 3, 2)

                    if 'atom_ridges_{}'.format(facet_atom_idx) in lines:
                        lines['atom_ridges_{}'.format(facet_atom_idx)] = np.concatenate(
                            [
                                lines['atom_ridges_{}'.format(facet_atom_idx)],
                                facet_lines,
                            ]
                        )
                    else:
                        lines.update({
                            'atom_ridges_{}'.format(facet_atom_idx): facet_lines
                        })

        return lines

    def get_geometry_group_boxes(self, include_atoms, show_vertices, show_ridges):

        boxes = {'supercell': Box(edge_vectors=self.box)}
        return boxes

    def get_geometry_group(self, include_atoms, show_vertices, show_ridges):
        'Get the GeometryGroup object for visualisation.'

        points = self.get_geometry_group_points(
            include_atoms, show_vertices, show_ridges, show_atoms=True)
        lines = self.get_geometry_group_lines(include_atoms, show_vertices, show_ridges)
        boxes = self.get_geometry_group_boxes(include_atoms, show_vertices, show_ridges)
        gg = GeometryGroup(points=points, boxes=boxes, lines=lines)

        return gg

    def _validate_include_atoms_arg(self, include_atoms):
        if include_atoms is None:
            include_atoms = 'all'

        if include_atoms == 'all':
            include_atoms = np.arange(self.points.shape[1])
        else:
            include_atoms = np.array(include_atoms)
        return include_atoms

    def show(self, layout_args=None, include_atoms=None, show_vertices=False,
             show_ridges=True):

        # TODO: colour_atoms_by_volume
        gg = self.get_geometry_group(include_atoms, show_vertices, show_ridges)
        style_points = {
            key: {
                'marker_symbol': 'cross',
                'marker_size': 4,
            } for key, val in gg.points.items() if 'vertices_' in key
        }

        return gg.show(style_points=style_points, layout_args=layout_args)
