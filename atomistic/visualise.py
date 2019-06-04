"""`atomistic.visualise.py`

Module containing code to visualise structures: CrystalStructure, CrystalBox,
AtomisticStructure.

Each of these classes share common instance attributes:
    `atom_sites`, `species`, `species_idx`, `lattice_sites` (optional),
    `bulk_interstitial_sites` (optional), `atom_labels` (optional)

"""

import os
import pickle

import numpy as np
from plotly import graph_objs
from plotly.offline import plot, iplot
from vecmaths.geometry import get_box_xyz

from atomistic import DATA_DIR
from atomistic.utils import combination_idx


def visualise_structure(structure, show_iplot=False, save=False, save_args=None,
                        plot_2d='xyz', ret_fig=False, group_atoms_by=None,
                        group_lattice_sites_by=None, group_interstices_by=None,
                        wrap_sym_op=True, style=None):
    """
    Parameters
    ----------
    structure : one of CrystalStructure, CrystalBox or AtomisticStructure.
    use_interstice_names : bool, optional
        If True, bulk interstices are plotted by names given in
        `interstice_names` according to `interstice_names_idx`.
    group_atoms_by : list of str, optional
        If set, atoms are grouped according to one or more of their labels.
        For instance, if set to `species_count`, which is an atom label that is
        automatically added to the CrystalStructure, atoms will be grouped by
        their position in the motif within their species. So for a motif which
        has two X atoms, these atoms will be plotted on separate traces:
        "X (#1)" and "X (#2)". Note that atoms are grouped by species
        (e.g. "X") by default.
    group_lattice_sites_by : list of str, optional
        If set, lattice sites are grouped according to one or more of their
        labels.
    group_interstices_by : list of str, optional
        If set, interstices are grouped according to one or more of their
        labels.

    TODO: add `colour_{atoms, lattice_sites, interstices}_by` parameters which 
          will be a string that must be in the corresponding group_{}_by list
          Or maybe don't have this restriction, would ideally want to be able
          to colour according to a colourscale e.g. by volume per atom, bond
          order parameter, etc. Can do this in Plotly by setting 
          marker.colorscale to an array of the same length as the number of 
          markers. And for Matplotlib: https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter

    TODO: consider merging parameters into a dict: 
          `group_sites_by` = {
              atoms: [...], lattice_sites: [...], interstices: [...]} etc.

    """

    style_def = {
        'width': 500,
        'height': 500,
    }
    style = style or {}
    style = {**style_def, **style}

    if save:
        if save_args is None:
            save_args = {
                'filename': 'plots.html',
                'auto_open': False
            }
        elif save_args.get('filename') is None:
            save_args.update({'filename': 'plots.html'})

    if group_atoms_by is None:
        group_atoms_by = []

    if group_lattice_sites_by is None:
        group_lattice_sites_by = []

    if group_interstices_by is None:
        group_interstices_by = []

    for lab in group_atoms_by:
        if lab not in structure.atom_labels.keys():
            raise ValueError(
                '"{}" is not a valid atom label.'.format(lab)
            )

    for lab in group_lattice_sites_by:
        if lab not in structure.lattice_labels.keys():
            raise ValueError(
                '"{}" is not a valid lattice site label.'.format(lab)
            )

    for lab in group_interstices_by:
        if lab not in structure.interstice_labels.keys():
            raise ValueError(
                '"{}" is not a valid interstice label.'.format(lab)
            )

    # Get colours for atom species:
    with DATA_DIR.joinpath('jmol_colours.pickle').open('rb') as handle:
        atom_cols = pickle.load(handle)

    # Add atom number labels:
    text = []
    text.append({
        'data': structure.atom_sites,
        'text': list(range(structure.atom_sites.shape[1])),
        'position': 'top center',
        'colour': 'gray',
        'name': 'Atom labels',
        'visible': 'legendonly',
    })

    points = []

    # Add atoms by groupings
    atom_groups_names = []
    atom_groups = []
    for k, v in structure.atom_labels.items():
        if k in group_atoms_by:
            atom_groups_names.append(k)
            atom_groups.append(v[0][v[1]])

    atm_col = 'black'
    atm_sym = 'circle'

    if len(atom_groups) > 0:
        atom_combs, atom_combs_idx = combination_idx(*atom_groups)

        for ac_idx in range(len(atom_combs)):

            c = atom_combs[ac_idx]
            c_idx = atom_combs_idx[ac_idx]
            skip_idx = []
            atoms_name = 'Atoms'

            # Special treatment for species and species_count if grouping requested:
            if 'species' in atom_groups_names:
                sp_group_name_idx = atom_groups_names.index('species')
                sp = c[sp_group_name_idx]
                atm_col = 'rgb' + str(atom_cols[sp])

                atoms_name += ': {}'.format(sp)
                skip_idx = [sp_group_name_idx]

                if 'species_count' in atom_groups_names:
                    sp_ct_group_name_idx = atom_groups_names.index(
                        'species_count')
                    atoms_name += ' #{}'.format(c[sp_ct_group_name_idx] + 1)
                    skip_idx.append(sp_ct_group_name_idx)

            for idx, (i, j) in enumerate(zip(atom_groups_names, c)):
                if idx in skip_idx:
                    continue
                atoms_name += '; {}: {}'.format(i, j)

            points.append({
                'data': structure.atom_sites[:, c_idx],
                'symbol': atm_sym,
                'colour': atm_col,
                'name': atoms_name,
            })

    else:
        points.append({
            'data': structure.atom_sites,
            'symbol': atm_sym,
            'colour': atm_col,
            'name': 'Atoms',
        })

    # Add lattice sites by groupings
    if structure.lattice_sites is not None:
        lat_groups_names = []
        lat_groups = []
        for k, v in structure.lattice_labels.items():
            if k in group_lattice_sites_by:
                lat_groups_names.append(k)
                lat_groups.append(v[0][v[1]])

        lat_col = 'grey'
        lat_sym = 'x'

        if len(lat_groups) > 0:
            lat_combs, lat_combs_idx = combination_idx(*lat_groups)

            for lc_idx in range(len(lat_combs)):
                c = lat_combs[lc_idx]
                c_idx = lat_combs_idx[lc_idx]
                skip_idx = []
                lats_name = 'Lattice sites'

                for idx, (i, j) in enumerate(zip(lat_groups_names, c)):
                    lats_name += '; {}: {}'.format(i, j)

                points.append({
                    'data': structure.lattice_sites[:, c_idx],
                    'symbol': lat_sym,
                    'colour': lat_col,
                    'name': lats_name,
                    'visible': 'legendonly',
                })

        else:
            points.append({
                'data': structure.lattice_sites,
                'symbol': lat_sym,
                'colour': lat_col,
                'name': 'Lattice sites',
                'visible': 'legendonly',
            })

    # Add interstices by groupings
    if structure.interstice_sites is not None:
        int_groups_names = []
        int_groups = []
        for k, v in structure.interstice_labels.items():
            if k in group_interstices_by:
                int_groups_names.append(k)
                int_groups.append(v[0][v[1]])

        int_col = 'orange'
        int_sym = 'x'

        if len(int_groups) > 0:
            int_combs, int_combs_idx = combination_idx(*int_groups)

            for ic_idx in range(len(int_combs)):
                c = int_combs[ic_idx]
                c_idx = int_combs_idx[ic_idx]
                skip_idx = []
                ints_name = 'Interstices'

                for idx, (i, j) in enumerate(zip(int_groups_names, c)):
                    ints_name += '; {}: {}'.format(i, j)

                points.append({
                    'data': structure.interstice_sites[:, c_idx],
                    'symbol': int_sym,
                    'colour': int_col,
                    'name': ints_name,
                })

        else:
            points.append({
                'data': structure.interstice_sites,
                'symbol': int_sym,
                'colour': int_col,
                'name': 'Interstices',
            })

    boxes = []

    if hasattr(structure, 'bravais_lattice'):
        # CrystalStructure
        boxes.append({
            'edges': structure.bravais_lattice.vecs,
            'name': 'Unit cell',
            'colour': 'navy'
        })

    if hasattr(structure, 'box_vecs'):
        # CrystalBox
        boxes.append({
            'edges': structure.box_vecs,
            'origin': structure.origin,
            'name': 'Crystal box',
            'colour': 'green',
        })

        # Add the bounding box trace:
        boxes.append({
            'edges': structure.bounding_box['bound_box'][0],
            'origin': structure.bounding_box['bound_box_origin'],
            'name': 'Bounding box',
            'colour': 'red',
        })

    if hasattr(structure, 'supercell'):

        # AtomisticStructure
        boxes.append({
            'edges': structure.supercell,
            'origin': structure.origin,
            'name': 'Supercell',
            'colour': '#98df8a',
        })

        crystal_cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for c_idx, c in enumerate(structure.crystals):

            boxes.append({
                'edges': c['crystal'],
                'origin': c['origin'],
                'name': 'Crystal #{}'.format(c_idx + 1),
                'colour': crystal_cols[c_idx],
            })

        # Add a symmetry operation
        if hasattr(structure, 'symmetry_ops'):
            if structure.symmetry_ops:
                so = structure.symmetry_ops[0]
                as_sym = np.dot(so[0], structure.atom_sites)
                as_sym += np.dot(structure.supercell, so[1][:, np.newaxis])

                if wrap_sym_op:
                    as_sym_frac = np.dot(structure.supercell_inv, as_sym)
                    as_sym_frac -= np.floor(as_sym_frac)
                    as_sym = np.dot(structure.supercell, as_sym_frac)

                points.append({
                    'data': as_sym,
                    'symbol': 'diamond-open',
                    'colour': 'purple',
                    'name': 'Atoms (symmetry)',
                })
                text.append({
                    'data': structure.atom_sites,
                    'text': np.arange(structure.num_atoms),
                    'position': 'bottom center',
                    'font': {
                        'color': 'purple',
                    },
                    'name': 'Atoms (symmetry labels)',
                })

                # # Add lines mapping symmetrically connected atoms:
                # for a_idx, a in enumerate(atom_sites_sym.T):

                #     data.append({
                #         'type': 'scatter3d',
                #         'x': [a[0], self.atom_sites.T[a_idx][0]],
                #         'y': [a[1], self.atom_sites.T[a_idx][1]],
                #         'z': [a[2], self.atom_sites.T[a_idx][2]],
                #         'mode': 'lines',
                #         'name': 'Sym op',
                #         'legendgroup': 'Sym op',
                #         'showlegend': False,
                #         'line': {
                #             'color': 'purple',
                #         },
                #     })

    f3d, f2d = plot_geometry_plotly(points, boxes, text, style=style)

    if show_iplot:
        iplot(f3d)
        iplot(f2d)

    if save:
        if plot_2d != '':
            div_2d = plot(f2d, **save_args, output_type='div',
                          include_plotlyjs=False)

        div_3d = plot(f3d, **save_args, output_type='div',
                      include_plotlyjs=True)

        html_all = div_3d + div_2d
        with open(save_args.get('filename'), 'w') as plt_file:
            plt_file.write(html_all)

    if ret_fig:
        return (f3d, f2d)


def plot_geometry_plotly(points=None, boxes=None, text=None, style=None,
                         plot_3d=True, plot_2d='xyz'):
    """

    Next time:
        factorise logic to get subplot sizes out so can share between plotly and mpl function


    """

    STYLE_DEF = {
        'width': 700,
        'height': 700,
        'aspect': 'equal',
        'labels': ['x', 'y', 'z'],
    }

    COLS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if points is None:
        points = []
    if boxes is None:
        boxes = []

    plot_dirs = []
    if 'x' in plot_2d:
        plot_dirs.append(0)
    if 'y' in plot_2d:
        plot_dirs.append(1)
    if 'z' in plot_2d:
        plot_dirs.append(2)

    ax_lab_2d = (('x', 'y'), ('x2', 'y'), ('x', 'y3'))
    show_leg_2d = (True, False, False)

    if len(plot_dirs) == 3:
        dirs_2d = ((1, 2), (0, 2), (1, 0))

    elif len(plot_dirs) == 2:

        if plot_dirs == [0, 1]:
            dirs_2d = ((1, 2), (0, 2), None)
            ax_lab_2d = (('x', 'y'), ('x2', 'y'), None)

        if plot_dirs == [0, 2]:
            dirs_2d = ((2, 1), None, (0, 1))
            ax_lab_2d = (('x', 'y'), None, ('x2', 'y'))

        if plot_dirs == [1, 2]:
            dirs_2d = (None, (2, 0), (1, 0))
            ax_lab_2d = (None, ('x', 'y'), ('x2', 'y'))

    elif len(plot_dirs) == 1:

        if plot_dirs == [0]:
            dirs_2d = ((1, 2), None, None)
            ax_lab_2d = (('x', 'y'), None, None)

        if plot_dirs == [1]:
            dirs_2d = (None, (0, 2), None)
            ax_lab_2d = (None, ('x', 'y'), None)

        if plot_dirs == [2]:
            dirs_2d = (None, None, (0, 1),)
            ax_lab_2d = (None, None, ('x', 'y'))

    style = {**STYLE_DEF, **(style or {})}

    data_3d = []
    data_2d = []

    layout_3d = {
        'width': style['width'],
        'height': style['height'],
    }
    scene = {
        'xaxis': {
            'title': style['labels'][0]
        },
        'yaxis': {
            'title': style['labels'][1]
        },
        'zaxis': {
            'title': style['labels'][2]
        }
    }
    if style['aspect'] == 'equal':
        scene.update({
            'aspectmode': 'data'
        })
    layout_3d.update({'scene': scene})

    for pts in points:
        com_d = {
            'mode': 'markers',
            'visible': pts.get('visible', True),
        }
        d = {
            'type': 'scatter3d',
            'marker': {
                'color': pts['colour'],
                'symbol': pts['symbol'],
                # Crosses are unusually large:
                'size': 5 if pts['symbol'] == 'x' else 8,
            },
            'x': pts['data'][0],
            'y': pts['data'][1],
            'z': pts['data'][2],
            **com_d,
        }
        if pts.get('name') is not None:
            d.update({'name': pts.get('name')})
        data_3d.append(d)

        for i in plot_dirs:

            d = {
                'type': 'scatter',
                'marker': {
                    'color': pts['colour'],
                    'symbol': pts['symbol'],
                },
                'x': pts['data'][dirs_2d[i][0]],
                'y': pts['data'][dirs_2d[i][1]],
                'xaxis': ax_lab_2d[i][0],
                'yaxis': ax_lab_2d[i][1],
                'showlegend': show_leg_2d[i],
                **com_d,
            }
            if pts.get('name') is not None:
                d.update(
                    {
                        'name': pts.get('name'),
                        'legendgroup': pts.get('name'),
                    }
                )
            data_2d.append(d)

    for txt in text:

        com_d = {
            'mode': 'text',
            'text': txt['text'],
            'textposition': txt.get('position', 'top center'),
            'textfont': txt.get('font', {}),
            'visible': txt.get('visible', True),
        }
        d = {
            'type': 'scatter3d',
            'x': txt['data'][0],
            'y': txt['data'][1],
            'z': txt['data'][2],
            **com_d,
        }
        if txt.get('name') is not None:
            d.update({'name': txt.get('name')})
        data_3d.append(d)

        for i in plot_dirs:

            d = {
                'type': 'scatter',
                'x': txt['data'][dirs_2d[i][0]],
                'y': txt['data'][dirs_2d[i][1]],
                'xaxis': ax_lab_2d[i][0],
                'yaxis': ax_lab_2d[i][1],
                'showlegend': show_leg_2d[i],
                **com_d,
            }
            if txt.get('name') is not None:
                d.update(
                    {
                        'name': txt.get('name'),
                        'legendgroup': txt.get('name'),
                    }
                )
            data_2d.append(d)

    for bx_idx, bx in enumerate(boxes):

        bx_def = {
            'colour': COLS[bx_idx],
            'origin': np.array([0, 0, 0]),
        }
        bx = {**bx_def, **(bx or {})}
        com_d = {
            'mode': 'lines',
            'line': {
                'color': bx['colour'],
            },
            'visible': bx.get('visible', True),
        }
        bx_trace = get_box_xyz(bx['edges'], origin=bx['origin'])[0]
        d = {
            'type': 'scatter3d',
            'x': bx_trace[0],
            'y': bx_trace[1],
            'z': bx_trace[2],
            **com_d,
        }
        if bx.get('name') is not None:
            d.update({'name': bx.get('name')})
        data_3d.append(d)

        for i in plot_dirs:

            d = {
                'type': 'scatter',
                'x': bx_trace[dirs_2d[i][0]],
                'y': bx_trace[dirs_2d[i][1]],
                'xaxis': ax_lab_2d[i][0],
                'yaxis': ax_lab_2d[i][1],
                'showlegend': show_leg_2d[i],
                **com_d,
            }
            if bx.get('name') is not None:
                d.update(
                    {
                        'name': bx.get('name'),
                        'legendgroup': bx.get('name'),
                    }
                )
            data_2d.append(d)

    # 2D projections layout
    # =====================

    hori_space = 0.05
    vert_space = 0.05

    # Get min and max data in plots and boxes:
    all_x = []
    all_y = []
    all_z = []
    for d in data_3d:
        all_x.extend(d['x'])
        all_y.extend(d['y'])
        all_z.extend(d['z'])

    min_x, min_y, min_z = [np.min(i) for i in [all_x, all_y, all_z]]
    max_x, max_y, max_z = [np.max(i) for i in [all_x, all_y, all_z]]

    x_rn = max_x - min_x
    y_rn = max_y - min_y
    z_rn = max_z - min_z

    y_frac = 1
    y3_frac = 0
    x_frac = 1
    x2_frac = 0

    if len(plot_dirs) == 3:
        tot_rn_vert = z_rn + x_rn
        tot_rn_hori = y_rn + x_rn

        y_frac = z_rn / tot_rn_vert
        y3_frac = x_rn / tot_rn_vert
        x_frac = y_rn / tot_rn_hori
        x2_frac = x_rn / tot_rn_hori

    elif len(plot_dirs) == 2:

        vert_space = 0

        if plot_dirs == [0, 1]:
            tot_rn_vert = z_rn
            tot_rn_hori = y_rn + x_rn
            x_frac = y_rn / tot_rn_hori
            x2_frac = x_rn / tot_rn_hori

        elif plot_dirs == [0, 2]:
            tot_rn_vert = y_rn
            tot_rn_hori = z_rn + x_rn
            x_frac = z_rn / tot_rn_hori
            x2_frac = x_rn / tot_rn_hori

        elif plot_dirs == [1, 2]:
            tot_rn_vert = x_rn
            tot_rn_hori = z_rn + y_rn
            x_frac = z_rn / tot_rn_hori
            x2_frac = y_rn / tot_rn_hori

    elif len(plot_dirs) == 1:

        hori_space = 0
        vert_space = 0

    xaxis1 = {
        'domain': [0, x_frac - hori_space / 2],
        'anchor': 'y',
    }
    yaxis1 = {
        'domain': [y3_frac + vert_space / 2, 1],
        'anchor': 'x',
        'scaleanchor': 'x',
    }
    xaxis2 = {
        'domain': [x_frac + hori_space / 2, 1],
        'anchor': 'y',
        'scaleanchor': 'y',
    }
    yaxis3 = {
        'domain': [0, y3_frac - vert_space / 2],
        'anchor': 'x',
        'scaleanchor': 'x',
    }

    if len(plot_dirs) == 3:
        xaxis1.update({
            'title': 'y',
            'side': 'top',
        })
        yaxis1.update({
            'title': 'z',
        })
        xaxis2.update({
            'title': 'x',
            'side': 'top',
        })
        yaxis3.update({
            'title': 'x',
        })

    elif len(plot_dirs) == 2:

        if plot_dirs == [0, 1]:
            xaxis1.update({'title': 'y', })
            yaxis1.update({'title': 'z', })
            xaxis2.update({
                'title': 'x',
            })

        elif plot_dirs == [0, 2]:
            xaxis1.update({'title': 'z', })
            yaxis1.update({'title': 'y', })
            xaxis2.update({
                'title': 'x',
            })

        elif plot_dirs == [1, 2]:
            xaxis1.update({'title': 'z', })
            yaxis1.update({'title': 'x', })
            xaxis2.update({
                'title': 'y',
            })

    elif len(plot_dirs) == 1:

        if plot_dirs == [0]:
            xaxis1.update({'title': 'y', })
            yaxis1.update({'title': 'z', })

        elif plot_dirs == [1]:
            xaxis1.update({'title': 'x', })
            yaxis1.update({'title': 'z', })

        elif plot_dirs == [2]:
            xaxis1.update({'title': 'x', })
            yaxis1.update({'title': 'y', })

    layout_2d = {
        'width': style['width'],
        'height': style['width'],
        'xaxis1': xaxis1,
        'yaxis1': yaxis1,
    }

    if len(plot_dirs) == 3:
        layout_2d.update({'yaxis3': yaxis3, })
    if len(plot_dirs) >= 2:
        layout_2d.update({'xaxis2': xaxis2, })

    fig_2d = graph_objs.Figure(data=data_2d, layout=layout_2d)
    fig_3d = graph_objs.Figure(data=data_3d, layout=layout_3d)

    return (fig_3d, fig_2d)
