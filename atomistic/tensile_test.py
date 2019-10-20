import copy
import json
from pathlib import Path

import numpy as np

from atomistic import ENERGY_PER_AREA_UNIT_CONV, TT_SUPERCELL_TYPE
from atomistic.bicrystal import Bicrystal
from atomistic.utils import zeropad


def requires_base_structure(func):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], AtomisticTensileTestCoordinate):
            tensile_test = args[0].tensile_test
        else:
            tensile_test = args[0]
        if not hasattr(tensile_test, '_base_structure'):
            msg = ('This method/property requires the `base_structure` attribute to be '
                   'set on the `AtomisticTensileTest`.')
            raise ValueError(msg)
        return func(*args, **kwargs)
    return wrapper


class AtomisticTensileTest(object):
    'Represents sets of bicrystals used in simulated tensile tests.'

    def __init__(self, base_structure, expansions, distribution='flat', data=None):
        """

        Parameters
        ----------
        base_structure : Bicrystal
            Base bicrystal whose gamma surface is to be (or was) investigated.
        expansions : ndarray of list of shape (N,)
            The set of expansions to apply in the out-of-boundary direction of
            `base_structure`.
        distribution : str, optional
            Determines how should the additional vacuum is distributed through the
            supercell. One of "sigmoid", "flat" (corresponding to a rigid grain shift)
            or "linear".
        data : dict, optional
            Dict whose keys are strings and labels are ndarrays of outer dimension (N, )            

        """

        expansions, data = self._validate(expansions, data)
        self.expansions = expansions
        self.data = data
        self.base_structure = base_structure
        self.distribution = distribution

    def __len__(self):
        return self.expansions.shape[0]

    def _validate(self, expansions, data):

        if expansions is None:
            expansions = np.empty((0, ))
        if data is None:
            data = {}

        expansions = np.array(expansions)

        if expansions.ndim != 1:
            msg = '`expansions` must have shape (N,), but has shape: {}'
            raise ValueError(msg.format(expansions.shape))

        for data_name, data_item in data.items():
            data[data_name] = np.array(data_item)
            if data[data_name].shape[0] != expansions.shape[0]:
                msg = ('Data named "{}" must have outer dimension of length {}, but has '
                       'shape {}.')
                raise ValueError(msg.format(data_name, expansions.shape[0]))

        return expansions, data

    def _validate_structure(self, structure):
        if not isinstance(structure, Bicrystal):
            raise ValueError('base_structure` must be a Bicrystal object.')
        return structure

    def get_expansion_idx(self, expansion):
        'Get the index of the coordinate with a given expansion.'

        idx = np.where(np.isclose(self.expansions, expansion))[0]
        if not idx.size:
            msg = 'No coordinate with expansion close to {}.'.format(expansion)
            raise ValueError(msg)
        assert len(idx) == 1
        idx = idx[0]

        return idx

    def get_coordinate(self, expansion):
        'Get coordinate with a given expansion.'

        idx = self.get_expansion_idx(expansion)
        coord = self.get_coordinate_by_index(idx)

        return coord

    def get_coordinate_by_index(self, index):
        'Get a coordinate by index.'
        return AtomisticTensileTestCoordinate(self, index)

    def all_coordinates(self):
        'Generate all coordinates.'

        for index in range(len(self)):
            yield AtomisticTensileTestCoordinate(self, index)

    @property
    @requires_base_structure
    def base_structure(self):
        return self._base_structure

    @base_structure.setter
    def base_structure(self, base_structure):
        if base_structure is not None:
            self._base_structure = self._validate_structure(base_structure)
            if TT_SUPERCELL_TYPE not in self._base_structure.meta['supercell_type']:
                self._base_structure.meta['supercell_type'].append(TT_SUPERCELL_TYPE)

    @classmethod
    def from_json_file(cls, base_structure, path):
        """Load a gamma surface from a base structure and a JSON file."""

        with Path(path).open() as handle:
            contents = json.load(handle)

        return cls(base_structure, **contents)

    def to_dict(self):
        'Generate a JSON-compatible dict of this GammaSurface (except the structure).'
        out = {
            'expansions': self.expansions.tolist(),
            'distribution': self.distribution,
            'data': {k: v.tolist() for k, v in self.data.items()},
        }
        return out

    def to_json_file(self, path):
        'Generate a JSON file of this GammaSurface (except the structure).'
        dct = self.to_dict()
        with Path(path).open('w') as handle:
            json.dump(dct, handle, indent=4, sort_keys=True)

        return path

    def add_interface_energy(self, bulk_energy, data_name='energy',
                             interface_data_name='interface_energy'):
        pass

    @requires_base_structure
    def add_traction_separation_energy(self, surface_A_energy, surface_B_energy,
                                       data_name='energy', ts_data_name='ts_energy'):
        print('add_traction_sep energy')

        total_area = 2 * self.base_structure.boundary_area
        surface_energy = surface_A_energy + surface_B_energy

        ts_data = []
        for i in self.data[data_name]:
            ts_data.append((i - surface_energy) / total_area)

        self.data[ts_data_name] = np.array(ts_data)

    def get_traction_separation_plot_data(self, ts_data_name='ts_energy', SI_energy=True):
        x = self.expansions
        y = self.data[ts_data_name] * ENERGY_PER_AREA_UNIT_CONV if SI_energy else 1

        srt_idx = np.argsort(x)
        x = x[srt_idx]
        y = y[srt_idx]

        plot_dat = {
            'x': x,
            'y': y,
        }
        return plot_dat


class AtomisticTensileTestCoordinate(object):

    def __init__(self, tensile_test, index):

        if index >= len(tensile_test):
            msg = ('No tensile test coordinate exists with index: {}. Number of '
                   'coordinates is: {}.')
            raise ValueError(msg.format(index, len(tensile_test)))

        self.tensile_test = tensile_test
        self.index = index
        self._structure = None

    @property
    def expansion(self):
        return self.tensile_test.expansions[self.index]

    @property
    def data(self):
        return {k: v[self.index] for k, v in self.tensile_test.data.items()}

    @property
    @requires_base_structure
    def structure(self):
        if not self._structure:
            structure = copy.deepcopy(self.tensile_test.base_structure)
            structure.apply_boundary_vac(
                self.tensile_test.expansions[self.index],
                self.tensile_test.distribution,
                wrap=False,
            )
            self._structure = structure
        return self._structure

    @property
    def coordinate_fmt(self):
        return '{}_{:+.3f}'.format(
            zeropad(self.index, len(self.tensile_test)), self.expansion)

    def __repr__(self):
        out = '{}(expansion={!r}'.format(
            self.__class__.__name__,
            self.expansion,
        )
        for k, v in self.data.items():
            out += ', {}={!r}'.format(k, v)
        out += ')'

        return out
