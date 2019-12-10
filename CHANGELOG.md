# Change Log

## [0.3.1] - 2019.10.12

### Added

- New `BulkCrystal` class
- New `AtomisticSimulation` class to represent a simulation that may have involved a relaxation.
- Added method `set_voronoi_tessellation` to `AtomisticStructure` class, which sets the `tessellation` attribute to a `VoronoiTessellation` object from the new `voronoi` module.
- Add `Bicrystal` methods: `get_point_in_boundary_plane` and `get_distance_from_boundary`.
- Add `Atomistic` method: `same_atoms` to check for equivalent atom positions.

### Fixed

- Fix non-implementation of `GammaSurface` fit size.
- `AtomisticTensileTest.get_traction_separation_plot_data` now returns sorted data

## [0.3.0] - 2019.09.15

### Added

- New `GammaSurface` and `GammaSurfaceCoordinate` classes for generating, visualising and manipulating bicrystal gamma surfaces.
- New `AtomisticTensileTest` and `AtomisticTensileTestCoordinate` classess for generating, visualising and manipulating atomistic tensile tests.

### Changed

- Atoms now use default JMOL colours in `AtomisticStructure` visualisation.

## [0.2.1] - 2019.08.24

### Fixed

- Fixed bug in `AtomisticStructure._init_sites` where sites not associated with a crystal were not added.
- Remove requirement in `AtomisticStructure.get_visual` that `atoms` must have label `species_order`.

## [0.2.0] - 2019.08.23

### Changed

- Use `Sites` from package `spatial-sites` to represent atoms and lattice sites etc. Changing development status to "4 - Beta".
- Use package `gemo` for visualisations. Further refinement is necessary.

## [0.1.0] - 2019.06.04

### Added

- Initial release from extracting the relevant parts of [matsim](https://github.com/aplowman/matsim); but needs some further refinement, so setting initial PyPI development status to "2 - Pre-Alpha".
