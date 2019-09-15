# Change Log

## [0.3.0] - 2019.09.15

### Added

- New `GammaSurface` and `GammaSurfaceCoordinate` classes for generating, visualising and manipulating bicrystal gamma surfaces.

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
