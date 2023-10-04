:mod:`hexsample.hexagon` --- Hexagonal sampling
===============================================


The module is an attempt at having a fully-fledged version of all the necessary
facilities related to the geometry on an hexagonal grid all in the same place,
most of the information coming from https://www.redblobgames.com/grids/hexagons/.


As a very quick, top level recap, hexagonal grids are classified according
to their orientation as either pointy-top (with staggered rows, in a vertical layout)
or flat-top (with staggered columns, in a horizontal layout).
Each layout comes into versions, depending on whether even or odd rows/columns
are shoved right/down. We end up with the four basic arrangements shown in the
figure below, all of which are supported.

.. figure:: figures/hexagonal_layouts.png

The module provides a :class:`HexagonalGrid <hexsample.hexagon.HexagonalGrid>` class,
representing an hexagonal grid, whose main public interfaces,
:meth:`pixel_to_world(col, row) <hexsample.hexagon.HexagonalGrid.pixel_to_world>`
and :meth:`world_to_pixel(x, y) <hexsample.hexagon.HexagonalGrid.world_to_pixel>`,
allow to go back and forth from logical to physical coordinates and vice versa.
It is worth pointing out that, in all four cases, the origin of the physical
reference system is assumed to be at the center of the grid, as illustrated in the
image.


Module documentation
--------------------

.. automodule:: hexsample.hexagon
