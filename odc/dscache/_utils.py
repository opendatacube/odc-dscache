"""Utils for compatibility with odc-geo"""

from odc.geo.types import XY, xy_, res_, SomeResolution


def to_tile_shape(tile_size: tuple, sres: SomeResolution) -> XY:
    # Convert tile size to tile shape
    # Assumes tile size has been provided in (y,x)
    res = res_(sres)
    tsz_y, tsz_x = tile_size
    return xy_(tsz_x / abs(res.x), tsz_y / abs(res.y))
