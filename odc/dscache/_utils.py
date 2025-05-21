"""Utils for compatibility with odc-geo"""

from odc.geo.types import XY, xy_, res_


def to_tile_shape(tile_size, res) -> XY:
    # Convert tile size to tile shape
    # Assumes tile size has been provided in (y,x)
    res = res_(res)
    tsz_x = tile_size[1]
    tsz_y = tile_size[0]
    return xy_(tsz_x / abs(res.x), tsz_y / abs(res.y))
