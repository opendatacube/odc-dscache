"""
Tools for dealing with datacube db
"""
import random
from typing import Any, Dict, Optional, Tuple

import datacube.utils.geometry as geom
import psycopg2
from datacube import Datacube
from datacube.api.grid_workflow import Tile
from datacube.config import LocalConfig
from datacube.model import Dataset, GridSpec

from .. import DatasetCache, TileIdx
from ._index import bin_dataset_stream, dataset_count, ordered_dss, all_datasets


def dictionary_from_product_list(
    dc, products, samples_per_product=10, dict_sz=8 * 1024, query=None
):

    """Get a sample of datasets from a bunch of products and train compression
    dictionary.

    dc -- Datcube object
    products -- list of product names
    samples_per_product -- number of datasets per product to use for training
    dict_sz -- size of dictionary in bytes
    """

    if isinstance(products, str):
        products = [products]

    if query is None:
        query = {}

    limit = samples_per_product * 10

    samples = []
    for p in products:
        dss = dc.find_datasets(product=p, limit=limit, **query)
        random.shuffle(dss)
        samples.extend(dss[:samples_per_product])

    if len(samples) == 0:
        return None

    return DatasetCache.train_dictionary(samples, dict_sz)  # type: ignore


def db_connect(cfg=None):
    """Create database connection from datacube config.

    cfg:
      None -- use default datacube config
      str  -- use config with a given name

      LocalConfig -- use loaded config object
    """
    if isinstance(cfg, str) or cfg is None:
        cfg = LocalConfig.find(env=cfg)

    cfg_remap = dict(
        dbname="db_database",
        user="db_username",
        password="db_password",
        host="db_hostname",
        port="db_port",
    )

    pg_cfg = {k: cfg.get(cfg_name, None) for k, cfg_name in cfg_remap.items()}

    return psycopg2.connect(**pg_cfg)


def mk_raw2ds(products):
    """Convert "raw" dataset to `datacube.model.Dataset`.

    products -- dictionary from product name to `Product` object (`DatasetType`)

    returns: function that maps: `dict` -> `datacube.model.Dataset`

    This function can raise `ValueError` if dataset product is not found in the
    supplied products dictionary.


    Here "raw dataset" is just a python dictionary with fields:

    - product: str -- product name
    - uris: [str] -- list of dataset uris
    - metadata: dict -- dataset metadata document

    see `raw_dataset_stream`

    """

    def raw2ds(ds):
        product = products.get(ds["product"], None)
        if product is None:
            raise ValueError(f"Missing product: {ds['product']}")
        return Dataset(product, ds["metadata"], uris=ds["uris"])

    return raw2ds


def raw_dataset_stream(product, db, read_chunk=100, limit=None):
    """Given a product name stream all "active" datasets from db that belong to that product.

    Datasets are returned in "raw form", basically just a python dictionary with fields:

    - product: str -- product name
    - uris: [str] -- list of dataset uris
    - metadata: dict -- dataset metadata document
    """

    assert isinstance(limit, (int, type(None)))

    if isinstance(db, str) or db is None:
        db = db_connect(db)

    _limit = f"LIMIT {limit:d}" if limit else ""
    query = f"""
select
jsonb_build_object(
  'product', %(product)s,
  'uris', array((select _loc_.uri_scheme ||':'||_loc_.uri_body
                 from agdc.dataset_location as _loc_
                 where _loc_.dataset_ref = agdc.dataset.id and _loc_.archived is null
                 order by _loc_.added desc, _loc_.id desc)),
  'metadata', metadata) as dataset
from agdc.dataset
where archived is null
and dataset_type_ref = (select id from agdc.dataset_type where name = %(product)s)
{_limit};
"""

    cur = db.cursor(name=f"c{random.randint(0, 0xFFFF):04X}")
    cur.execute(query, dict(product=product))

    while True:
        chunk = cur.fetchmany(read_chunk)
        if not chunk:
            break

        for (ds,) in chunk:
            yield ds

    cur.close()


def gs_albers():
    return GridSpec(
        crs=geom.CRS("EPSG:3577"), tile_size=(100000.0, 100000.0), resolution=(-25, 25)
    )


# pylint: disable=too-few-public-methods
class DcTileExtract:
    """Construct ``datacube.api.grid_workflow.Tile`` object from dataset cache."""

    def __init__(self, cache, grid=None, group_by="time"):

        gs = cache.grids.get(grid, None)
        if gs is None:
            raise ValueError(f"No such grid: ${grid}")

        self._cache = cache
        self._grid = grid
        self._gs = gs
        self._default_groupby = group_by

    def __call__(self, tile_idx, _y=None, group_by=None):
        if _y is not None:
            tile_idx = (tile_idx, _y)

        if group_by is None:
            group_by = self._default_groupby

        dss = list(self._cache.stream_grid_tile(tile_idx, grid=self._grid))
        sources = Datacube.group_datasets(dss, group_by)

        geobox = self._gs.tile_geobox(tile_idx)
        return Tile(sources, geobox)


def grid_tiles_to_geojson(
    cache: DatasetCache,
    grid: str,
    style: Optional[Dict[str, Any]] = None,
    wrapdateline: bool = False,
) -> Dict[str, Any]:
    """
    Render tiles of a given grid to GeoJSON.

    each tile is a GeoJSON Feature with following properties:

     .title   -- str: grid index as a string
     .count   -- int: count of datasets overlapping with this grid tile

    :param cache: Dataset cache from which to read gridspec and tiles
    :param grid: Name of the grid to dump
    :param style: Optional style dictionary (will be included in every tile)
    :param wrapdateline: If set check for lon=180 intersect and do "the right thing" (slower)
    """
    if style is None:
        # these are understood by github renderer
        style = {"fill-opacity": 0, "stroke-width": 0.5}

    gs = cache.grids.get(grid, None)
    if gs is None:
        raise ValueError(f"No such grid: {grid}")

    resolution = abs(gs.tile_size[0]) / 4  # up to 4 points per side

    def mk_feature(tidx: TileIdx, count: int) -> Dict[str, Any]:
        if len(tidx) == 3:
            _xy: Tuple[int, int] = tidx[1:]  # type: ignore
        else:
            _xy: Tuple[int, int] = tidx  # type: ignore

        return dict(
            type="Feature",
            geometry=gs.tile_geobox(_xy)
            .extent.to_crs(
                "epsg:4326", resolution=resolution, wrapdateline=wrapdateline
            )
            .json,
            properties={
                "title": f"{_xy[0]:+05d},{_xy[1]:+05d}",
                "count": count,
                **style,  # type: ignore
            },
        )

    features = [mk_feature(tidx, cc) for tidx, cc in cache.tiles(grid)]

    return {"type": "FeatureCollection", "features": features}
