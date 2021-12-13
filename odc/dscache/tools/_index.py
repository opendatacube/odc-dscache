"""These should probably be in datacube library."""
import datetime
from datetime import timedelta
from random import randint
from types import SimpleNamespace
from typing import Iterator, Optional, Set, Tuple
from uuid import UUID
from warnings import warn

import psycopg2
from datacube import Datacube
from datacube.api.query import Query
from datacube.model import Dataset, Range
from datacube.utils.geometry import Geometry
from pandas import Period


def dataset_count(index, **query):
    """Return number of datasets matching a query."""
    return index.datasets.count(**Query(**query).search_terms)


def count_by_year(index, product, min_year=None, max_year=None):
    """
    Return dictionary Int->Int: `year` -> `dataset count for this year`.

    Only non-empty years are reported.
    """
    # TODO: get min/max from datacube properly
    if min_year is None:
        min_year = 1970
    if max_year is None:
        max_year = datetime.datetime.now().year

    year_count = (
        (year, dataset_count(index, product=product, time=str(year)))
        for year in range(min_year, max_year + 1)
    )

    return {year: c for year, c in year_count if c > 0}


def count_by_month(index, product, year):
    """
    Return 12 integer tuple.

    counts for January, February ... December
    """
    return tuple(
        dataset_count(index, product=product, time=f"{year}-{month:02d}")
        for month in range(1, 12 + 1)
    )


def time_range(begin, end, freq="m"):
    """
    Return tuples of datetime objects aligned to boundaries of requested period.

    (month is default).
    """
    tzinfo = begin.tzinfo
    t = Period(begin, freq)

    def to_pydate(t):
        return t.to_pydatetime(warn=False).replace(tzinfo=tzinfo)

    while True:
        t0, t1 = map(to_pydate, (t.start_time, t.end_time))
        if t0 > end:
            break

        yield (max(t0, begin), min(t1, end))
        t += 1


def month_range(
    year: int, month: int, n: int
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Construct month aligned time range.

    Return time range covering n months starting from year, month
    month 1..12
    month can also be negative
    2020, -1 === 2019, 12
    """
    if month < 0:
        return month_range(year - 1, 12 + month + 1, n)

    y2 = year
    m2 = month + n
    if m2 > 12:
        m2 -= 12
        y2 += 1
    dt_eps = datetime.timedelta(microseconds=1)

    return (
        datetime.datetime(year=year, month=month, day=1),
        datetime.datetime(year=y2, month=m2, day=1) - dt_eps,
    )


def season_range(year: int, season: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Season is one of djf, mam, jja, son.

    DJF for year X starts in Dec X-1 and ends in Feb X.
    """
    seasons = dict(djf=-1, mam=3, jja=6, son=9)

    start_month = seasons.get(season.lower())
    if start_month is None:
        raise ValueError(f"No such season {season}, valid seasons are: djf,mam,jja,son")
    return month_range(year, start_month, 3)


def chop_query_by_time(q: Query, freq: str = "m") -> Iterator[Query]:
    """
    Split query along time dimension.

    Given a query over longer period of time, chop it up along the time dimension
    into smaller queries each covering a shorter time period (year, month, week or day).
    """
    qq = dict(**q.search_terms)
    time = qq.pop("time", None)
    if time is None:
        raise ValueError("Need time range in the query")

    for (t0, t1) in time_range(time.begin, time.end, freq=freq):
        yield Query(**qq, time=Range(t0, t1))


def ordered_dss(dc: Datacube, freq: str = "m", key=None, **query):
    """
    Emulate "order by time" streaming interface for datacube queries.

        Basic idea is to perform a lot of smaller queries (shorter time
        periods), sort results then yield them to the calling code.

    :param dc: Datacube instance

    :param freq: 'm' month sized chunks, 'w' week sized chunks, 'd' day

    :param key: Optional sorting function Dataset -> Comparable, for example
                ``lambda ds: (ds.center_time, ds.metadata.region_code)``
    """
    qq = Query(**query)
    if key is None:
        key = lambda ds: ds.center_time

    _last_uuids: Set[UUID] = set()

    for q in chop_query_by_time(qq, freq=freq):
        _dss = dc.find_datasets(**q.search_terms)
        dss = [ds for ds in _dss if ds.id not in _last_uuids]
        _last_uuids = {ds.id for ds in _dss}
        dss.sort(key=key)
        yield from dss


def chopped_dss(dc: Datacube, freq: str = "m", **query):
    """
    Emulate streaming interface for datacube queries.

    Basic idea is to perform a lot of smaller queries (shorter time
    periods)
    """
    qq = Query(**query)
    _last_uuids: Set[UUID] = set()

    for q in chop_query_by_time(qq, freq=freq):
        _dss = dc.find_datasets(**q.search_terms)
        dss = [ds for ds in _dss if ds.id not in _last_uuids]
        _last_uuids = {ds.id for ds in _dss}
        yield from dss


def bin_dataset_stream(gridspec, dss, cells, persist=None):
    """
    Intersect Grid Spec cells with Datasets.

    :param gridspec: GridSpec
    :param dss: Sequence of datasets (can be lazy)
    :param cells: Dictionary to populate with tiles
    :param persist: Dataset -> SomeThing mapping, defaults to keeping dataset id only

    The ``cells`` dictionary is a mapping from (x,y) tile index to object with the
    following properties:

     .idx     - tile index (x,y)
     .geobox  - tile geobox
     .utc_offset - timedelta to add to timestamp to get day component in local time
     .dss     - list of UUIDs, or results of ``persist(dataset)`` if custom ``persist`` is supplied
    """
    geobox_cache = {}

    def default_persist(ds):
        return ds.id

    def register(tile, geobox, val):
        cell = cells.get(tile)
        if cell is None:
            utc_ofset = solar_offset(geobox.extent)
            cells[tile] = SimpleNamespace(
                geobox=geobox, idx=tile, utc_offset=utc_ofset, dss=[val]
            )
        else:
            cell.dss.append(val)

    if persist is None:
        persist = default_persist

    for ds in dss:
        ds_val = persist(ds)

        if ds.extent is None:
            warn(f"Dataset without extent info: {str(ds.id)}")
            continue

        for tile, geobox in gridspec.tiles_from_geopolygon(
            ds.extent, geobox_cache=geobox_cache
        ):
            register(tile, geobox, ds_val)

        yield ds


def bin_dataset_stream2(gridspec, dss, geobox_cache=None):
    """
    For every input dataset compute tiles of the GridSpec it overlaps with.

    Iterable[Dataset] -> Iterator[(Dataset, List[Tuple[int, int]])]
    """
    if geobox_cache is None:
        geobox_cache = {}

    for ds in dss:
        if ds.extent is None:
            warn(f"Dataset without extent info: {ds.id}")
            tiles = []
        else:
            tiles = [
                tile
                for tile, _ in gridspec.tiles_from_geopolygon(
                    ds.extent, geobox_cache=geobox_cache
                )
            ]

        yield ds, tiles


def all_datasets(
    dc: Datacube, product: str, read_chunk: int = 1000, limit: Optional[int] = None
):
    """
    Properly lazy version of ``dc.find_datasets_lazy(product=product)``.

    Uses db cursors to reduce latency of results arriving from the database, original
    method in datacube fetches entire query result from the database and only then lazily
    constructs Dataset objects from the returned table.

    :param dc: Datacube object
    :param product: Product name to extract
    :param read_chunk: Number of datasets per page
    :param limit: Optionally constrain query to just return that many dataset
    """
    assert isinstance(limit, (int, type(None)))

    db = psycopg2.connect(str(dc.index.url))
    _limit = "" if limit is None else f"LIMIT {limit}"

    _product = dc.index.products.get_by_name(product)
    if _product is None:
        raise ValueError(f"No such product: {product}")

    query = f"""select
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
    cursor_name = f"c{randint(0, 0xFFFF):04X}"
    with db.cursor(name=cursor_name) as cursor:
        cursor.execute(query, dict(product=product))

        while True:
            chunk = cursor.fetchmany(read_chunk)
            if not chunk:
                break
            for (ds,) in chunk:
                yield Dataset(_product, ds["metadata"], ds["uris"])


def mid_longitude(geom: Geometry) -> float:
    """Return longitude of the middle point of the geomtry."""
    ((lon,), _) = geom.centroid.to_crs("epsg:4326").xy
    return lon


def solar_offset(geom: Geometry, precision: str = "h") -> timedelta:
    """
    Given a geometry compute offset to add to UTC timestamp to get solar day right.

    This only work when geometry is "local enough".
    :param precision: one of ``'h'`` or ``'s'``, defaults to hour precision
    """
    lon = mid_longitude(geom)

    if precision == "h":
        return timedelta(hours=int(lon * 24 / 360 + 0.5))

    # 240 == (24*60*60)/360 (seconds of a day per degree of longitude)
    return timedelta(seconds=int(lon * 240))
