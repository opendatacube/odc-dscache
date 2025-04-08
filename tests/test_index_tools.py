import datetime

import pytest
from odc.geo.geom import box as geom_box
from odc.dscache.tools._index import (
    mid_longitude,
    month_range,
    season_range,
    solar_offset,
    time_range,
)


@pytest.mark.parametrize("lon,lat", [(0, 10), (100, -10), (-120, 30)])
def test_mid_lon(lon, lat):
    r = 0.1
    rect = geom_box(lon - r, lat - r, lon + r, lat + r, "epsg:4326")
    assert rect.centroid.coords[0] == pytest.approx((lon, lat))

    assert mid_longitude(rect) == pytest.approx(lon)
    assert mid_longitude(rect.to_crs("epsg:3857")) == pytest.approx(lon)

    offset = solar_offset(rect, "h")
    assert offset.seconds % (60 * 60) == 0

    offset_sec = solar_offset(rect, "s")
    assert abs((offset - offset_sec).seconds) <= 60 * 60


def test_month_range():
    m1, m2 = month_range(2019, 1, 3)
    assert m1.year == 2019
    assert m2.year == 2019
    assert m1.month == 1 and m2.month == 3

    m1, m2 = month_range(2019, 12, 3)
    assert m1 == datetime.datetime(2019, 12, 1)
    assert m2 == datetime.datetime(2020, 2, 29, 23, 59, 59, 999999)

    assert month_range(2018, 12, 4) == month_range(2019, -1, 4)

    assert season_range(2019, "djf") == month_range(2019, -1, 3)
    assert season_range(2019, "mam") == month_range(2019, 3, 3)
    assert season_range(2002, "jja") == month_range(2002, 6, 3)
    assert season_range(2000, "son") == month_range(2000, 9, 3)

    with pytest.raises(ValueError):
        season_range(2000, "ham")


@pytest.mark.parametrize(
    "t1, t2",
    [
        ("2020-01-17", "2020-03-19"),
        ("2020-04-17", "2020-04-19"),
        ("2020-02-01T13:08:01", "2020-09-03T00:33:46.103"),
        ("2000-01-23", "2001-01-19"),
    ],
)
def test_time_range(t1, t2):
    t1, t2 = map(datetime.datetime.fromisoformat, (t1, t2))
    tt = list(time_range(t1, t2))

    assert tt[0][0] == t1
    assert tt[-1][-1] == t2

    if len(tt) == 1:
        assert tt[0] == (t1, t2)

    t_prev = tt[0][-1]
    for t1, t2 in tt[1:]:
        assert t1 > t_prev
        assert (t1 - t_prev).seconds < 1
        t_prev = t2
