"""
Tests for rasterio datapipes.
"""
import numpy as np
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import RasterioRasterizer


# %%
def test_rasterio_rasterizer_geoseries():
    """
    Ensure that RasterioRasterizer works to rasterize a geopandas.GeoSeries
    object into a numpy.ndarray image.
    """
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")

    geoseries = gpd.GeoSeries(
        data=[
            shapely.geometry.box(minx=0, miny=1, maxx=2, maxy=3),
            shapely.geometry.box(minx=1, miny=2, maxx=3, maxy=4),
            shapely.geometry.box(minx=2, miny=3, maxx=4, maxy=5),
        ]
    )
    dp = IterableWrapper(iterable=[geoseries])

    # Using class constructors
    dp_rasterio = RasterioRasterizer(source_datapipe=dp, out_shape=(5, 4))
    # Using functional form (recommended)
    dp_rasterio = dp.rasterize_with_rasterio(out_shape=(5, 4))

    assert len(dp_rasterio) == 1
    it = iter(dp_rasterio)
    ndarray = next(it)

    np.testing.assert_allclose(
        actual=ndarray,
        desired=np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )
