"""
Tests for rasterio datapipes.
"""
import numpy as np
import pytest
import rioxarray
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import RasterioRasterizer


# %%
def test_rasterio_rasterizer_geoseries_to_ndarray():
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


def test_rasterio_rasterizer_geodataframe_to_dataarray():
    """
    Ensure that RasterioRasterizer works to rasterize a geopandas.GeoDataFrame
    object into an xarray.DataArray image.
    """
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")

    # Vector data
    geodataframe = gpd.GeoDataFrame(
        data={
            "geometry": [
                shapely.geometry.box(minx=-20, miny=30, maxx=0, maxy=60),
                shapely.geometry.box(minx=0, miny=0, maxx=20, maxy=30),
                shapely.geometry.box(minx=20, miny=-30, maxx=40, maxy=0),
            ]
        },
        crs="EPSG:4326",
    )
    geodataframe = geodataframe.to_crs(epsg=2193)  # New Zealand Transverse Mercator
    dp_geodataframe = IterableWrapper(iterable=[geodataframe])

    # Raster data
    dataarray_day = rioxarray.open_rasterio(
        filename="https://github.com/GenericMappingTools/gmtserver-admin/raw/master/cache/earth_day_HD.tif",
    )
    dataarray_night = rioxarray.open_rasterio(
        filename="https://github.com/GenericMappingTools/gmtserver-admin/raw/master/cache/earth_night_HD.tif",
    )
    dp_dataarray = IterableWrapper(iterable=[dataarray_day, dataarray_night])

    # Using class constructors
    dp_rasterio = RasterioRasterizer(source_datapipe=dp_geodataframe, out=dp_dataarray)
    # Using functional form (recommended)
    dp_rasterio = dp_geodataframe.rasterize_with_rasterio(out=dp_dataarray)

    # Iterate over DataPipe
    assert len(dp_rasterio) == 2
    it = iter(dp_rasterio)
    _ = next(it)  # 1st iteration
    dataarray = next(it)  # 2nd iteration

    assert dataarray.shape == (1, 960, 1920)
    assert dataarray.dims == ("band", "y", "x")
    assert dataarray.dtype == np.uint8
    assert dataarray.rio.crs == 4326
    assert dataarray.rio.transform() == dataarray_night.rio.transform()
    assert dataarray.sum() == 51200
    assert dataarray.isin(0).sum() == 1792000
    assert (~dataarray.isin(test_elements=[0, 1])).sum() == 0
