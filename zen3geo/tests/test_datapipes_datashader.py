"""
Tests for datashader datapipes.
"""
import numpy as np
import pytest
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import DatashaderRasterizer, XarrayCanvas

datashader = pytest.importorskip("datashader")

# %%
@pytest.fixture(scope="module", name="geometries")
def fixture_geoms():
    """
    Collection of shapely.geometry objects to use in the tests.
    """
    shapely = pytest.importorskip("shapely")

    geometries = shapely.geometry.GeometryCollection(
        geoms=[
            shapely.geometry.MultiPoint([(4.5, 4.5), (3.5, 1), (6, 3.5)]),
            shapely.geometry.LineString([(3, 5), (5, 3), (3, 2), (5, 0)]),
            shapely.geometry.Polygon([(6, 5), (3.5, 2.5), (6, 0), (6, 2.5), (5, 2.5)]),
        ]
    )
    return geometries


# %%
def test_datashader_canvas_dataset():
    """
    Ensure that XarrayCanvas works to create a blank datashader.Canvas object
    from an xarray.Dataset.
    """

    dataset: xr.Dataset = xr.Dataset(
        data_vars={"temperature": (["y", "x"], 15 * np.ones(shape=(12, 8)))},
        coords={
            "y": (["y"], np.linspace(start=6, stop=0, num=12)),
            "x": (["x"], np.linspace(start=0, stop=4, num=8)),
        },
    )
    dp = IterableWrapper(iterable=[dataset])

    # Using class constructors
    dp_canvas = XarrayCanvas(source_datapipe=dp)
    # Using functional form (recommended)
    dp_canvas = dp.canvas_from_xarray()

    assert len(dp_canvas) == 1
    it = iter(dp_canvas)
    canvas = next(it)

    assert canvas.plot_height == 12
    assert canvas.plot_width == 8
    assert hasattr(canvas, "raster")


@pytest.mark.parametrize(
    ("geom_type", "sum_val"), [("Point", 3), ("Line", 13), ("Polygon", 15)]
)
def test_datashader_rasterize_vector_geometry(geometries, geom_type, sum_val):
    """
    Ensure that DatashaderRasterizer works to rasterize a geopandas.GeoSeries
    object of point, line or polygon type into an xarray.DataArray grid.
    """
    gpd = pytest.importorskip("geopandas")

    canvas = datashader.Canvas(
        plot_width=14, plot_height=10, x_range=(1, 8), y_range=(0, 5)
    )
    canvas.crs = "EPSG:4326"
    dp = IterableWrapper(iterable=[canvas])

    geoms = [geom for geom in geometries.geoms if geom_type in geom.type]
    vector = gpd.GeoSeries(data=geoms)
    vector = vector.set_crs(epsg=4326)
    dp_vector = IterableWrapper(iterable=[vector])

    # Using class constructors
    dp_canvas = DatashaderRasterizer(source_datapipe=dp, vector_datapipe=dp_vector)
    # Using functional form (recommended)
    dp_datashader = dp.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader) == 1
    it = iter(dp_datashader)
    dataarray = next(it)

    assert dataarray.data.sum() == sum_val
    assert dataarray.dims == ("y", "x")
    assert dataarray.rio.crs == "EPSG:4326"
    assert dataarray.rio.shape == (10, 14)
    assert dataarray.rio.transform().e == -0.5


def test_datashader_rasterize_missing_crs(geometries):
    """
    Ensure that DatashaderRasterizer raises a ValueError when either the input
    datashader.Canvas or geopandas.GeoDataFrame has no crs attribute.
    """
    gpd = pytest.importorskip("geopandas")

    vector = gpd.GeoDataFrame(data={"geometry": geometries})
    dp_vector = IterableWrapper(iterable=[vector])

    # When datashader.Canvas has no crs
    canvas = datashader.Canvas(
        plot_width=2, plot_height=3, x_range=(0, 2), y_range=(3, 6)
    )
    dp = IterableWrapper(iterable=[canvas])
    dp_datashader = dp.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader) == 1
    it = iter(dp_datashader)
    with pytest.raises(
        AttributeError, match="Missing crs information for datashader.Canvas"
    ):
        raster = next(it)

    # When geopandas.GeoDataFrame has no crs
    canvas.crs = "EPSG:4326"
    dp_canvas2 = IterableWrapper(iterable=[canvas])
    dp_datashader2 = dp_canvas2.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader2) == 1
    it = iter(dp_datashader2)
    with pytest.raises(AttributeError, match="Missing crs information for input"):
        raster = next(it)


def test_datashader_rasterize_vector_geometrycollection(geometries):
    """
    Ensure that DatashaderRasterizer raises a ValueError when an unsupported
    vector type like GeometryCollection is used.
    """
    gpd = pytest.importorskip("geopandas")

    canvas = datashader.Canvas(
        plot_width=10, plot_height=5, x_range=(0, 10), y_range=(0, 5)
    )
    canvas.crs = "EPSG:4326"
    dp = IterableWrapper(iterable=[canvas])

    geocollection = gpd.GeoSeries(data=geometries)
    geocollection = geocollection.set_crs(epsg=4326)
    dp_vector = IterableWrapper(iterable=[geocollection])

    # Using class constructors
    dp_canvas = DatashaderRasterizer(source_datapipe=dp, vector_datapipe=dp_vector)
    # Using functional form (recommended)
    dp_datashader = dp.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader) == 1
    it = iter(dp_datashader)
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        raster = next(it)
