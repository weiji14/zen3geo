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
@pytest.fixture(scope="function", name="canvas")
def fixture_canvas():
    """
    The blank datashader.Canvas to use in the tests.
    """
    canvas = datashader.Canvas(
        plot_width=14, plot_height=10, x_range=(1, 8), y_range=(0, 5)
    )
    canvas.crs = "EPSG:4326"
    return canvas


@pytest.fixture(scope="module", name="geodataframe")
def fixture_geodataframe():
    """
    A geopandas.GeoDataFrame containing a collection of shapely.geometry
    objects to use in the tests.
    """
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")

    geometries: list = [
        shapely.geometry.MultiPoint([(4.5, 4.5), (3.5, 1), (6, 3.5)]),
        shapely.geometry.LineString([(3, 5), (5, 3), (3, 2), (5, 0)]),
        shapely.geometry.Polygon([(6, 5), (3.5, 2.5), (6, 0), (6, 2.5), (5, 2.5)]),
    ]
    geodataframe = gpd.GeoDataFrame(data={"geometry": geometries})
    geodataframe = geodataframe.set_crs(epsg=4326)

    return geodataframe


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
    assert hasattr(canvas, "crs")
    assert hasattr(canvas, "raster")


@pytest.mark.parametrize(
    ("geom_type", "sum_val"), [("Point", 3), ("Line", 13), ("Polygon", 15)]
)
def test_datashader_rasterize_vector_geometry(canvas, geodataframe, geom_type, sum_val):
    """
    Ensure that DatashaderRasterizer works to rasterize a
    geopandas.GeoDataFrame of point, line or polygon type into an
    xarray.DataArray grid.
    """
    dp = IterableWrapper(iterable=[canvas])

    vector = geodataframe[geodataframe.type.str.contains(geom_type)]
    dp_vector = IterableWrapper(iterable=[vector])

    # Using class constructors
    dp_datashader = DatashaderRasterizer(source_datapipe=dp, vector_datapipe=dp_vector)
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


def test_datashader_rasterize_canvas_missing_crs(canvas, geodataframe):
    """
    Ensure that DatashaderRasterizer raises an AttributeError when the
    input datashader.Canvas has no crs attribute.
    """
    canvas.crs = None
    dp_canvas = IterableWrapper(iterable=[canvas])
    dp_vector = IterableWrapper(iterable=[geodataframe.geometry])
    dp_datashader = dp_canvas.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader) == 1
    it = iter(dp_datashader)
    with pytest.raises(
        AttributeError, match="Missing crs information for datashader.Canvas"
    ):
        raster = next(it)


def test_datashader_rasterize_vector_missing_crs(canvas, geodataframe):
    """
    Ensure that DatashaderRasterizer raises an AttributeError when the
    input geopandas.GeoSeries has no crs attribute.
    """
    vector = geodataframe.geometry
    vector.crs = None
    dp_canvas = IterableWrapper(iterable=[canvas])
    dp_vector = IterableWrapper(iterable=[vector])
    dp_datashader = dp_canvas.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader) == 1
    it = iter(dp_datashader)
    with pytest.raises(AttributeError, match="Missing crs information for input"):
        raster = next(it)


def test_datashader_rasterize_unmatched_lengths(canvas, geodataframe):
    """
    Ensure that DatashaderRasterizer raises a ValueError when the length of the
    canvas datapipe is unmatched with the length of the vector datapipe.
    """
    # Canvas:Vector ratio of 3:2
    dp_canvas = IterableWrapper(iterable=[canvas, canvas, canvas])
    dp_vector = IterableWrapper(iterable=[geodataframe, geodataframe])

    with pytest.raises(ValueError, match="Unmatched lengths for the"):
        dp_datashader = dp_canvas.rasterize_with_datashader(vector_datapipe=dp_vector)


def test_datashader_rasterize_vector_geometrycollection(canvas, geodataframe):
    """
    Ensure that DatashaderRasterizer raises a NotImplementedError when an
    unsupported vector type like GeometryCollection is used.
    """
    gpd = pytest.importorskip("geopandas")

    # Merge points, lines and polygons into a single GeometryCollection
    geocollection = gpd.GeoSeries(data=geodataframe.unary_union)
    geocollection = geocollection.set_crs(epsg=4326)

    dp = IterableWrapper(iterable=[canvas])
    dp_vector = IterableWrapper(iterable=[geocollection])
    dp_datashader = dp.rasterize_with_datashader(vector_datapipe=dp_vector)

    assert len(dp_datashader) == 1
    it = iter(dp_datashader)
    with pytest.raises(NotImplementedError, match="Unsupported geometry type"):
        raster = next(it)
