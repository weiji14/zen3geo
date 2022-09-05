"""
Tests for geopandas datapipes.
"""
import numpy as np
import pytest
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import GeoPandasRectangleClipper

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

# %%
@pytest.fixture(scope="module", name="geodataframe")
def fixture_geodataframe():
    """
    A geopandas.GeoDataFrame containing a collection of shapely.geometry
    objects to use in the tests.
    """
    geometries: list = [
        shapely.geometry.box(minx=0.0, miny=0.0, maxx=2.0, maxy=2.0),
        shapely.geometry.box(minx=2.0, miny=2.0, maxx=4.0, maxy=4.0),
    ]
    geodataframe = gpd.GeoDataFrame(data={"geometry": geometries})
    geodataframe = geodataframe.set_crs(crs="OGC:CRS84")

    return geodataframe


@pytest.fixture(scope="function", name="dataset")
def fixture_dataset():
    """
    The sample xarray.Dataset to use in the tests.
    """
    dataarray = xr.DataArray(
        data=np.ones(shape=(1, 5, 7)),
        coords=dict(
            band=[0],
            y=np.linspace(start=4.0, stop=0.0, num=5),
            x=np.linspace(start=-1.0, stop=5, num=7),
        ),
        dims=("band", "y", "x"),
        name="foo",
    )
    dataset: xr.Dataset = dataarray.to_dataset()
    dataset: xr.Dataset = dataset.rio.write_crs(input_crs="OGC:CRS84")

    return dataset


# %%
def test_geopandas_rectangle_clipper_geoseries_dataset(geodataframe, dataset):
    """
    Ensure that GeoPandasRectangleClipper works to clip a geopandas.GeoSeries
    vector with xarray.Dataset rasters and outputs a tuple made up of a
    spatially subsetted geopandas.GeoSeries and an xarray.Dataset raster mask.
    """
    dp_vector = IterableWrapper(iterable=[geodataframe.geometry])
    dp_raster = IterableWrapper(
        iterable=[
            dataset.rio.clip_box(minx=-1, miny=0, maxx=1, maxy=1),
            dataset.rio.clip_box(minx=3, miny=3, maxx=5, maxy=4),
        ]
    )

    # Using class constructors
    dp_clipped = GeoPandasRectangleClipper(
        source_datapipe=dp_vector, mask_datapipe=dp_raster
    )
    # Using functional form (recommended)
    dp_clipped = dp_vector.clip_vector_with_rectangle(mask_datapipe=dp_raster)

    assert len(dp_clipped) == 2
    it = iter(dp_clipped)

    clipped_geoseries, raster_chip = next(it)
    assert clipped_geoseries.crs == "OGC:CRS84"
    assert all(clipped_geoseries.geom_type == "Polygon")
    assert clipped_geoseries.shape == (1,)
    assert clipped_geoseries[0].bounds == (0.0, 0.0, 1.5, 1.5)
    assert raster_chip.dims == {"band": 1, "y": 2, "x": 3}
    assert raster_chip.rio.bounds() == (-1.5, -0.5, 1.5, 1.5)

    clipped_geoseries, raster_chip = next(it)
    assert clipped_geoseries.shape == (1,)
    assert clipped_geoseries[1].bounds == (2.5, 2.5, 4.0, 4.0)
    assert raster_chip.dims == {"band": 1, "y": 2, "x": 3}
    assert raster_chip.rio.bounds() == (2.5, 2.5, 5.5, 4.5)
    assert raster_chip.rio.crs == "OGC:CRS84"


def test_geopandas_rectangle_clipper_different_crs(geodataframe, dataset):
    """
    Ensure that GeoPandasRectangleClipper works to clip a geopandas.GeoSeries
    vector with xarray.Dataset rasters which have different coordinate
    reference systems, and outputs a tuple made up of a spatially subsetted
    geopandas.GeoSeries and an xarray.Dataset raster mask that both have the
    same coordinate reference system.
    """
    dp_vector = IterableWrapper(iterable=[geodataframe.geometry])

    dataset_3857 = dataset.rio.clip_box(minx=-1, miny=0, maxx=1, maxy=1).rio.reproject(
        "EPSG:3857"
    )
    dataset_32631 = dataset.rio.clip_box(minx=3, miny=3, maxx=5, maxy=4).rio.reproject(
        "EPSG:32631"
    )
    dp_raster = IterableWrapper(iterable=[dataset_3857, dataset_32631])

    # Using class constructors
    dp_clipped = GeoPandasRectangleClipper(
        source_datapipe=dp_vector, mask_datapipe=dp_raster
    )
    # Using functional form (recommended)
    dp_clipped = dp_vector.clip_vector_with_rectangle(mask_datapipe=dp_raster)

    assert len(dp_clipped) == 2
    it = iter(dp_clipped)

    clipped_geoseries, raster_chip = next(it)
    assert clipped_geoseries.crs == "EPSG:3857"
    assert all(clipped_geoseries.geom_type == "Polygon")
    assert clipped_geoseries.shape == (1,)
    assert clipped_geoseries[0].bounds == (
        0.0,
        0.0,
        166988.3675623712,
        166998.31375292226,
    )
    assert raster_chip.dims == {"band": 1, "y": 2, "x": 3}
    assert raster_chip.rio.bounds() == (
        -166979.23618991036,
        -55646.75541526544,
        166988.3675623712,
        166998.31375292226,
    )
    assert raster_chip.rio.crs == "EPSG:3857"

    clipped_geoseries, raster_chip = next(it)
    assert clipped_geoseries.crs == "EPSG:32631"
    assert clipped_geoseries.shape == (1,)
    assert clipped_geoseries[1].bounds == (
        444414.4114896285,
        276009.81064532325,
        611163.137304327,
        442194.9725083875,
    )
    assert raster_chip.dims == {"band": 1, "y": 2, "x": 3}
    assert raster_chip.rio.bounds() == (
        444414.4114896285,
        276009.81064532325,
        777205.5384580799,
        497870.56195762416,
    )
    assert raster_chip.rio.crs == "EPSG:32631"


def test_geopandas_rectangle_clipper_incorrect_length(geodataframe, dataset):
    """
    Ensure that GeoPandasRectangleClipper raises a NotImplementedError when the
    length of the vector datapipe is not equal to 1.
    """
    dp_vector = IterableWrapper(iterable=[geodataframe, geodataframe])
    dp_raster = IterableWrapper(iterable=[dataset, dataset, dataset])

    with pytest.raises(NotImplementedError, match="The vector datapipe's length can"):
        dp_clipped = dp_vector.clip_vector_with_rectangle(mask_datapipe=dp_raster)
