"""
Tests for pystac datapipes.
"""
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import XpySTACAssetReader


# %%
def test_xpystac_asset_reader_cog():
    """
    Ensure that XpySTACAssetReader works to read in a pystac.Asset object
    stored as a Cloud-Optimized GeoTIFF and output to an xarray.Dataset object.
    """
    pystac = pytest.importorskip("pystac")
    xpystac = pytest.importorskip("xpystac")

    item_url: str = "https://github.com/stac-utils/pystac/raw/v1.7.1/tests/data-files/raster/raster-sentinel2-example.json"
    asset: pystac.Asset = pystac.Item.from_file(href=item_url).assets["overview"]
    assert asset.media_type == pystac.MediaType.COG

    dp = IterableWrapper(iterable=[asset])

    # Using class constructors
    dp_xpystac = XpySTACAssetReader(source_datapipe=dp)
    # Using functional form (recommended)
    dp_xpystac = dp.read_from_xpystac()

    assert len(dp_xpystac) == 1
    it = iter(dp_xpystac)
    dataset = next(it)

    assert dataset.sizes == {"band": 3, "x": 343, "y": 343}
    assert dataset.band_data.dtype == "float32"
    assert dataset.rio.bounds() == (399960.0, 4090240.0, 509720.0, 4200000.0)
    assert dataset.rio.resolution() == (320.0, -320.0)
    assert dataset.rio.crs == "EPSG:32633"


def test_xpystac_asset_reader_zarr():
    """
    Ensure that XpySTACAssetReader works to read in a pystac.Asset object
    stored as a Zarr file and output to an xarray.Dataset object.
    """
    pystac = pytest.importorskip("pystac")
    xpystac = pytest.importorskip("xpystac")

    collection_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/daymet-daily-hi"
    asset: pystac.Asset = pystac.Collection.from_file(href=collection_url).assets[
        "zarr-https"
    ]
    assert asset.media_type == "application/vnd+zarr"

    dp = IterableWrapper(iterable=[asset])

    # Using class constructors
    dp_xpystac = XpySTACAssetReader(source_datapipe=dp)
    # Using functional form (recommended)
    dp_xpystac = dp.read_from_xpystac()

    assert len(dp_xpystac) == 1
    it = iter(dp_xpystac)
    dataset = next(it)

    assert dataset.sizes == {"time": 14965, "y": 584, "x": 284, "nv": 2}
    assert dataset.prcp.dtype == "float32"
    assert dataset.rio.bounds() == (-5802750.0, -622500.0, -5518750.0, -38500.0)
    assert dataset.rio.resolution() == (1000.0, -1000.0)
    assert dataset.rio.grid_mapping == "lambert_conformal_conic"


def test_xpystac_asset_reader_geotiff_without_xpystac():
    """
    Ensure that XpySTACAssetReader works to read in a GeoTIFF file and output
    to an xarray.Dataset object, even when xpystac is not installed.

    Note that `engine="rasterio"` has been removed in xarray v2023.04.0, see
    https://github.com/pydata/xarray/pull/7671. So, this test will need to be
    updated once we change to require an xarray verson greater than 2023.04.0.
    Only included this test to check an alternative to `engine="stac"` that
    did not require installing extra required dependencies like `netcdf4` or
    `h5netcdf`.
    """
    tif_url: str = "https://github.com/corteva/rioxarray/raw/0.14.1/test/test_data/input/cint16.tif"

    dp = IterableWrapper(iterable=[tif_url])

    # Using class constructors
    dp_xpystac = XpySTACAssetReader(source_datapipe=dp, engine="rasterio")
    # Using functional form (recommended)
    dp_xpystac = dp.read_from_xpystac(engine="rasterio")

    assert len(dp_xpystac) == 1
    it = iter(dp_xpystac)
    dataset = next(it)

    assert dataset.sizes == {"band": 1, "x": 100, "y": 100}
    assert dataset.band_data.dtype == "complex64"
    assert dataset.rio.bounds() == (0.0, 100.0, 100.0, 0.0)
    assert dataset.rio.resolution() == (1.0, 1.0)
    assert dataset.rio.crs == "EPSG:4326"
