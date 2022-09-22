"""
Tests for stackstac datapipes.
"""
import numpy as np
import pytest
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import StackSTACStacker

pystac = pytest.importorskip("pystac")
stackstac = pytest.importorskip("stackstac")

# %%
def test_stackstac_mosaicker():
    """
    Ensure that StackSTACMosaicker works to mosaic tiles within a 4D
    xarray.DataArray to a 3D xarray.DataArray.
    """
    datacube: xr.DataArray = xr.DataArray(
        data=np.ones(shape=(3, 1, 32, 32)), dims=["tile", "band", "y", "x"]
    )
    dataarray = stackstac.mosaic(arr=datacube, dim="tile")
    assert dataarray.sizes == {"band": 1, "y": 32, "x": 32}
    assert dataarray.sum() == 1 * 32 * 32


def test_stackstac_stacker():
    """
    Ensure that StackSTACStacker works to stack multiple bands within a STAC
    item and outputs an xarray.DataArray object.
    """
    item_url: str = "https://github.com/stac-utils/pystac/raw/v1.6.1/tests/data-files/raster/raster-sentinel2-example.json"
    stac_item = pystac.Item.from_file(href=item_url)
    dp = IterableWrapper(iterable=[stac_item])

    # Using class constructors
    dp_stackstac = StackSTACStacker(source_datapipe=dp, assets=["B02", "B03", "B04"])
    # Using functional form (recommended)
    dp_stackstac = dp.stack_stac_items(assets=["B02", "B03", "B04"])

    assert len(dp_stackstac) == 1
    it = iter(dp_stackstac)
    dataarray = next(it)

    assert dataarray.shape == (1, 3, 10980, 10980)
    assert dataarray.dims == ("time", "band", "y", "x")
    assert dataarray.rio.bounds() == (399955.0, 4090205.0, 509755.0, 4200005.0)
    assert dataarray.rio.resolution() == (10.0, -10.0)
    assert dataarray.rio.crs == "EPSG:32633"
