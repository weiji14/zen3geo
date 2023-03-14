"""
Tests for pystac datapipes.
"""
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import XpySTACAssetReader

pystac = pytest.importorskip("pystac")
xpystac = pytest.importorskip("xpystac")

# %%
def test_xpystac_asset_reader():
    """
    Ensure that XpySTACAssetReader works to read in a pystac.Asset object and
    output to an xarray.Dataset object.
    """
    item_url: str = "https://github.com/stac-utils/pystac/raw/v1.7.0/tests/data-files/raster/raster-sentinel2-example.json"
    asset: pystac.Asset = pystac.Item.from_file(href=item_url).assets["overview"]

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
