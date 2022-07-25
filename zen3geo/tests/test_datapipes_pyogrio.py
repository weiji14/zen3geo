"""
Tests for pyogrio datapipes.
"""
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import PyogrioReader

pyogrio = pytest.importorskip("pyogrio")

# %%
def test_pyogrio_reader():
    """
    Ensure that PyogrioReader works to read in a GeoTIFF file and outputs a
    tuple made up of a filename and an xarray.DataArray object.
    """
    file_url: str = "https://github.com/geopandas/pyogrio/raw/v0.4.0/pyogrio/tests/fixtures/test_gpkg_nulls.gpkg"
    dp = IterableWrapper(iterable=[file_url])

    # Using class constructors
    dp_pyogrio = PyogrioReader(source_datapipe=dp)
    # Using functional form (recommended)
    dp_pyogrio = dp.read_from_pyogrio()

    assert len(dp_pyogrio) == 1
    it = iter(dp_pyogrio)
    geodataframe = next(it)

    assert geodataframe.shape == (4, 12)
    assert any(geodataframe.isna())
    assert all(geodataframe.geom_type == "Point")
