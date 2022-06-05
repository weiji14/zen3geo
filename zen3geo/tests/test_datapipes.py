"""
Tests for datapipes.
"""
from torchdata.datapipes.iter import IterableWrapper

from zen3geo import RioXarrayReader


# %%
def test_rioxarray_reader():
    """
    Ensure that RioXarrayReader works to read in a GeoTIFF file.
    """
    file_url: str = "https://github.com/GenericMappingTools/gmtserver-admin/raw/master/cache/earth_day_HD.tif"
    dp = IterableWrapper(iterable=[file_url])

    # Using class constructors
    dp_rioxarray = RioXarrayReader(source_datapipe=dp)
    # Using functional form (recommended)
    dp_rioxarray = dp.read_from_rioxarray()

    it = iter(dp_rioxarray)
    filename, dataarray = next(it)

    assert isinstance(filename, str)
    assert dataarray.shape == (1, 960, 1920)
    assert dataarray.dims == ("band", "y", "x")