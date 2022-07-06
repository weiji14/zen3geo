"""
Tests for xbatcher datapipes.
"""
import numpy as np
import pytest
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import XbatcherSlicer

xbatcher = pytest.importorskip("xbatcher")


# %%
def test_xbatcher_slicer_dataarray():
    """
    Ensure that XbatcherSlicer works to slice an xarray.DataArray object and
    outputs a smaller xarray.DataArray chip.
    """

    dataarray: xr.DataArray = xr.DataArray(
        data=np.ones(shape=(3, 128, 128)), dims=["band", "y", "x"]
    ).chunk({"band": 1})
    dp = IterableWrapper(iterable=[dataarray])

    # Using class constructors
    dp_xbatcher = XbatcherSlicer(source_datapipe=dp, input_dims={"y": 64, "x": 64})
    # Using functional form (recommended)
    dp_xbatcher = dp.slice_with_xbatcher(input_dims={"y": 64, "x": 64})

    it = iter(dp_xbatcher)
    dataarray_chip = next(it)

    assert dataarray_chip.sizes == {"band": 3, "y": 64, "x": 64}
    assert dataarray_chip.sum() == 3 * 64 * 64


def test_xbatcher_slicer_dataset():
    """
    Ensure that XbatcherSlicer works to slice an xarray.Dataset object and
    outputs a smaller xarray.Dataset chip.
    """

    dataset: xr.Dataset = xr.Dataset(
        data_vars={"temperature": (["x", "y"], 15 * np.ones(shape=(32, 32)))},
        coords={
            "lon": (["x"], np.linspace(start=0, stop=32, num=32)),
            "lat": (["y"], np.linspace(start=64, stop=32, num=32)),
        },
    )
    dp = IterableWrapper(iterable=[dataset])

    # Using class constructors
    dp_xbatcher = XbatcherSlicer(source_datapipe=dp, input_dims={"y": 16, "x": 16})
    # Using functional form (recommended)
    dp_xbatcher = dp.slice_with_xbatcher(input_dims={"y": 16, "x": 16})

    it = iter(dp_xbatcher)
    dataset_chip = next(it)

    assert dataset_chip.temperature.sizes == {"y": 16, "x": 16}
    assert dataset_chip.temperature.sum() == 15 * 16 * 16
