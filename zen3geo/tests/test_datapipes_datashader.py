"""
Tests for datashader datapipes.
"""
import numpy as np
import pytest
import xarray as xr
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import XarrayCanvas

datashader = pytest.importorskip("datashader")


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
