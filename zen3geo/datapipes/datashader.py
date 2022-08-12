"""
DataPipes for :doc:`datashader <datashader:index>`.
"""
from typing import Any, Dict, Iterator, Optional, Union

try:
    import datashader
except ImportError:
    datashader = None
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("canvas_from_xarray")
class XarrayCanvasIterDataPipe(IterDataPipe[Union[xr.DataArray, xr.Dataset]]):
    """
    Takes an :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
    and creates a blank :py:class:`datashader.Canvas` based on the spatial
    extent and coordinates of the input (functional name:
    ``canvas_from_xarray``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[xr.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` objects. These data objects need to have
        both a ``.rio.x_dim`` and ``.rio.y_dim`` attribute, which is present
        if the original dataset was opened using
        :py:func:`rioxarray.open_rasterio`, or by setting it manually using
        :py:meth:`rioxarray.rioxarray.XRasterBase.set_spatial_dims`.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`datashader.Canvas`.

    Yields
    ------
    canvas : datashader.Canvas
        A :py:class:`datashader.Canvas` object representing the same spatial
        extent and x/y coordinates of the input raster image.

    Raises
    ------
    ModuleNotFoundError
        If ``datashader`` is not installed. Follow
        :doc:`install instructions for datashader <datashader:getting_started/index>`
        before using this class.

    Example
    -------
    >>> import pytest
    >>> import numpy as np
    >>> import xarray as xr
    >>> datashader = pytest.importorskip("datashader")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import XarrayCanvas
    ...
    >>> # Create blank canvas from xarray.DataArray using DataPipe
    >>> y = np.arange(0, -3, step=-1)
    >>> x = np.arange(0, 6)
    >>> dataarray: xr.DataArray = xr.DataArray(
    ...     data=np.zeros(shape=(1, 3, 6)),
    ...     coords=dict(band=[1], y=y, x=x),
    ... )
    >>> dataarray = dataarray.rio.set_spatial_dims(x_dim="x", y_dim="y")
    >>> dp = IterableWrapper(iterable=[dataarray])
    >>> dp_canvas = dp.canvas_from_xarray()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_canvas)
    >>> canvas = next(it)
    >>> print(canvas.raster(source=dataarray))
    <xarray.DataArray (band: 1, y: 3, x: 6)>
    array([[[0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]]])
    Coordinates:
      * x        (x) int64 0 1 2 3 4 5
      * y        (y) int64 0 -1 -2
      * band     (band) int64 1
    ...
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Union[xr.DataArray, xr.Dataset]],
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if datashader is None:
            raise ModuleNotFoundError(
                "Package `datashader` is required to be installed to use this datapipe. "
                "Please use `pip install datashader` or "
                "`conda install -c conda-forge datashader` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe[
            Union[xr.DataArray, xr.Dataset]
        ] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:
        for dataarray in self.source_datapipe:
            x_dim: str = dataarray.rio.x_dim
            y_dim: str = dataarray.rio.y_dim
            plot_width: int = len(dataarray[x_dim])
            plot_height: int = len(dataarray[y_dim])
            xmin, ymin, xmax, ymax = dataarray.rio.bounds()

            canvas = datashader.Canvas(
                plot_width=plot_width,
                plot_height=plot_height,
                x_range=(xmin, xmax),
                y_range=(ymin, ymax),
                **self.kwargs
            )
            yield canvas

    def __len__(self) -> int:
        return len(self.source_datapipe)
