"""
DataPipes for :doc:`xbatcher <xbatcher:index>`.
"""
from typing import Any, Dict, Hashable, Iterator, Optional, Tuple, Union

import xarray as xr

try:
    import xbatcher
except ImportError:
    xbatcher = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("slice_with_xbatcher")
class XbatcherSlicerIterDataPipe(IterDataPipe[Union[xr.DataArray, xr.Dataset]]):
    """
    Takes an :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
    and creates a sliced window view (also known as a chip or tile) of the
    n-dimensional array (functional name: ``slice_with_xbatcher``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[xarray.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` objects.

    input_dims : dict
        A dictionary specifying the size of the inputs in each dimension to
        slice along, e.g. ``{'lon': 64, 'lat': 64}``. These are the dimensions
        the machine learning library will see. All other dimensions will be
        stacked into one dimension called ``batch``.

    kwargs : Optional
        Extra keyword arguments to pass to :py:class:`xbatcher.BatchGenerator`.

    Yields
    ------
    chip : xarray.DataArray
        An :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset` object
        containing the sliced raster data, with the size/shape defined by the
        ``input_dims`` parameter.

    Raises
    ------
    ModuleNotFoundError
        If ``xbatcher`` is not installed. Follow
        :doc:`install instructions for xbatcher <xbatcher:index>`
        before using this class.

    Example
    -------
    >>> import pytest
    >>> import numpy as np
    >>> import xarray as xr
    >>> xbatcher = pytest.importorskip("xbatcher")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import XbatcherSlicer
    ...
    >>> # Sliced window view of xarray.DataArray using DataPipe
    >>> dataarray: xr.DataArray = xr.DataArray(
    ...     data=np.ones(shape=(3, 64, 64)),
    ...     name="foo",
    ...     dims=["band", "y", "x"]
    ... )
    >>> dp = IterableWrapper(iterable=[dataarray])
    >>> dp_xbatcher = dp.slice_with_xbatcher(input_dims={"y": 2, "x": 2})
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_xbatcher)
    >>> dataarray_chip = next(it)
    >>> dataarray_chip
    <xarray.DataArray 'foo' (band: 3, y: 2, x: 2)>
    array([[[1., 1.],
            [1., 1.]],
    <BLANKLINE>
           [[1., 1.],
            [1., 1.]],
    <BLANKLINE>
           [[1., 1.],
            [1., 1.]]])
    Dimensions without coordinates: band, y, x
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Union[xr.DataArray, xr.Dataset]],
        input_dims: Dict[Hashable, int],
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if xbatcher is None:
            raise ModuleNotFoundError(
                "Package `xbatcher` is required to be installed to use this datapipe. "
                "Please use `pip install xbatcher` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe[
            Union[xr.DataArray, xr.Dataset]
        ] = source_datapipe
        self.input_dims: Dict[Hashable, int] = input_dims
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Union[xr.DataArray, xr.Dataset]]:
        for dataarray in self.source_datapipe:
            for chip in dataarray.batch.generator(
                input_dims=self.input_dims, **self.kwargs
            ):
                yield chip

    def __len__(self) -> int:
        return sum(
            len(dataarray.batch.generator(input_dims=self.input_dims, **self.kwargs))
            for dataarray in self.source_datapipe
        )
