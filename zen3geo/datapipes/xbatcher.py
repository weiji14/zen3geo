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
        Extra keyword arguments to pass to :py:func:`xbatcher.BatchGenerator`.

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
    ...     data=np.ones(shape=(3, 128, 128)),
    ...     name="foo",
    ...     dims=["band", "y", "x"]
    ... )
    >>> dp = IterableWrapper(iterable=[dataarray])
    >>> dp_xbatcher = dp.slice_with_xbatcher(input_dims={"y": 64, "x": 64})
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_xbatcher)
    >>> dataarray_chip = next(it)
    >>> dataarray_chip
    <xarray.Dataset>
    Dimensions:  (band: 3, y: 64, x: 64)
    Dimensions without coordinates: band, y, x
    Data variables:
        foo      (band, y, x) float64 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0 1.0
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Union[xr.DataArray, xr.Dataset]],
        input_dims: Dict[Hashable, int],
        **kwargs: Optional[Dict[str, Any]]
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
            if hasattr(dataarray, "name") and dataarray.name is None:
                # Workaround for ValueError: unable to convert unnamed
                # DataArray to a Dataset without providing an explicit name
                dataarray = dataarray.to_dataset(
                    name=xr.backends.api.DATAARRAY_VARIABLE
                )[xr.backends.api.DATAARRAY_VARIABLE]
                # dataarray.name = "z"  # doesn't work for some reason
            for chip in dataarray.batch.generator(
                input_dims=self.input_dims, **self.kwargs
            ):
                yield chip

    # def __len__(self) -> int:
    #     return len(self.source_datapipe)
