"""
DataPipes for :doc:`stackstac <stackstac:index>`.
"""
from typing import Any, Dict, Iterator, Optional

import xarray as xr

try:
    import stackstac
except ImportError:
    stackstac = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("mosaic_dataarray")
class StackSTACMosaickerIterDataPipe(IterDataPipe[xr.DataArray]):
    """
    Takes :py:class:`xarray.DataArray` objects, flattens a dimension by picking
    the first valid pixel, to yield mosaicked :py:class:`xarray.DataArray`
    objects (functional name: ``mosaic_dataarray``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[xarray.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` objects, with
        e.g. dimensions ("time", "band", "y", "x").

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`stackstac.mosaic`.

    Yields
    ------
    dataarray : xarray.DataArray
        An :py:class:`xarray.DataArray` that has been mosaicked with e.g.
        dimensions ("band", "y", "x").

    Raises
    ------
    ModuleNotFoundError
        If ``stackstac`` is not installed. See
        :doc:`install instructions for stackstac <stackstac:index>`, (e.g. via
        ``pip install stackstac``) before using this class.

    Example
    -------
    >>> import pytest
    >>> import xarray as xr
    >>> pystac = pytest.importorskip("pystac")
    >>> stackstac = pytest.importorskip("stackstac")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import StackSTACMosaicker
    ...
    >>> # Get list of ALOS DEM tiles to mosaic together later
    >>> item_urls = [
    ...     "https://planetarycomputer.microsoft.com/api/stac/v1/collections/alos-dem/items/ALPSMLC30_N022E113_DSM",
    ...     "https://planetarycomputer.microsoft.com/api/stac/v1/collections/alos-dem/items/ALPSMLC30_N022E114_DSM",
    ... ]
    >>> stac_items = [pystac.Item.from_file(href=url) for url in item_urls]
    >>> dataarray = stackstac.stack(items=stac_items)
    >>> assert dataarray.sizes == {'time': 2, 'band': 1, 'y': 3600, 'x': 7200}
    ...
    >>> # Mosaic different tiles in an xarray.DataArray using DataPipe
    >>> dp = IterableWrapper(iterable=[dataarray])
    >>> dp_mosaic = dp.mosaic_dataarray()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_mosaic)
    >>> dataarray = next(it)
    >>> print(dataarray.sizes)
    Frozen({'band': 1, 'y': 3600, 'x': 7200})
    >>> print(dataarray.coords)
    Coordinates:
      * band         (band) <U4 'data'
      * x            (x) float64 113.0 113.0 113.0 113.0 ... 115.0 115.0 115.0 115.0
      * y            (y) float64 23.0 23.0 23.0 23.0 23.0 ... 22.0 22.0 22.0 22.0
    ...
    >>> print(dataarray.attrs["spec"])
    RasterSpec(epsg=4326, bounds=(113.0, 22.0, 115.0, 23.0), resolutions_xy=(0.0002777777777777778, 0.0002777777777777778))
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[xr.DataArray],
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if stackstac is None:
            raise ModuleNotFoundError(
                "Package `stackstac` is required to be installed to use this datapipe. "
                "Please use `pip install stackstac` or "
                "`conda install -c conda-forge stackstac` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[xr.DataArray]:
        for dataarray in self.source_datapipe:
            yield stackstac.mosaic(arr=dataarray, **self.kwargs)

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("stack_stac_items")
class StackSTACStackerIterDataPipe(IterDataPipe[xr.DataArray]):
    """
    Takes :py:class:`pystac.Item` objects, reprojects them to the same grid
    and stacks them along time, to yield :py:class:`xarray.DataArray` objects
    (functional name: ``stack_stac_items``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[pystac.Item]
        A DataPipe that contains :py:class:`pystac.Item` objects.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`stackstac.stack`.

    Yields
    ------
    datacube : xarray.DataArray
        An :py:class:`xarray.DataArray` backed by a
        :py:class:`dask.array.Array` containing the time-series datacube. The
        dimensions will be ("time", "band", "y", "x").

    Raises
    ------
    ModuleNotFoundError
        If ``stackstac`` is not installed. See
        :doc:`install instructions for stackstac <stackstac:index>`, (e.g. via
        ``pip install stackstac``) before using this class.

    Example
    -------
    >>> import pytest
    >>> pystac = pytest.importorskip("pystac")
    >>> stacstac = pytest.importorskip("stackstac")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import StackSTACStacker
    ...
    >>> # Stack different bands in a STAC Item using DataPipe
    >>> item_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20220914T093226_20220914T093252_044999_056053"
    >>> stac_item = pystac.Item.from_file(href=item_url)
    >>> dp = IterableWrapper(iterable=[stac_item])
    >>> dp_stackstac = dp.stack_stac_items(
    ...     assets=["vh", "vv"], epsg=32652, resolution=10
    ... )
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_stackstac)
    >>> dataarray = next(it)
    >>> print(dataarray.sizes)
    Frozen({'time': 1, 'band': 2, 'y': 20686, 'x': 28043})
    >>> print(dataarray.coords)
    Coordinates:
      * time                                   (time) datetime64[ns] 2022-09-14T0...
        id                                     (time) <U62 'S1A_IW_GRDH_1SDV_2022...
      * band                                   (band) <U2 'vh' 'vv'
      * x                                      (x) float64 1.354e+05 ... 4.158e+05
      * y                                      (y) float64 4.305e+06 ... 4.098e+06
    ...
    >>> print(dataarray.attrs["spec"])
    RasterSpec(epsg=32652, bounds=(135370, 4098080, 415800, 4304940), resolutions_xy=(10, 10))
    """

    def __init__(
        self, source_datapipe: IterDataPipe, **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if stackstac is None:
            raise ModuleNotFoundError(
                "Package `stackstac` is required to be installed to use this datapipe. "
                "Please use `pip install stackstac` or "
                "`conda install -c conda-forge stackstac` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[xr.DataArray]:
        for stac_items in self.source_datapipe:
            yield stackstac.stack(items=stac_items, **self.kwargs)

    def __len__(self) -> int:
        return len(self.source_datapipe)
