"""
DataPipes for `xpystac <https://github.com/jsignell/xpystac>`__.
"""
from typing import Any, Dict, Iterator, Optional

import xarray as xr

try:
    import pystac
    import xpystac
except ImportError:
    pystac = None
    xpystac = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("read_from_xpystac")
class XpySTACAssetReaderIterDataPipe(IterDataPipe[StreamWrapper]):
    """
    Takes a :py:class:`pystac.Asset` object containing raster data (e.g.
    :doc:`Zarr <zarr:index>`,
    `NetCDF <https://www.unidata.ucar.edu/software/netcdf>`,
    `Cloud-Optimized GeoTIFF <https://www.cogeo.org>`__, etc) from local disk
    or URLs (as long as they can be read by xpystac) and yields
    :py:class:`xarray.Dataset` objects (functional name:
    ``read_from_xpystac``).

    Based on
    https://github.com/pytorch/data/blob/v0.5.1/torchdata/datapipes/iter/load/iopath.py#L42-L97

    Parameters
    ----------
    source_datapipe : IterDataPipe[pystac.Asset]
        A DataPipe that contains :py:class:`pystac.Asset` objects to raster
        files such as :doc:`Zarr <zarr:index>`,
        `NetCDF <https://www.unidata.ucar.edu/software/netcdf>`,
        `Cloud-Optimized GeoTIFF <https://www.cogeo.org>`__, etc.

    engine : str or xarray.backends.BackendEntrypoint
        Engine to use when reading files. If not provided, the default engine
        will be the "stac" backend from ``xpystac``. Alternatively, set
        ``engine=None`` to let ``xarray`` choose the default engine based on
        available dependencies, with a preference for "netcdf4". See also
        :py:func:`xarray.open_dataset` for details about other engine options.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`xarray.open_dataset`.

    Yields
    ------
    stream_obj : xarray.Dataset
        A :py:class:`xarray.Dataset` object containing the raster data.

    Raises
    ------
    ModuleNotFoundError
        If ``xpystac`` is not installed. See
        `install instructions for xpystac
        <https://github.com/jsignell/xpystac#install>`__,
        (e.g. via ``pip install xpystac``) before using this class.

    Example
    -------
    >>> import pytest
    >>> pystac = pytest.importorskip("pystac")
    >>> xpystac = pytest.importorskip("xpystac")
    >>> zarr = pytest.importorskip("zarr")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import XpySTACAssetReader
    ...
    >>> # Read in STAC Asset using DataPipe
    >>> collection_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/nasa-nex-gddp-cmip6"
    >>> asset: pystac.Asset = pystac.Collection.from_file(href=collection_url).assets[
    ...     "ACCESS-CM2.historical"
    ... ]
    >>> dp = IterableWrapper(iterable=[asset])
    >>> dp_xpystac = dp.read_from_xpystac()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_xpystac)
    >>> dataset = next(it)
    >>> dataset.sizes
    Frozen({'time': 23741, 'lat': 600, 'lon': 1440})
    >>> print(dataset.data_vars)
    Data variables:
        hurs     (time, lat, lon) float32 ...
        huss     (time, lat, lon) float32 ...
        pr       (time, lat, lon) float32 ...
        rlds     (time, lat, lon) float32 ...
        rsds     (time, lat, lon) float32 ...
        sfcWind  (time, lat, lon) float32 ...
        tas      (time, lat, lon) float32 ...
        tasmax   (time, lat, lon) float32 ...
        tasmin   (time, lat, lon) float32 ...
    >>> dataset.attrs  # doctest: +NORMALIZE_WHITESPACE
    {'Conventions': 'CF-1.7',
     'activity': 'NEX-GDDP-CMIP6',
     'cmip6_institution_id': 'CSIRO-ARCCSS',
     'cmip6_license': 'CC-BY-SA 4.0',
     'cmip6_source_id': 'ACCESS-CM2',
     ...
     'history': '2021-10-04T13:59:21.654137+00:00: install global attributes',
     'institution': 'NASA Earth Exchange, NASA Ames Research Center, ...
     'product': 'output',
     'realm': 'atmos',
     'references': 'BCSD method: Thrasher et al., 2012, ...
     'resolution_id': '0.25 degree',
     'scenario': 'historical',
     'source': 'BCSD',
     'title': 'ACCESS-CM2, r1i1p1f1, historical, global downscaled CMIP6 ...
     'tracking_id': '16d27564-470f-41ea-8077-f4cc3efa5bfe',
     'variant_label': 'r1i1p1f1',
     'version': '1.0'}
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[pystac.Asset],
        engine: str = "stac",
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if xpystac is None:
            raise ModuleNotFoundError(
                "Package `xpystac` is required to be installed to use this datapipe. "
                "Please use `pip install xpystac` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.engine: str = engine
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[StreamWrapper]:
        for asset in self.source_datapipe:
            yield StreamWrapper(
                xr.open_dataset(asset, engine=self.engine, **self.kwargs)
            )

    def __len__(self) -> int:
        return len(self.source_datapipe)
