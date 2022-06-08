"""
DataPipes for rioxarray.
"""
from typing import Any, Dict, Iterator, Optional, Tuple

import rioxarray
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("read_from_rioxarray")
class RioXarrayReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    """
    Takes raster files (e.g. GeoTIFFs) from local disk or URLs
    (as long as they can be read by rioxarray and/or rasterio)
    and yields tuples of filename and :py:class:`xarray.DataArray` objects
    (functional name: ``read_from_rioxarray``).

    Based on
    https://github.com/pytorch/data/blob/v0.3.0/torchdata/datapipes/iter/load/online.py#L29-L59

    Parameters
    ----------
    source_datapipe : IterDataPipe[str]
        A DataPipe that contains filepaths or URL links to raster files such as
        GeoTIFFs.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`rioxarray.open_rasterio`
        and/or :py:func:`rasterio.open`.

    Yields
    ------
    stream_obj : Tuple[str, xarray.DataArray]
        A tuple consisting of the filename that was passed in, and an
        :py:class:`xarray.DataArray` object containing the raster data.

    Example
    -------
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import RioXarrayReader
    ...
    >>> # Read in GeoTIFF data using DataPipe
    >>> file_url: str = "https://github.com/GenericMappingTools/gmtserver-admin/raw/master/cache/earth_day_HD.tif"
    >>> dp = IterableWrapper(iterable=[file_url])
    >>> dp_rioxarray = dp.read_from_rioxarray()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_rioxarray)
    >>> filename, dataarray = next(it)
    >>> filename
    'https://github.com/GenericMappingTools/gmtserver-admin/raw/master/cache/earth_day_HD.tif'
    >>> dataarray
    StreamWrapper<<xarray.DataArray (band: 1, y: 960, x: 1920)>
    [1843200 values with dtype=uint8]
    Coordinates:
      * band         (band) int64 1
      * x            (x) float64 -179.9 -179.7 -179.5 -179.3 ... 179.5 179.7 179.9
      * y            (y) float64 89.91 89.72 89.53 89.34 ... -89.53 -89.72 -89.91
        spatial_ref  int64 0
    ...
    """

    def __init__(
        self, source_datapipe: IterDataPipe[str], **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple]:
        for filename in self.source_datapipe:
            yield (
                filename,
                StreamWrapper(
                    rioxarray.open_rasterio(filename=filename, **self.kwargs)
                ),
            )

    def __len__(self) -> int:
        return len(self.source_datapipe)
