"""
DataPipes for :doc:`rasterio <rasterio:index>`.
"""
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import rasterio
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("rasterize_with_rasterio")
class RasterioRasterizerIterDataPipe(IterDataPipe):
    """
    Takes a vector :py:class:`shapely.geometry.GeometryCollection`,
    :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame` and
    yields a raster :py:class:`numpy.ndarray` or :py:class:`xarray.DataArray`
    image with input geometries burned in
    (functional name: ``rasterize_with_rasterio``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains geometries or (`geometry`, `value`) pairs. The
        `geometry` can either be an object that implements the geo interface
        (e.g. :py:class:`shapely.geometry.GeometryCollection`, :py:class:`geopandas.GeoSeries` or
        :py:class:`geopandas.GeoDataFrame`) or a GeoJSON-like object. If no
        `value` is provided the `default_value` will be used. If `value` is
        `None` the `fill` value will be used.

    kwargs : Optional
        Extra keyword arguments to pass to
        :py:func:`rasterio.features.rasterize`.

    Yields
    ------
    image : numpy.ndarray or xarray.DataArray
        An :py:class:`numpy.ndarray` or :py:class:`xarray.DataArray` object
        containing the raster data.

    Example
    -------
    >>> import shapely.geometry
    >>>
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import RasterioRasterizer
    ...
    >>> # Create a vector point geometry
    >>> geometry = shapely.geometry.MultiPoint(points=[[0, 0], [1, 2], [3, 3]])
    >>> dp = IterableWrapper(iterable=[geometry])
    >>> dp_rasterio = dp.rasterize_with_rasterio(out_shape=(5, 4))
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_rasterio)
    >>> array = next(it)
    >>> array
    array([[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 0]], dtype=uint8)
    """

    def __init__(
        self, source_datapipe: IterDataPipe, **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.source_datapipe: IterDataPipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Union[np.ndarray]]:
        for geodataframe in self.source_datapipe:
            yield rasterio.features.rasterize(shapes=geodataframe, **self.kwargs)

    def __len__(self) -> int:
        return len(self.source_datapipe)
