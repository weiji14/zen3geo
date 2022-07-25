"""
DataPipes for :doc:`rasterio <rasterio:index>`.
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import affine
import numpy as np
import rasterio
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


@functional_datapipe("rasterize_with_rasterio")
class RasterioRasterizerIterDataPipe(IterDataPipe):
    """
    Takes a vector :py:class:`geopandas.GeoSeries`,
    :py:class:`geopandas.GeoDataFrame` or
    :py:class:`shapely.geometry.GeometryCollection`, and yields a raster
    :py:class:`xarray.DataArray` or :py:class:`numpy.ndarray` image with input
    geometries burned in (functional name: ``rasterize_with_rasterio``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains geometries or (`geometry`, `value`) pairs. The
        `geometry` can either be:

        - :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
          (recommended)
        - :py:class:`shapely.geometry.GeometryCollection` or other
          ``shapely.geometry`` object (Point, Line, Polygon)
        - a GeoJSON-like :py:class:`dict` object
        - an object that implements the geo interface (see
          https://gist.github.com/sgillies/2217756#__geo_interface__)

        If no `value` is provided the `default_value` will be used. If `value`
        is `None` the `fill` value will be used.

    out : IterDataPipe[xarray.DataArray]
        Optional. An :py:class:`xarray.DataArray` (recommended) or
        :py:class:`numpy.ndarray` with a given shape in which to store results.

    out_shape : tuple[int, int] or list[int, int]
        Required if ``out`` is not set. Shape of output numpy ndarray.

    transform : affine.Affine
        Optional. An affine.Affine object (e.g. ``from affine import Affine;
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0)`` giving the affine
        transformation used to convert raster coordinates (e.g. [0, 0]) to
        geographic coordinates. If none is provided, the function will attempt
        to obtain an affine transformation from the xarray object (i.e. using
        ``out.rio.transform()``).

    kwargs : Optional
        Extra keyword arguments to pass to
        :py:func:`rasterio.features.rasterize`.

    Yields
    ------
    image : xarray.DataArray or numpy.ndarray
        An array-like object containing the raster data. The return type
        depends on the ``out`` or ``out_shape`` parameter:

        - :py:class:`xarray.DataArray` if ``out`` is an
          :py:class:`xarray.DataArray`
        - :py:class:`numpy.ndarray` if ``out`` is a :py:class:`numpy.ndarray`
          or if ``out_shape`` is set.

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
        self,
        source_datapipe: IterDataPipe,
        out: Optional[
            Union[IterDataPipe[xr.DataArray], IterDataPipe[np.ndarray]]
        ] = IterableWrapper([None]),
        out_shape: Optional[
            Union[IterDataPipe[Tuple[int]], IterDataPipe[List[int]]]
        ] = None,
        transform: rasterio.Affine = rasterio.transform.IDENTITY,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        self.source_datapipe: IterDataPipe = source_datapipe
        self.out: IterDataPipe = out
        self.out_shape: IterDataPipe = out_shape
        self.transform: rasterio.Affine = transform
        self.kwargs = kwargs

        self.len_iter_vector = len(self.source_datapipe)
        self.len_iter_raster = len(self.out)

    def __iter__(self) -> Iterator[Union[xr.DataArray, np.ndarray]]:
        # Broadcast vector iterator to match length of raster iterator
        fill_value = (
            list(self.source_datapipe)[0] if self.len_iter_vector == 1 else None
        )
        for _out, _geoseries in self.out.zip_longest(
            self.source_datapipe, fill_value=fill_value
        ):
            if hasattr(_out, "rio"):  # if raster has rioxarray accessors
                # Get affine transform from template raster
                _out = _out * 0  # empty xarray.DataArray template
                if self.transform is rasterio.transform.IDENTITY:
                    self.transform = _out.rio.transform()

                # Reproject vector to same coordinate reference system as the
                # template raster image
                if hasattr(_geoseries, "crs") and _geoseries.crs != _out.rio.crs:
                    _geoseries: gpd.GeoSeries = _geoseries.to_crs(
                        crs=_out.rio.crs
                    ).geometry

            yield rasterio.features.rasterize(
                shapes=_geoseries,
                out=_out,
                out_shape=self.out_shape,
                transform=self.transform,
                **self.kwargs
            )

    def __len__(self) -> int:
        return max(self.len_iter_vector, self.len_iter_raster)
