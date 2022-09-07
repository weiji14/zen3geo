"""
DataPipes for :doc:`geopandas <geopandas:index>`.
"""
from typing import Any, Dict, Iterator, Optional, Union

try:
    import geopandas as gpd
except ImportError:
    gpd = None
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("clip_vector_with_rectangle")
class GeoPandasRectangleClipperIterDataPipe(IterDataPipe):
    """
    Takes vector :py:class:`geopandas.GeoSeries` or
    :py:class:`geopandas.GeoDataFrame` geometries and clips them with the
    rectangular extent of an :py:class:`xarray.DataArray` or
    :py:class:`xarray.Dataset` grid to yield tuples of spatially subsetted
    :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
    vectors and the correponding :py:class:`xarray.DataArray` or
    :py:class:`xarray.Dataset` raster object used as the clip mask (functional
    name: ``clip_vector_with_rectangle``).

    Uses the rectangular clip algorithm of :py:func:`geopandas.clip`, with the
    bounding box rectangle (minx, miny, maxx, maxy) derived from input raster
    mask's bounding box extent.

    Note
    ----
    If the input vector's coordinate reference system (``crs``) is different to
    the raster mask's coordinate reference system (``rio.crs``), the vector
    will be reprojected using :py:meth:`geopandas.GeoDataFrame.to_crs` to match
    the raster's coordinate reference system.

    Parameters
    ----------
    source_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains :py:class:`geopandas.GeoSeries` or
        :py:class:`geopandas.GeoDataFrame` vector geometries with a
        :py:attr:`.crs <geopandas.GeoDataFrame.crs>` property.

    mask_datapipe : IterDataPipe[xarray.DataArray]
        A DataPipe that contains :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` objects with a
        :py:attr:`.rio.crs <rioxarray.rioxarray.XRasterBase.crs>` property and
        :py:meth:`.rio.bounds <rioxarray.rioxarray.XRasterBase.bounds>` method.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`geopandas.clip`.

    Yields
    ------
    paired_obj : Tuple[geopandas.GeoDataFrame, xarray.DataArray]
        A tuple consisting of the spatially subsetted
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
        vector, and the corresponding :py:class:`xarray.DataArray` or
        :py:class:`xarray.Dataset` raster used as the clip mask.

    Raises
    ------
    ModuleNotFoundError
        If ``geopandas`` is not installed. See
        :doc:`install instructions for geopandas <geopandas:getting_started/install>`
        (e.g. via ``pip install geopandas``) before using this class.

    NotImplementedError
        If the length of the vector ``source_datapipe`` is not 1. Currently,
        all of the vector geometries have to be merged into a single
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`.
        Refer to the section on Appending under geopandas'
        :doc:`geopandas:docs/user_guide/mergingdata` docs.

    Example
    -------
    >>> import pytest
    >>> import rioxarray
    >>> gpd = pytest.importorskip("geopandas")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import GeoPandasRectangleClipper
    ...
    >>> # Read in a vector polygon data source
    >>> geodataframe = gpd.read_file(
    ...     filename="https://github.com/geopandas/geopandas/raw/v0.11.1/geopandas/tests/data/overlay/polys/df1.geojson",
    ... )
    >>> assert geodataframe.crs == "EPSG:4326"  # latitude/longitude coords
    >>> dp_vector = IterableWrapper(iterable=[geodataframe])
    ...
    >>> # Get list of raster grids to cut up the vector polygon later
    >>> dataarray = rioxarray.open_rasterio(
    ...     filename="https://github.com/rasterio/rasterio/raw/1.3.2/tests/data/world.byte.tif"
    ... )
    >>> assert dataarray.rio.crs == "EPSG:4326"  # latitude/longitude coords
    >>> dp_raster = IterableWrapper(
    ...     iterable=[
    ...         dataarray.sel(x=slice(0, 2)),  # longitude 0 to 2 degrees
    ...         dataarray.sel(x=slice(2, 4)),  # longitude 2 to 4 degrees
    ...     ]
    ... )
    ...
    >>> # Clip vector point geometries based on raster masks
    >>> dp_clipped = dp_vector.clip_vector_with_rectangle(
    ...     mask_datapipe=dp_raster
    ... )
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_clipped)
    >>> geodataframe0, raster0 = next(it)
    >>> geodataframe0
       col1                                           geometry
    0     1  POLYGON ((0.00000 0.00000, 0.00000 2.00000, 2....
    >>> raster0
    <xarray.DataArray (band: 1, y: 1200, x: 16)>
    array([[[0, 0, ..., 0, 0],
            [0, 0, ..., 0, 0],
            ...,
            [1, 1, ..., 1, 1],
            [1, 1, ..., 1, 1]]], dtype=uint8)
    Coordinates:
      * band         (band) int64 1
      * x            (x) float64 0.0625 0.1875 0.3125 0.4375 ... 1.688 1.812 1.938
      * y            (y) float64 74.94 74.81 74.69 74.56 ... -74.69 -74.81 -74.94
        spatial_ref  int64 0
    ...
    >>> geodataframe1, raster1 = next(it)
    >>> geodataframe1
       col1                                           geometry
    1     2  POLYGON ((2.00000 2.00000, 2.00000 4.00000, 4....
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        mask_datapipe: IterDataPipe[Union[xr.DataArray, xr.Dataset]],
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if gpd is None:
            raise ModuleNotFoundError(
                "Package `geopandas` is required to be installed to use this datapipe. "
                "Please use `pip install geopandas` or "
                "`conda install -c conda-forge geopandas` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe
        self.mask_datapipe: IterDataPipe[xr.DataArray] = mask_datapipe
        self.kwargs = kwargs

        len_vector_datapipe: int = len(self.source_datapipe)
        if len_vector_datapipe != 1:
            raise NotImplementedError(
                f"The vector datapipe's length can only be (1) for now, but got "
                f"({len_vector_datapipe}) instead. Consider merging your vector data "
                f"into a single `geopandas.GeoSeries` or `geopandas.GeoDataFrame`, "
                f"e.g. using `geodataframe0.append(geodataframe2)`."
            )

    def __iter__(self) -> Iterator:
        geodataframe = list(self.source_datapipe).pop()

        for raster in self.mask_datapipe:
            mask = raster.rio.bounds()

            try:
                assert geodataframe.crs == raster.rio.crs
                _geodataframe = geodataframe
            except AssertionError:
                _geodataframe = geodataframe.to_crs(crs=raster.rio.crs)

            clipped_geodataframe = _geodataframe.clip(mask=mask, **self.kwargs)

            yield clipped_geodataframe, raster

    def __len__(self) -> int:
        return len(self.mask_datapipe)
