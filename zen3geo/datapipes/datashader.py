"""
DataPipes for :doc:`datashader <datashader:index>`.
"""
from typing import Any, Dict, Iterator, Optional, Union

try:
    import datashader
except ImportError:
    datashader = None
try:
    import spatialpandas
except ImportError:
    spatialpandas = None

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("rasterize_with_datashader")
class DatashaderRasterizerIterDataPipe(IterDataPipe):
    """
    Takes vector :py:class:`geopandas.GeoSeries` or
    :py:class:`geopandas.GeoDataFrame` geometries and rasterizes them using
    :py:class:`datashader.Canvas` to yield an :py:class:`xarray.DataArray`
    raster image with input geometries burned in
    (functional name: ``rasterize_with_datashader``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[datashader.Canvas]
        A DataPipe that contains :py:class:`datashader.Canvas` objects with a
        ``.crs`` attribute. This will be the template defining the output
        raster's spatial extent and x/y range.

    vector_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains :py:class:`geopandas.GeoSeries` or
        :py:class:`geopandas.GeoDataFrame` vector geometries.

    kwargs : Optional
        Extra keyword arguments to pass to the :py:class:`datashader.Canvas`
        class's aggregation methods such as ``datashader.Canvas.points``.

    Yields
    ------
    raster : xarray.DataArray
        An :py:class:`xarray.DataArray` object containing the raster data. This
        raster will have a :py:attr:`rioxarray.rioxarray.XRasterBase.crs`
        property and a proper affine transform viewable with
        :py:meth:`rioxarray.rioxarray.XRasterBase.transform`.

    Raises
    ------
    ModuleNotFoundError
        If ``spatialpandas`` is not installed. Please install it (e.g. via
        ``pip install spatialpandas``) before using this class.

    Example
    -------
    >>> import pytest
    >>> datashader = pytest.importorskip("datashader")
    >>> pyogrio = pytest.importorskip("pyogrio")
    >>> spatialpandas = pytest.importorskip("spatialpandas")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import DatashaderRasterizer
    ...
    >>> # Read in a vector point data source
    >>> geodataframe = pyogrio.read_dataframe(
    ...     "https://github.com/geopandas/pyogrio/raw/v0.4.0/pyogrio/tests/fixtures/test_gpkg_nulls.gpkg",
    ...     read_geometry=True,
    ... )
    >>> assert geodataframe.crs == "EPSG:4326"  # longitude/latitude coords
    >>> dp_vector = IterableWrapper(iterable=[geodataframe])
    ...
    >>> # Setup blank raster canvas where we will burn vector geometries onto
    >>> canvas = datashader.Canvas(
    ...     plot_width=5,
    ...     plot_height=6,
    ...     x_range=(160000.0, 620000.0),
    ...     y_range=(0.0, 450000.0),
    ... )
    >>> canvas.crs = "EPSG:32631"  # UTM Zone 31N, North of Gulf of Guinea
    >>> dp_canvas = IterableWrapper(iterable=[canvas])
    ...
    >>> # Rasterize vector point geometries onto blank canvas
    >>> dp_datashader = dp_canvas.rasterize_with_datashader(
    ...     vector_datapipe=dp_vector
    ... )
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_datashader)
    >>> dataarray = next(it)
    >>> dataarray
    <xarray.DataArray (y: 6, x: 5)>
    array([[0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0]], dtype=uint32)
    Coordinates:
      * x            (x) float64 2.094e+05 3.083e+05 4.072e+05 5.06e+05 6.049e+05
      * y            (y) float64 4.157e+05 3.47e+05 2.783e+05 ... 1.41e+05 7.237e+04
        spatial_ref  int64 0
    ...
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        vector_datapipe: IterDataPipe,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        if spatialpandas is None:
            raise ModuleNotFoundError(
                "Package `spatialpandas` is required to be installed to use this datapipe. "
                "Please use `pip install spatialpandas` or "
                "`conda install -c conda-forge spatialpandas` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe = source_datapipe  # datashader.Canvas
        self.vector_datapipe: IterDataPipe = vector_datapipe  # geopandas.GeoDataFrame
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[xr.DataArray]:
        # Broadcast vector iterator to match length of raster iterator
        fill_value: Optional = (
            list(self.vector_datapipe).pop() if len(self.vector_datapipe) == 1 else None
        )
        for canvas, vector in self.source_datapipe.zip_longest(
            self.vector_datapipe, fill_value=fill_value
        ):
            # If canvas has no CRS attribute, set one to prevent AttributeError
            canvas.crs = getattr(canvas, "crs", None)
            # Reproject vector geometries to coordinate reference system
            # of the raster canvas if both are different
            if vector.crs != canvas.crs:
                vector = vector.to_crs(crs=canvas.crs)

            # Convert vector to spatialpandas format to allow datashader's
            # rasterization methods to work
            try:
                _vector = spatialpandas.GeoDataFrame(data=vector.geometry)
            except ValueError as e:
                raise ValueError(
                    f"Unsupported geometry type(s) {set(vector.geom_type)} detected, "
                    "only point, line or polygon vector geometry types are supported."
                ) from e

            # Determine geometry type to know which rasterization method to use
            vector_dtype: spatialpandas.geometry.GeometryDtype = _vector.geometry.dtype
            if isinstance(
                vector_dtype,
                (
                    spatialpandas.geometry.PointDtype,
                    spatialpandas.geometry.MultiPointDtype,
                ),
            ):
                raster: xr.DataArray = canvas.points(
                    source=_vector, geometry="geometry", **self.kwargs
                )

            # Set coordinate transform for raster and ensure affine
            # transform is correct (the y-coordinate goes from North to South)
            raster: xr.DataArray = raster.rio.set_crs(input_crs=canvas.crs)
            # assert raster.rio.transform().e > 0  # y goes South to North
            _raster: xr.DataArray = raster.rio.reproject(
                dst_crs=canvas.crs, shape=raster.rio.shape
            )
            # assert _raster.rio.transform().e < 0  # y goes North to South

            yield _raster

    def __len__(self) -> int:
        return len(self.source_datapipe)


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
        Extra keyword arguments to pass to :py:class:`datashader.Canvas`.

    Yields
    ------
    canvas : datashader.Canvas
        A :py:class:`datashader.Canvas` object representing the same spatial
        extent and x/y coordinates of the input raster image. This canvas
        will also have a ``.crs`` attribute that captures the original
        Coordinate Reference System from the input xarray object's
        :py:attr:`rioxarray.rioxarray.XRasterBase.crs` property.

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
        **kwargs: Optional[Dict[str, Any]],
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
                **self.kwargs,
            )
            canvas.crs = dataarray.rio.crs
            yield canvas

    def __len__(self) -> int:
        return len(self.source_datapipe)
