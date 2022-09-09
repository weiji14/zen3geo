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
    from spatialpandas.geometry import (
        LineDtype,
        MultiLineDtype,
        MultiPointDtype,
        MultiPolygonDtype,
        PointDtype,
        PolygonDtype,
    )
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
    raster with the input geometries aggregated into a fixed-sized grid
    (functional name: ``rasterize_with_datashader``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[datashader.Canvas]
        A DataPipe that contains :py:class:`datashader.Canvas` objects with a
        ``.crs`` attribute. This will be the template defining the output
        raster's spatial extent and x/y range.

    vector_datapipe : IterDataPipe[geopandas.GeoDataFrame]
        A DataPipe that contains :py:class:`geopandas.GeoSeries` or
        :py:class:`geopandas.GeoDataFrame` vector geometries with a
        :py:attr:`.crs <geopandas.GeoDataFrame.crs>` property.

    agg : Optional[datashader.reductions.Reduction]
        Reduction operation to compute. Default depends on the input vector
        type:

        - For points, default is :py:class:`datashader.reductions.count`
        - For lines, default is :py:class:`datashader.reductions.any`
        - For polygons, default is :py:class:`datashader.reductions.any`

        For more information, refer to the section on Aggregation under
        datashader's :doc:`datashader:getting_started/Pipeline` docs.

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

    ValueError
        If either the length of the ``vector_datapipe`` is not 1, or if the
        length of the ``vector_datapipe`` is not equal to the length of the
        ``source_datapipe``. I.e. the ratio of vector:canvas must be 1:N or
        be exactly N:N.

    AttributeError
        If either the canvas in ``source_datapipe`` or vector geometry in
        ``vector_datapipe`` is missing a ``.crs`` attribute. Please set the
        coordinate reference system (e.g. using ``canvas.crs = 'OGC:CRS84'``
        for the :py:class:`datashader.Canvas` input or
        ``vector = vector.set_crs(crs='OGC:CRS84')`` for the
        :py:class:`geopandas.GeoSeries` or :py:class:`geopandas.GeoDataFrame`
        input) before passing them into the datapipe.

    NotImplementedError
        If the input vector geometry type to ``vector_datapipe`` is not
        supported, typically when a
        :py:class:`shapely.geometry.GeometryCollection` is used. Supported
        types include `Point`, `LineString`, and `Polygon`, plus their
        multipart equivalents `MultiPoint`, `MultiLineString`, and
        `MultiPolygon`.

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
    >>> assert geodataframe.crs == "EPSG:4326"  # latitude/longitude coords
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
    >>> dataarray.rio.crs
    CRS.from_epsg(32631)
    >>> dataarray.rio.transform()
    Affine(98871.00388807665, 0.0, 160000.0,
           0.0, -68660.4193667199, 450000.0)
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        vector_datapipe: IterDataPipe,
        agg: Optional = None,
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
        self.agg: Optional = agg  # Datashader Aggregation/Reduction function
        self.kwargs = kwargs

        len_vector_datapipe: int = len(self.vector_datapipe)
        len_canvas_datapipe: int = len(self.source_datapipe)
        if len_vector_datapipe != 1 or len_vector_datapipe != len_canvas_datapipe:
            raise ValueError(
                f"Unmatched lengths for the canvas datapipe ({self.source_datapipe}) "
                f"and vector datapipe ({self.vector_datapipe}). \n"
                f"The vector datapipe's length ({len_vector_datapipe}) should either "
                f"be (1) to allow for broadcasting, or match the canvas datapipe's "
                f"length of ({len_canvas_datapipe})."
            )

    def __iter__(self) -> Iterator[xr.DataArray]:
        # Broadcast vector iterator to match length of raster iterator
        for canvas, vector in self.source_datapipe.zip_longest(
            self.vector_datapipe, fill_value=list(self.vector_datapipe).pop()
        ):
            # print(canvas, vector)
            # If canvas has no CRS attribute, set one to prevent AttributeError
            canvas.crs = getattr(canvas, "crs", None)
            if canvas.crs is None:
                raise AttributeError(
                    "Missing crs information for datashader.Canvas with "
                    f"x_range: {canvas.x_range} and y_range: {canvas.y_range}. "
                    "Please set crs using e.g. `canvas.crs = 'OGC:CRS84'`."
                )

            # Reproject vector geometries to coordinate reference system
            # of the raster canvas if both are different
            try:
                if vector.crs != canvas.crs:
                    vector = vector.to_crs(crs=canvas.crs)
            except (AttributeError, ValueError) as e:
                raise AttributeError(
                    f"Missing crs information for input {vector.__class__} object "
                    f"with the following bounds: \n {vector.bounds} \n"
                    f"Please set crs using e.g. `vector = vector.set_crs(crs='OGC:CRS84')`."
                ) from e

            # Convert vector to spatialpandas format to allow datashader's
            # rasterization methods to work
            try:
                columns = ["geometry"] if not hasattr(vector, "columns") else None
                _vector = spatialpandas.GeoDataFrame(data=vector, columns=columns)
            except ValueError as e:
                raise NotImplementedError(
                    f"Unsupported geometry type(s) {set(vector.geom_type)} detected, "
                    "only point, line or polygon vector geometry types are supported."
                ) from e

            # Determine geometry type to know which rasterization method to use
            vector_dtype: spatialpandas.geometry.GeometryDtype = _vector.geometry.dtype

            if isinstance(vector_dtype, (PointDtype, MultiPointDtype)):
                raster: xr.DataArray = canvas.points(
                    source=_vector, agg=self.agg, geometry="geometry", **self.kwargs
                )
            elif isinstance(vector_dtype, (LineDtype, MultiLineDtype)):
                raster: xr.DataArray = canvas.line(
                    source=_vector, agg=self.agg, geometry="geometry", **self.kwargs
                )
            elif isinstance(vector_dtype, (PolygonDtype, MultiPolygonDtype)):
                raster: xr.DataArray = canvas.polygons(
                    source=_vector, agg=self.agg, geometry="geometry", **self.kwargs
                )

            # Convert boolean dtype rasters to uint8 to enable reprojection
            if raster.dtype == "bool":
                raster: xr.DataArray = raster.astype(dtype="uint8")
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
    source_datapipe : IterDataPipe[xarrray.DataArray]
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
        extent and x/y coordinates of the input raster grid. This canvas
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
