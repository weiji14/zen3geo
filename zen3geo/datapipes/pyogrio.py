"""
DataPipes for :doc:`pyogrio <pyogrio:index>`.
"""
from typing import Any, Dict, Iterator, Optional

try:
    import pyogrio
except ImportError:
    pyogrio = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("read_from_pyogrio")
class PyogrioReaderIterDataPipe(IterDataPipe[StreamWrapper]):
    """
    Takes vector files (e.g. FlatGeoBuf, GeoPackage, GeoJSON) from local disk
    or URLs (as long as they can be read by pyogrio) and yields
    :py:class:`geopandas.GeoDataFrame` objects (functional name:
    ``read_from_pyogrio``).

    Based on
    https://github.com/pytorch/data/blob/v0.4.0/torchdata/datapipes/iter/load/iopath.py#L42-L97

    Parameters
    ----------
    source_datapipe : IterDataPipe[str]
        A DataPipe that contains filepaths or URL links to vector files such as
        FlatGeoBuf, GeoPackage, GeoJSON, etc.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`pyogrio.read_dataframe`.

    Yields
    ------
    stream_obj : geopandas.GeoDataFrame
        A :py:class:`geopandas.GeoDataFrame` object containing the vector data.

    Raises
    ------
    ModuleNotFoundError
        If ``pyogrio`` is not installed. See
        :doc:`install instructions for pyogrio <pyogrio:install>`, and ensure
        that ``geopandas`` is installed too (e.g. via
        ``pip install pyogrio[geopandas]``) before using this class.

    Example
    -------
    >>> import pytest
    >>> pyogrio = pytest.importorskip("pyogrio")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import PyogrioReader
    ...
    >>> # Read in GeoPackage data using DataPipe
    >>> file_url: str = "https://github.com/geopandas/pyogrio/raw/v0.4.0/pyogrio/tests/fixtures/test_gpkg_nulls.gpkg"
    >>> dp = IterableWrapper(iterable=[file_url])
    >>> dp_pyogrio = dp.read_from_pyogrio()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_pyogrio)
    >>> geodataframe = next(it)
    >>> geodataframe
    StreamWrapper<   col_bool  col_int8  ...  col_float64                 geometry
    0       1.0       1.0  ...          1.5  POINT (0.00000 0.00000)
    1       0.0       2.0  ...          2.5  POINT (1.00000 1.00000)
    2       1.0       3.0  ...          3.5  POINT (2.00000 2.00000)
    3       NaN       NaN  ...          NaN  POINT (4.00000 4.00000)
    <BLANKLINE>
    [4 rows x 12 columns]>
    """

    def __init__(
        self, source_datapipe: IterDataPipe[str], **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if pyogrio is None:
            raise ModuleNotFoundError(
                "Package `pyogrio` is required to be installed to use this datapipe. "
                "Please use `pip install pyogrio[geopandas]` or "
                "`conda install -c conda-forge pyogrio` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[StreamWrapper]:
        for filename in self.source_datapipe:
            yield StreamWrapper(pyogrio.read_dataframe(filename, **self.kwargs))

    def __len__(self) -> int:
        return len(self.source_datapipe)
