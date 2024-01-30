"""
DataPipes for :doc:`laspy <laspy:index>`.
"""
import io
from typing import Any, Dict, Iterator, Optional

try:
    import laspy
except ImportError:
    laspy = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("read_from_laspy")
class LaspyReaderIterDataPipe(IterDataPipe[StreamWrapper]):
    """
    Takes LAS/LAZ files from local disk or an :py:class:`io.BytesIO` stream (as long as
    they can be read by laspy) and yields :py:class:`laspy.lasdata.LasData` objects
    (functional name: ``read_from_laspy``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[str]
        A DataPipe that contains filepaths or an :py:class:`io.BytesIO` stream to point
        cloud files in LAS or LAZ format.

    kwargs : Optional
        Extra keyword arguments to pass to :py:func:`laspy.read`.

    Yields
    ------
    stream_obj : laspy.lasdata.LasData
        A :py:class:`laspy.lasdata.LasData` object containing the point cloud data.

    Raises
    ------
    ModuleNotFoundError
        If ``laspy`` is not installed. See
        :doc:`install instructions for laspy <laspy:installation>`, (e.g. via
        ``pip install laspy[lazrs]``) before using this class.

    Example
    -------
    >>> import pytest
    >>> laspy = pytest.importorskip("laspy")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import LaspyReader
    ...
    >>> # Read in LAZ data using DataPipe
    >>> file_url: str = "https://opentopography.s3.sdsc.edu/pc-bulk/NZ19_Wellington/CL2_BQ31_2019_1000_2138.laz"
    >>> dp = IterableWrapper(iterable=[file_url])
    >>> _, dp_stream = dp.read_from_http().unzip(sequence_length=2)
    >>> dp_laspy = dp_stream.read_from_laspy()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_laspy)
    >>> lasdata = next(it)
    >>> lasdata.header
    <LasHeader(1.4, <PointFormat(6, 0 bytes of extra dims)>)>
    >>> lasdata.xyz
    array([[ 1.74977156e+06,  5.42749877e+06, -7.24000000e-01],
           [ 1.74977152e+06,  5.42749846e+06, -7.08000000e-01],
           [ 1.74977148e+06,  5.42749815e+06, -7.00000000e-01],
           ...,
           [ 1.74976026e+06,  5.42756798e+06, -4.42000000e-01],
           [ 1.74976029e+06,  5.42756829e+06, -4.17000000e-01],
           [ 1.74976032e+06,  5.42756862e+06, -4.04000000e-01]])
    """

    def __init__(
        self, source_datapipe: IterDataPipe[str], **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if laspy is None:
            raise ModuleNotFoundError(
                "Package `laspy` is required to be installed to use this datapipe. "
                "Please use `pip install laspy` or "
                "`conda install -c conda-forge laspy` to install the package"
            )
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[StreamWrapper]:
        for lazstream in self.source_datapipe:
            yield StreamWrapper(laspy.read(source=lazstream, **self.kwargs))

    def __len__(self) -> int:
        return len(self.source_datapipe)
