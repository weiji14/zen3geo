"""
DataPipes for :doc:`pystac <pystac:index>`.
"""
from typing import Any, Dict, Iterator, Optional

try:
    import pystac
except ImportError:
    pystac = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("read_to_pystac_item")
class PySTACItemReaderIterDataPipe(IterDataPipe):
    """
    Takes files from local disk or URLs (as long as they can be read by pystac)
    and yields :py:class:`pystac.Item` objects (functional name:
    ``read_to_pystac_item``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[str]
        A DataPipe that contains filepaths or URL links to STAC items.

    kwargs : Optional
        Extra keyword arguments to pass to :py:meth:`pystac.Item.from_file`.

    Yields
    ------
    stac_item : pystac.Item
        An :py:class:`pystac.Item` object containing the specific STACObject
        implementation class represented in a JSON format.

    Raises
    ------
    ModuleNotFoundError
        If ``pystac`` is not installed. See
        :doc:`install instructions for pystac <pystac:installation>`, (e.g. via
        ``pip install pystac``) before using this class.

    Example
    -------
    >>> import pytest
    >>> pystac = pytest.importorskip("pystac")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import PySTACItemReader
    ...
    >>> # Read in STAC Item using DataPipe
    >>> item_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_20220115T032101_R118_T48NUG_20220115T170435"
    >>> dp = IterableWrapper(iterable=[item_url])
    >>> dp_pystac = dp.read_to_pystac_item()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_pystac)
    >>> stac_item = next(it)
    >>> stac_item.bbox
    [103.20205689, 0.81602476, 104.18934086, 1.8096362]
    >>> stac_item.properties  # doctest: +NORMALIZE_WHITESPACE
    {'datetime': '2022-01-15T03:21:01.024000Z',
     'platform': 'Sentinel-2A',
     'proj:epsg': 32648,
     'instruments': ['msi'],
     's2:mgrs_tile': '48NUG',
     'constellation': 'Sentinel 2',
     's2:granule_id': 'S2A_OPER_MSI_L2A_TL_ESRI_20220115T170436_A034292_T48NUG_N03.00',
     'eo:cloud_cover': 17.352597,
     's2:datatake_id': 'GS2A_20220115T032101_034292_N03.00',
     's2:product_uri': 'S2A_MSIL2A_20220115T032101_N0300_R118_T48NUG_20220115T170435.SAFE',
     's2:datastrip_id': 'S2A_OPER_MSI_L2A_DS_ESRI_20220115T170436_S20220115T033502_N03.00',
     's2:product_type': 'S2MSI2A',
     'sat:orbit_state': 'descending',
    ...
    """

    def __init__(
        self, source_datapipe: IterDataPipe[str], **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if pystac is None:
            raise ModuleNotFoundError(
                "Package `pystac` is required to be installed to use this datapipe. "
                "Please use `pip install pystac` or "
                "`conda install -c conda-forge pystac` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:
        for href in self.source_datapipe:
            yield pystac.Item.from_file(href=href, **self.kwargs)

    def __len__(self) -> int:
        return len(self.source_datapipe)
