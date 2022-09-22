"""
DataPipes for :doc:`pystac-client <pystac_client:index>`.
"""
from typing import Any, Dict, Iterator, Optional

try:
    import pystac_client
except ImportError:
    pystac_client = None
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("search_for_pystac_item")
class PySTACAPISearcherIterDataPipe(IterDataPipe):
    """
    Takes dictionaries containing a STAC API query (as long as the parameters
    are understood by :py:meth:`pystac_client.Client.search`) and yields
    :py:class:`pystac_client.ItemSearch` objects (functional name:
    ``search_for_pystac_item``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[dict]
        A DataPipe that contains STAC API query parameters in the form of a
        Python dictionary to pass to :py:meth:`pystac_client.Client.search`.
        For example:

        - **bbox** -  A list, tuple, or iterator representing a bounding box of
          2D or 3D coordinates. Results will be filtered to only those
          intersecting the bounding box.
        - **datetime** - Either a single datetime or datetime range used to
          filter results. You may express a single datetime using a
          :py:class:`datetime.datetime` instance, a
          `RFC 3339-compliant <https://tools.ietf.org/html/rfc3339>`_
          timestamp, or a simple date string.
        - **collections** - List of one or more Collection IDs or
          :py:class:`pystac.Collection` instances. Only Items in one of the
          provided Collections will be searched.

    catalog_url : str
        The URL of a STAC Catalog. If not specified, this will use the
        ``STAC_URL`` environment variable.

    kwargs : Optional
        Extra keyword arguments to pass to
        :py:meth:`pystac_client.Client.open`. For example:

        - **headers** - A dictionary of additional headers to use in all
          requests made to any part of this Catalog/API.
        - **parameters** - Optional dictionary of query string parameters to
          include in all requests.
        - **modifier** - A callable that modifies the children collection and
          items returned by this Client. This can be useful for injecting
          authentication parameters into child assets to access data from
          non-public sources.

    Yields
    ------
    item_search : pystac_client.ItemSearch
        A :py:class:`pystac_client.ItemSearch` object instance that represents
        a deferred query to a STAC search endpoint as described in the
        `STAC API - Item Search spec <https://github.com/radiantearth/stac-api-spec/tree/main/item-search>`_.

    Raises
    ------
    ModuleNotFoundError
        If ``pystac_client`` is not installed. See
        :doc:`install instructions for pystac-client <pystac_client:index>`,
        (e.g. via ``pip install pystac-client``) before using this class.

    Example
    -------
    >>> import pytest
    >>> pystac_client = pytest.importorskip("pystac_client")
    ...
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> from zen3geo.datapipes import PySTACAPISearcher
    ...
    >>> # Peform STAC API query using DataPipe
    >>> query = dict(
    ...     bbox=[174.5, -41.37, 174.9, -41.19],
    ...     datetime=["2012-02-20T00:00:00Z", "2022-12-22T00:00:00Z"],
    ...     collections=["cop-dem-glo-30"],
    ... )
    >>> dp = IterableWrapper(iterable=[query])
    >>> dp_pystac_client = dp.search_for_pystac_item(
    ...     catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    ...     # modifier=planetary_computer.sign_inplace,
    ... )
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_pystac_client)
    >>> stac_item_search = next(it)
    >>> stac_items = list(stac_item_search.items())
    >>> stac_items
    [<Item id=Copernicus_DSM_COG_10_S42_00_E174_00_DEM>]
    >>> stac_items[0].properties  # doctest: +NORMALIZE_WHITESPACE
    {'gsd': 30,
     'datetime': '2021-04-22T00:00:00Z',
     'platform': 'TanDEM-X',
     'proj:epsg': 4326,
     'proj:shape': [3600, 3600],
     'proj:transform': [0.0002777777777777778,
      0.0,
      173.9998611111111,
      0.0,
      -0.0002777777777777778,
      -40.99986111111111]}
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[dict],
        catalog_url: str,
        **kwargs: Optional[Dict[str, Any]]
    ) -> None:
        if pystac_client is None:
            raise ModuleNotFoundError(
                "Package `pystac_client` is required to be installed to use this datapipe. "
                "Please use `pip install pystac-client` or "
                "`conda install -c conda-forge pystac-client` "
                "to install the package"
            )
        self.source_datapipe: IterDataPipe[dict] = source_datapipe
        self.catalog_url: str = catalog_url
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:
        catalog = pystac_client.Client.open(url=self.catalog_url, **self.kwargs)

        for query in self.source_datapipe:
            search = catalog.search(**query)
            yield search

    def __len__(self) -> int:
        return len(self.source_datapipe)
