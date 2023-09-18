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
        The URL of a STAC Catalog.

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
    ...     bbox=[174.5, -41.37, 174.9, -41.19],  # xmin, ymin, xmax, ymax
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


@functional_datapipe("list_pystac_items_by_search")
class PySTACAPIItemListerIterDataPipe(IterDataPipe):
    """
    Lists the :py:class:`pystac.Item` objects that match the provided STAC API
    search parameters (functional name: ``list_pystac_items_by_search``).

    Parameters
    ----------
    source_datapipe : IterDataPipe[pystac_client.ItemSearch]
        A DataPipe that contains :py:class:`pystac_client.ItemSearch` object
        instances that represents
        a deferred query to a STAC search endpoint as described in the
        `STAC API - Item Search spec <https://github.com/radiantearth/stac-api-spec/tree/main/item-search>`_.

    Yields
    ------
    stac_item : pystac.Item
        A :py:class:`pystac.Item` object containing the specific
        :py:class:`pystac.STACObject` implementation class represented in a
        JSON format.

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
    >>> from zen3geo.datapipes import PySTACAPIItemLister
    ...
    >>> # List STAC Items from a STAC API query
    >>> catalog = pystac_client.Client.open(
    ...     url="https://explorer.digitalearth.africa/stac/"
    ... )
    >>> search = catalog.search(
    ...     bbox=[57.2, -20.6, 57.9, -19.9],  # xmin, ymin, xmax, ymax
    ...     datetime=["2023-01-01T00:00:00Z", "2023-01-31T00:00:00Z"],
    ...     collections=["s2_l2a"],
    ... )
    >>> dp = IterableWrapper(iterable=[search])
    >>> dp_pystac_item_list = dp.list_pystac_items_by_search()
    ...
    >>> # Loop or iterate over the DataPipe stream
    >>> it = iter(dp_pystac_item_list)
    >>> stac_item = next(it)
    >>> stac_item
    <Item id=ec16dbf6-9729-5a8f-9d72-5e83a8b9f30d>
    >>> stac_item.properties  # doctest: +NORMALIZE_WHITESPACE
    {'title': 'S2B_MSIL2A_20230103T062449_N0509_R091_T40KED_20230103T075000',
     'gsd': 10,
     'proj:epsg': 32740,
     'platform': 'sentinel-2b',
     'view:off_nadir': 0,
     'instruments': ['msi'],
     'eo:cloud_cover': 0.02,
     'odc:file_format': 'GeoTIFF',
     'odc:region_code': '40KED',
     'constellation': 'sentinel-2',
     'sentinel:sequence': '0',
     'sentinel:utm_zone': 40,
     'sentinel:product_id': 'S2B_MSIL2A_20230103T062449_N0509_R091_T40KED_20230103T075000',
     'sentinel:grid_square': 'ED',
     'sentinel:data_coverage': 28.61,
     'sentinel:latitude_band': 'K',
     'created': '2023-01-03T06:24:53Z',
     'sentinel:valid_cloud_cover': True,
     'sentinel:boa_offset_applied': True,
     'sentinel:processing_baseline': '05.09',
     'proj:shape': [10980, 10980],
     'proj:transform': [10.0, 0.0, 499980.0, 0.0, -10.0, 7900000.0, 0.0, 0.0, 1.0],
     'cubedash:region_code': '40KED',
     'datetime': '2023-01-03T06:24:53Z'}
    """

    def __init__(self, source_datapipe):
        if pystac_client is None:
            raise ModuleNotFoundError(
                "Package `pystac_client` is required to be installed to use this datapipe. "
                "Please use `pip install pystac-client` or "
                "`conda install -c conda-forge pystac-client` "
                "to install the package"
            )
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for item_search in self.source_datapipe:
            yield from item_search.items()

    def __len__(self):
        return sum(item_search.matched() for item_search in self.source_datapipe)
