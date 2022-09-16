"""
Tests for pystac-client datapipes.
"""
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import PySTACAPISearch

pystac_client = pytest.importorskip("pystac_client")

# %%
def test_pystac_client_item_search():
    """
    Ensure that PySTACAPISearch works to query a STAC API /search/ endpoint and
    outputs a pystac_client.ItemSearch object.
    """
    query: dict = dict(
        bbox=[150.9, -34.36, 151.3, -33.46],
        datetime=["2000-01-01T00:00:00Z", "2020-12-31T00:00:00Z"],
        collections=["nidem"],
    )
    dp = IterableWrapper(iterable=[query])

    # Using class constructors
    dp_pystac_client = PySTACAPISearch(
        source_datapipe=dp, catalog_url="https://explorer.sandbox.dea.ga.gov.au/stac/"
    )
    # Using functional form (recommended)
    dp_pystac_client = dp.search_for_pystac_item(
        catalog_url="https://explorer.sandbox.dea.ga.gov.au/stac/"
    )

    assert len(dp_pystac_client) == 1
    it = iter(dp_pystac_client)
    stac_item_search = next(it)
    assert stac_item_search.client.title == "AWS Explorer"
    assert stac_item_search.matched() == 2

    stac_items = list(stac_item_search.items())
    stac_item = stac_items[0]

    assert stac_item.bbox == [
        149.965907628116,
        -35.199398016548095,
        152.10531016837078,
        -32.972806586656844,
    ]
    assert stac_item.datetime.isoformat() == "2001-07-02T00:00:00+00:00"
    assert stac_item.geometry["type"] == "Polygon"
    assert stac_item.properties == {
        "title": "NIDEM_104_151.29_-34.22",
        "created": "2018-10-15T10:00:00Z",
        "proj:epsg": 4326,
        "datetime": "2001-07-02T00:00:00Z",
        "cubedash:region_code": None,
    }
    assert stac_item.assets["nidem"].extra_fields["eo:bands"] == [{"name": "nidem"}]
