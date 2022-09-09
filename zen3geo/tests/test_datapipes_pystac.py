"""
Tests for pystac datapipes.
"""
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import PySTACItemReader

pystac = pytest.importorskip("pystac")

# %%
def test_pystac_item_reader():
    """
    Ensure that PySTACItemReader works to read in a JSON STAC item and outputs
    to a pystac.Item object.
    """
    item_url: str = "https://github.com/stac-utils/pystac/raw/v1.6.1/tests/data-files/item/sample-item.json"
    dp = IterableWrapper(iterable=[item_url])

    # Using class constructors
    dp_pystac = PySTACItemReader(source_datapipe=dp)
    # Using functional form (recommended)
    dp_pystac = dp.read_to_pystac_item()

    assert len(dp_pystac) == 1
    it = iter(dp_pystac)
    stac_item = next(it)

    assert stac_item.bbox == [-122.59750209, 37.48803556, -122.2880486, 37.613537207]
    assert stac_item.datetime.isoformat() == "2016-05-03T13:22:30.040000+00:00"
    assert stac_item.geometry["type"] == "Polygon"
    assert stac_item.properties == {
        "datetime": "2016-05-03T13:22:30.040000Z",
        "title": "A CS3 item",
        "license": "PDDL-1.0",
        "providers": [
            {
                "name": "CoolSat",
                "roles": ["producer", "licensor"],
                "url": "https://cool-sat.com/",
            }
        ],
    }
    assert (
        stac_item.assets["analytic"].extra_fields["product"]
        == "http://cool-sat.com/catalog/products/analytic.json"
    )
