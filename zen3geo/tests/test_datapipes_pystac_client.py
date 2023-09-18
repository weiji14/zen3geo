"""
Tests for pystac-client datapipes.
"""
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import PySTACAPIItemLister, PySTACAPISearcher

pystac_client = pytest.importorskip("pystac_client")


# %%
def test_pystac_client_item_search():
    """
    Ensure that PySTACAPISearcher works to query a STAC API /search/ endpoint
    and outputs a pystac_client.ItemSearch object.
    """
    query: dict = dict(
        bbox=[150.9, -34.36, 151.3, -33.46],
        datetime=["2000-01-01T00:00:00Z", "2020-12-31T00:00:00Z"],
        collections=["nidem"],
    )
    dp = IterableWrapper(iterable=[query])

    # Using class constructors
    dp_pystac_client = PySTACAPISearcher(
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


def test_pystac_client_item_search_open_headers():
    """
    Ensure that PySTACAPISearcher works to query a STAC API /search/ endpoint
    with headers passed to pystac_client.Client.open.
    """
    query: dict = dict(
        bbox=[150.9, -34.36, 151.3, -33.46],
        datetime=["2020-01-01T00:00:00Z", "2022-12-31T00:00:00Z"],
        collections=["HLSS30.v2.0"],
    )
    dp = IterableWrapper(iterable=[query])

    # Using class constructors
    dp_pystac_client = PySTACAPISearcher(
        source_datapipe=dp,
        catalog_url="https://cmr.earthdata.nasa.gov/cloudstac/LPCLOUD",
        headers={"Authorization": "Bearer <EDL_TOKEN>"},
    )
    # Using functional form (recommended)
    dp_pystac_client = dp.search_for_pystac_item(
        catalog_url="https://cmr.earthdata.nasa.gov/cloudstac/LPCLOUD",
        headers={"Authorization": "Bearer <EDL_TOKEN>"},
    )

    assert len(dp_pystac_client) == 1
    it = iter(dp_pystac_client)
    stac_item_search = next(it)
    assert stac_item_search.client.title == "LPCLOUD"
    assert stac_item_search.client.description == "Root catalog for LPCLOUD"


def test_pystac_client_item_lister():
    """
    Ensure that PySTACAPIItemLister works to yield pystac.Item instances for
    each item matching the given search parameters in a
    pystac_client.ItemSearch query.
    """
    catalog = pystac_client.Client.open(
        url="https://earth-search.aws.element84.com/v1/"
    )
    search = catalog.search(
        bbox=[134.2, 6.9, 134.8, 8.5],
        datetime=["2023-01-01T00:00:00Z", "2023-01-31T00:00:00Z"],
        collections=["sentinel-2-l1c"],
    )
    dp = IterableWrapper(iterable=[search])

    # Using class constructors
    dp_pystac_item_list = PySTACAPIItemLister(source_datapipe=dp)
    # Using functional form (recommended)
    dp_pystac_item_list = dp.list_pystac_items_by_search()

    assert len(dp_pystac_item_list) == 14
    it = iter(dp_pystac_item_list)
    stac_item = next(it)
    assert stac_item.bbox == [
        134.093840347073,
        6.2442879900058115,
        135.08840137750929,
        7.237809826458827,
    ]
    assert stac_item.datetime.isoformat() == "2023-01-29T01:35:24.640000+00:00"
    assert stac_item.geometry["type"] == "Polygon"
    assert stac_item.properties == {
        "created": "2023-01-29T06:01:33.679Z",
        "platform": "sentinel-2b",
        "constellation": "sentinel-2",
        "instruments": ["msi"],
        "eo:cloud_cover": 92.7676417582305,
        "proj:epsg": 32653,
        "mgrs:utm_zone": 53,
        "mgrs:latitude_band": "N",
        "mgrs:grid_square": "MH",
        "grid:code": "MGRS-53NMH",
        "view:sun_azimuth": 135.719785438016,
        "view:sun_elevation": 55.1713941690268,
        "s2:degraded_msi_data_percentage": 0.2816,
        "s2:product_type": "S2MSI1C",
        "s2:processing_baseline": "05.09",
        "s2:product_uri": "S2B_MSIL1C_20230129T013449_N0509_R031_T53NMH_20230129T025811.SAFE",
        "s2:generation_time": "2023-01-29T02:58:11.000000Z",
        "s2:datatake_id": "GS2B_20230129T013449_030802_N05.09",
        "s2:datatake_type": "INS-NOBS",
        "s2:datastrip_id": "S2B_OPER_MSI_L1C_DS_2BPS_20230129T025811_S20230129T013450_N05.09",
        "s2:granule_id": "S2B_OPER_MSI_L1C_TL_2BPS_20230129T025811_A030802_T53NMH_N05.09",
        "s2:reflectance_conversion_factor": 1.03193080888673,
        "datetime": "2023-01-29T01:35:24.640000Z",
        "s2:sequence": "0",
        "earthsearch:s3_path": "s3://earthsearch-data/sentinel-2-l1c/53/N/MH/2023/1/S2B_53NMH_20230129_0_L1C",
        "earthsearch:payload_id": "roda-sentinel2/workflow-sentinel2-to-stac/15626e44fb54c2182e5ed5d3aec4a209",
        "processing:software": {"sentinel2-to-stac": "0.1.0"},
        "updated": "2023-01-29T06:01:33.679Z",
    }
    assert stac_item.assets["visual"].extra_fields["eo:bands"] == [
        {
            "name": "red",
            "common_name": "red",
            "description": "Red (band 4)",
            "center_wavelength": 0.665,
            "full_width_half_max": 0.038,
        },
        {
            "name": "green",
            "common_name": "green",
            "description": "Green (band 3)",
            "center_wavelength": 0.56,
            "full_width_half_max": 0.045,
        },
        {
            "name": "blue",
            "common_name": "blue",
            "description": "Blue (band 2)",
            "center_wavelength": 0.49,
            "full_width_half_max": 0.098,
        },
    ]
