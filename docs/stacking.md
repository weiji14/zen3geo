---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Stacking layers

> Do not see them differently
>
> Do not consider all as the same
>
> Unwaveringly, practice guarding the One

In Geographic Information Systems 🌐, geographic data is arranged as different
'layers' 🍰. For example:

- Multispectral or hyperspectral 🌈 optical satellites collect different
  radiometric **bands** from slices along the electromagnetic spectrum
- Synthetic Aperture Radar (SAR) 📡 sensors have different **polarizations**
  such as HH, HV, VH & VV
- Satellite laser and radar altimeters 🛰️ measure **elevation** which can be
  turned into a Digital Elevation Model (DEM)

As long as these layers are georeferenced 📍 though, they can be stacked! This
tutorial will cover the following topics:

- Searching for spatiotemporal datasets in a
  [dynamic STAC Catalog](https://stacspec.org/en/about/stac-spec) 📚
- Stacking time-series 📆 data into a 4D tensor of shape (time, channel, y, x)
- Organizing different geographic 🗺️ layers into a dataset suitable for change
  detection


## 🎉 **Getting started**

Load up them libraries!

```{code-cell}
import planetary_computer
import pystac
import torchdata
import zen3geo
```

## 0️⃣ Search for spatiotemporal data 📅

This time, we'll be looking at change detection using time-series data. The
focus area is [Gunung Talamau](https://ban.wikipedia.org/wiki/Gunung_Talamau),
Sumatra Barat, Indonesia 🇮🇩 where an
[earthquake on 25 Feb 2022](https://id.wikipedia.org/wiki/Gempa_bumi_Pasaman_Barat_2022)
triggered a series of landslides ⛰️. Affected areas will be mapped using
Sentinel-1 Ground-Range Detected (GRD) SAR data 📡 obtained via a
spatiotemporal query to a [STAC](https://stacspec.org) API.

🔗 Links:
- https://unosat.org/services
- [UNOSAT satellite-detected landslide maps](https://unosat.org/products/3064)
- [Microsoft Planetary Computer STAC Explorer](https://planetarycomputer.microsoft.com/explore?c=99.9822%2C0.0563&z=11.34&v=2&d=sentinel-1-grd&m=cql%3Ac3f87a557aa4e237d4820f413f9d33d8&r=VV%2C+VH+False-color+composite&s=false%3A%3A100%3A%3Atrue&ae=0)

This is how the Sentinel-1 radar image looks like over Sumatra Barat, Indonesia
on 23 February 2022, two days before the earthquake.

![Sentinel-1 image over Sumatra Barat, Indonesia on 20220223](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-1-grd&item=S1A_IW_GRDH_1SDV_20220223T114141_20220223T114206_042039_0501F9&assets=vv&assets=vh&expression=vv%2Cvh%2Cvv%2Fvh&rescale=0%2C500&rescale=0%2C300&rescale=0%2C7&tile_format=png)

### Sentinel-1 PolSAR time-series ⏳

To start, let’s define an 🧭 area of interest and 📆 time range covering one
month before and one month after the earthquake ⚠️.

```{code-cell}
# Spatiotemporal query on STAC catalog for Sentinel-1 SAR data
query = dict(
    bbox=[99.8, -0.24, 100.07, -0.15],  # West, South, North, East
    datetime=["2022-01-25T00:00:00Z", "2022-03-25T23:59:59Z"],
    collections=["sentinel-1-grd"],
)
dp = torchdata.datapipes.iter.IterableWrapper(iterable=[query])
```

Then, search over a dynamic STAC Catalog 📚 for items matching the
spatiotemporal query ❔ using
{py:class}`zen3geo.datapipes.PySTACAPISearch` (functional name:
`search_for_pystac_item`).

```{code-cell}
dp_pystac_client = dp.search_for_pystac_item(
    catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
```

```{tip}
Confused about which parameters go where 😕? Here's some clarification:

1. **Different** spatiotemporal queries (e.g. for multiple geographical areas)
   should go in {py:class}`torchdata.datapipes.iter.IterableWrapper`, e.g.
   `IterableWrapper(iterable=[query_area1, query_area2])`. The query
   dictionaries will be passed to {py:meth}`pystac_client.Client.search`.
2. **Common** parameters to interact with the STAC API Client should go in
   [`search_for_pystac_item()`](zen3geo.datapipes.PySTACAPISearch), e.g. the
   STAC API's URL (see https://stacindex.org/catalogs?access=public&type=api
   for a public list) and connection related parameters. These will be passed
   to {py:meth}`pystac_client.Client.open`.
```

The output is a {py:class}`pystac_client.ItemSearch` 🔎 instance that only
holds the STAC API query information ℹ️ but doesn't request for data! We'll
need to order it 🧞 to return something like a
{py:class}`pystac.ItemCollection`.

```{code-cell}
def get_all_items(item_search) -> pystac.ItemCollection:
    return item_search.item_collection()
```

```{code-cell}
dp_sen1_items = dp_pystac_client.map(fn=get_all_items)
dp_sen1_items
```

Take a peek 🫣 to see if the query does contain STAC items.

```{code-cell}
it = iter(dp_sen1_items)
item_collection = next(it)
item_collection.items
```

### Copernicus Digital Elevation Model (DEM) ⛰️

Since landslides 🛝 typically happen on steep slopes, it can be useful to have
a 🏔️ topographic layer. Let's set up a STAC query 🙋 to get the
30m spatial resolution [Copernicus DEM](https://doi.org/10.5069/G9028PQB).

```{code-cell}
# Spatiotemporal query on STAC catalog for Copernicus DEM 30m data
query = dict(
    bbox=[99.8, -0.24, 100.07, -0.15],  # West, South, North, East
    collections=["cop-dem-glo-30"],
)
dp_copdem30 = torchdata.datapipes.iter.IterableWrapper(iterable=[query])
```

Just to be fancy, let's chain 🔗 together the next two datapipes.

```{code-cell}
dp_copdem30_items = dp_copdem30.search_for_pystac_item(
    catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
).map(fn=get_all_items)
dp_copdem30_items
```

This query yields the following DEM tile 🀫.

![Copernicus 30m DEM over Sumatra Barat, Indonesia](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=cop-dem-glo-30&item=Copernicus_DSM_COG_10_N00_00_E099_00_DEM&assets=data&colormap_name=terrain&rescale=-1000%2C4000)


## 1️⃣ Stack bands, append variables 📚

Each {py:class}`pystac.Item` in a {py:class}`pystac.ItemCollection` represents
a 🛰️ Sentinel-1 GRD image captured at a particular datetime ⌚. Let's stack 🥞
all of the items into 4D time-series tensor using
{py:class}`zen3geo.datapipes.StackSTACStacker` (functional name:
`stack_stac_items`)!

```{code-cell}
dp_sen1_stack = dp_sen1_items.stack_stac_items(
    assets=["vh", "vv"],  # SAR polarizations
    epsg=32647,  # UTM Zone 47N
    resolution=10,  # Spatial resolution of 10 metres
)
dp_sen1_stack
```

The result is a single {py:class}`xarray.DataArray` 'datacube' 🧊 with
dimensions (time, band, y, x).

```{code-cell}
it = iter(dp_sen1_stack)
dataarray = next(it)
dataarray
```
