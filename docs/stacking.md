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

## 0️⃣ Search for time-series geospatial data 📅

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


To start, let’s define an 🧭 area of interest and 📆 time range.

```{code-cell}
# Spatiotemporal query on STAC catalog for Sentinel-1 SAR data
query = dict(
    bbox=[99.8, -0.24, 100.07, -0.15],  # West, South, North, East
    datetime=["2022-01-04T00:00:00Z", "2022-04-24T23:59:59Z"],
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

The output is a {py:class}`pystac_client.ItemSearch` 🔎 instance that only
holds the STAC API query information ℹ️ but doesn't request for data! We'll
need to order it to return something like a {py:class}`pystac.ItemCollection`.

```{code-cell}
def get_all_items(item_search) -> pystac.ItemCollection:
    return item_search.item_collection()
```

```{code-cell}
dp_stac_items = dp_pystac_client.map(fn=get_all_items)
dp_stac_items
```

Each {py:class}`pystac.Item` in a {py:class}`pystac.ItemCollection` represents
a 🛰️ Sentinel-1 GRD image captured at a particular datetime ⌚. Let's stack 🥞
all of the items into 4D time-series tensor using
{py:class}`zen3geo.datapipes.StackSTACStacker` (functional name:
`stack_stac_items`)!

```{code-cell}
dp_stackstac = dp_stac_items.stack_stac_items(
    assets=["vh", "vv"],  # SAR polarizations
    epsg=32647,  # UTM Zone 47N
    resolution=10,  # Spatial resolution of 10 metres
)
dp_stackstac
```

The result is a single {py:class}`xarray.DataArray` 'datacube' 🧊 with
dimensions (time, band, y, x).

```{code-cell}
it = iter(dp_stackstac)
dataarray = next(it)
dataarray
```
