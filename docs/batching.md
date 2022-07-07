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

# Batching data

Following on from the previous tutorial,
let's learn more about creating a more complicated raster data pipeline.
Specifically, we'll go through the following:
- Loading Cloud-Optimized GeoTIFFs (COGs) from different geographic projections
- Cut up each large GeoTIFF into several non-overlapping chips
- Create batches of chips/tensors to feed into a DataLoader

Some terminology disambiguation:
- scene - the big image (e.g. 10000x10000 pixels) from a satellite (e.g. a GeoTIFF)
- chip - the small image (e.g. 512x512 pixels) cut out from a satellite scene to be loaded as a tensor

See also:
- https://github.com/microsoft/torchgeo/wiki/Design-Decisions#chip-vs-tile-vs-region
- https://github.com/cogeotiff/cog-spec/blob/master/spec.md

## 🎉 **Getting started**

Load up them libraries!

```{code-cell}
import pystac
import planetary_computer
import rioxarray

import torchdata
import zen3geo
```

## 0️⃣ Find [Cloud-Optimized GeoTIFFs](https://www.cogeo.org) 🗺️

Synthetic-Aperture Radar (SAR) from a [STAC](https://stacspec.org) catalog!
We'll get some Sentinel-1 Ground-Range Detected (GRD) data over Osaka and Tokyo
in Japan 🇯🇵.

🔗 Links:
- [Official Sentinel-1 description page at ESA](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)
- [Microsoft Planetary Computer STAC Explorer](https://planetarycomputer.microsoft.com/explore?c=137.4907%2C35.0014&z=7.94&v=2&d=sentinel-1-grd&s=false%3A%3A100%3A%3Atrue&ae=0&m=cql%3A08211c0dd907a5066c41422c75629d5f&r=VV%2C+VH+False-color+composite)
- [AWS Sentinel-1 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-1)


```{code-cell}
item_urls = [
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20220614T210034_20220614T210059_043664_05368A",  # Osaka
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20220616T204349_20220616T204414_043693_053764",  # Tokyo
]

# Load each STAC item's metadata and sign the assets
items = [pystac.Item.from_file(item_url) for item_url in item_urls]
signed_items = [planetary_computer.sign(item) for item in items]
signed_items
```

### Inspect one of the data assets 🍱

The Sentinel-1 STAC item contains several assets.
These include different 〰️ polarizations (e.g. 'VH', 'VV').
Let's just use the 'thumbnail' product for now which is an RGB preview, with
the red 🟥 channel (R) representing the co-polarization (VV or HH), the green
🟩 channel (G) representing the cross-polarization (VH or HV) and the blue 🟦
channel (B) representing the ratio of the cross and co-polarizations.

```{code-cell}
url: str = signed_items[0].assets["thumbnail"].href
da = rioxarray.open_rasterio(filename=url)
da
```

This is how the Sentinel-1 radar image looks like over Osaka on 14 June 2022.

![Sentinel-1 image over Osaka, Japan on 20220614](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-1-grd&item=S1A_IW_GRDH_1SDV_20220614T210034_20220614T210059_043664_05368A&assets=vv&assets=vh&expression=vv%2Cvh%2Cvv%2Fvh&rescale=0%2C500&rescale=0%2C300&rescale=0%2C7&tile_format=png)

## 1️⃣ Creating 512x512 chips from large satellite scenes 🗺️

Unless you have a lot of RAM, it is common to cut ✂️ a large satellite scene
into multiple smaller chips (or patches, tiles 🀄, etc) first.
This is typically done in a rolling or sliding window 🪟 fashion,
via a nested loop through the y-dimension and x-dimension in strides of say,
512 pixels x 512 pixels.

Let's begin by setting up the first part of the DataPipe,
which is to read the satellite scene 🖼️ using `rioxarray`.

```{code-cell}
# Just get the VV polarization for now from Sentinel-1
urls = [item.assets["vv"].href for item in signed_items]
dp = torchdata.datapipes.iter.IterableWrapper(iterable=urls)
dp_rioxarray = dp.read_from_rioxarray(overview_level=3)
dp_rioxarray
```

### Slicing with XbatcherSlicer 🍕

To create the chips, we'll be using ``xbatcher`` which allows slicing 🔪 of an
n-dimensional datacube along any dimension (e.g. longitude, latitude, time 🕛).
This ``xbatcher`` library is integrated into ☯ ``zen3geo`` as a DataPipe called
{py:class}`zen3geo.datapipes.XbatcherSlicer`, which can be used as follows:

```{code-cell}
dp_xbatcher = dp_rioxarray.slice_with_xbatcher(input_dims={"y": 512, "x": 512})
dp_xbatcher
```

This should give us about 12 chips in total.

```{code-cell}
chips = [chip for chip in dp_xbatcher]
print(f"Number of chips: {len(chips)}")
```

Now, if you want to customize the sliding window (e.g. do overlapping strides),
pass in extra parameters to ``slice_with_xbatcher``, and it will be handled by
{py:class}`xbatcher.BatchGenerator`.

```{code-cell}
dp_xbatcher = dp_rioxarray.slice_with_xbatcher(
        input_dims={"y": 512, "x": 512}, input_overlap={"y": 256, "x": 256}
)
dp_xbatcher
```

Great, and this overlapping stride method should give us more 512x512 chips 🧮
than before.

```{code-cell}
chips = [chip for chip in dp_xbatcher]
print(f"Number of chips: {len(chips)}")
```

Double-check that single chips are of the correct dimensions
(band: 1, y: 512, x: 512).

```{code-cell}
chips[0]
```
