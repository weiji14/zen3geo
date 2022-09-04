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

# Object detection boxes

> *You shouldn't set up limits in boundless openness,
> but if you set up limitlessness as boundless openness,
> you've trapped yourself*

Boxes are quick to draw ✏️, but finicky to train a neural network with.
This time, we'll show you a geospatial object detection 🕵️ problem, where the
objects are defined by a bounding box 🔲 with a specific class.
By the end of this lesson, you should be able to:

- Read OGR supported vector files and obtain the bounding boxes 🟨 of each
  geometry
- Convert bounding boxes from geographic coordinates to 🖼️ image coordinates
  while clipping to the image extent
- Use an affine transform to convert boxes in image coordinates to 🌐
  geographic coordinates

🔗 Links:
- https://planetarycomputer.microsoft.com/dataset/ms-buildings#Example-Notebook
- https://github.com/microsoft/GlobalMLBuildingFootprints/
- https://mlhub.earth/datasets?tags=object+detection

## 🎉 **Getting started**

These are the tools 🛠️ you'll need.

```{code-cell}
import contextily
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr
```

## 0️⃣ Find high-resolution imagery and building footprints 🌇

Let's take a look at buildings over
[Kampong Ayer](https://en.wikipedia.org/wiki/Kampong_Ayer), Brunei 🇧🇳! We'll
use {py:func}`contextily.bounds2img` to get some 4-band RGBA
🌈 [optical imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)
in a {py:class}`numpy.ndarray` format.

```{code-cell}
image, extent = contextily.bounds2img(
    w=114.94,
    s=4.88,
    e=114.95,
    n=4.89,
    ll=True,
    source=contextily.providers.Esri.WorldImagery,
)
print(f"Spatial extent in EPSG:3857: {extent}")
print(f"Image dimensions (height, width, channels): {image.shape}")
```

This is how Brunei's 🚣 Venice of the East looks like from above.

```{code-cell}
fig, ax = plt.subplots(nrows=1, figsize=(9, 9))
plt.imshow(X=image, extent=extent)
```

```{tip}
For more raster basemaps, check out:
- https://xyzservices.readthedocs.io/en/stable/introduction.html#overview-of-built-in-providers
- https://leaflet-extras.github.io/leaflet-providers/preview/
```

### Georeference image using rioxarray 🌐

To enable slicing 🔪 with xbatcher later, we'll need to turn the
{py:class}`numpy.ndarray` image 🖼️ into a {py:class}`xarray.DataArray` grid
with coordinates 🖼️. If you already have a georeferenced grid (e.g. from
{py:class}`zen3geo.datapipes.RioXarrayReader`), this step can be skipped ⏭️.


```{code-cell}
# Turn RGBA image from channel-last to channel-first and get 3-band RGB only
_image = image.transpose(2, 0, 1)  # Change image from (H, W, C) to (C, H, W)
rgb_image = _image[0:3, :, :]  # Get just RGB by dropping RGBA's alpha channel
print(f"RGB image shape: {rgb_image.shape}")
```

Georeferencing is done by putting the 🚦 RGB image into an
{py:class}`xarray.DataArray` object with (band, y, x) coordinates, and then
setting a coordinate reference system 📐 using
{py:meth}`rioxarray.rioxarray.XRasterBase.set_crs`.

```{code-cell}
left, right, bottom, top = extent  # xmin, xmax, ymin, ymax
dataarray = xr.DataArray(
    data=rgb_image,
    coords=dict(
        band=[0, 1, 2],  # Red, Green, Blue
        y=np.linspace(start=top, stop=bottom, num=rgb_image.shape[1]),
        x=np.linspace(start=left, stop=right, num=rgb_image.shape[2]),
    ),
    dims=("band", "y", "x"),
)
dataarray = dataarray.rio.set_crs(input_crs="EPSG:3857")
dataarray
```

### Load cloud-native vector files 💠

Now to pull in some building footprints 🛖. Let's make a STAC API query to get
a [GeoParquet](https://github.com/opengeospatial/geoparquet) file (a
cloud-native columnar 🀤 geospatial vector file format) over our study area.

```{code-cell}
catalog = pystac_client.Client.open(
    url="https://planetarycomputer.microsoft.com/api/stac/v1"
)
items = catalog.search(
    collections=["ms-buildings"], query={"msbuildings:region": {"eq": "Brunei"}}
)
item = next(items.get_items())
item
```

Next, we'll sign 🔏 the URL to the STAC Item Asset, and load ⤵️ the GeoParquet
file using {py:func}`geopandas.read_parquet`.

```{code-cell}
asset = planetary_computer.sign(item.assets["data"])

geodataframe = gpd.read_parquet(
    path=asset.href, storage_options=asset.extra_fields["table:storage_options"]
)
geodataframe
```

This {py:class}`geopandas.GeoDataFrame` contains building outlines across
Brunei 🇧🇳. Let's do a spatial subset ✂️ to just the Kampong Ayer study area
using {py:attr}`geopandas.GeoDataFrame.cx`, and reproject it using
{py:meth}`geopandas.GeoDataFrame.to_crs` to match the coordinate reference
system of the optical image.

```{code-cell}
_gdf_kpgayer = geodataframe.cx[114.94:114.95, 4.88:4.89]
gdf_kpgayer = _gdf_kpgayer.to_crs(crs="EPSG:3857")
gdf_kpgayer
```

Preview 👀 the building footprints to check that things are in the right place.

```{code-cell}
ax = gdf_kpgayer.plot(figsize=(9, 9))
contextily.add_basemap(
    ax=ax,
    source=contextily.providers.Stamen.TonerLite,
    crs=gdf_kpgayer.crs.to_string(),
)
ax
```

Hmm, seems like the Stamen basemap doesn't know the buildings are on water 😂.
