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
import pystac_client
import planetary_computer
import geopandas as gpd
import matplotlib.pyplot as plt
```

## 0️⃣ Find high-resolution imagery and building footprints 🌇

Let's take a look at buildings over
[Kampong Ayer](https://en.wikipedia.org/wiki/Kampong_Ayer), Brunei 🇧🇳! We'll
use {py:meth}`contextily.bounds2img` to get some 4-band RGBA
🌈 [optical imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)
in a {py:class}`numpy.ndarray` format.

```{code-cell}
img, extent = contextily.bounds2img(
    w=114.94,
    s=4.88,
    e=114.95,
    n=4.89,
    ll=True,
    source=contextily.providers.Esri.WorldImagery,
)
print(f"Spatial extent in EPSG:3857: {extent}")
print(f"Image dimensions (height, width, channels): {img.shape}")
```

This is how Brunei's 🚣 Venice of the East looks like from above.

```{code-cell}
fig, ax = plt.subplots(nrows=1, figsize=(9, 9))
plt.imshow(X=img, extent=extent)
```

```{tip}
For more raster basemaps, check out:
- https://xyzservices.readthedocs.io/en/stable/introduction.html#overview-of-built-in-providers
- https://leaflet-extras.github.io/leaflet-providers/preview/
```
