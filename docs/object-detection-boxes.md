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
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray
import shapely.affinity
import shapely.geometry
import torch
import torchdata
import torchdata.dataloader2
import xarray as xr
import zen3geo
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
{py:class}`numpy.ndarray` image 🖼️ into an {py:class}`xarray.DataArray` grid
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
dataarray = dataarray.rio.write_crs(input_crs="EPSG:3857")
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

Hmm, seems like the Stamen basemap doesn't know that some of the buildings are
on water 😂.


## 1️⃣ Pair image chips with bounding boxes 🧑‍🤝‍🧑

Here comes the fun 🛝 part! This section is all about generating 128x128 chips
🫶 paired with bounding boxes. Let's go 🚲!

### Create 128x128 raster chips and clip vector geometries with it ✂️

From the large 1280x1280 scene 🖽️, we will first slice out a hundred 128x128
chips 🍕 using {py:class}`zen3geo.datapipes.XbatcherSlicer` (functional name:
`slice_with_xbatcher`).

```{code-cell}
dp_raster = torchdata.datapipes.iter.IterableWrapper(iterable=[dataarray])
dp_xbatcher = dp_raster.slice_with_xbatcher(input_dims={"y": 128, "x": 128})
dp_xbatcher
```

For each 128x128 chip 🍕, we'll then find the vector geometries 🌙 that fit
within the chip's spatial extent. This will be 🤸 done using
{py:class}`zen3geo.datapipes.GeoPandasRectangleClipper` (functional name:
`clip_vector_with_rectangle`).

```{code-cell}
dp_vector = torchdata.datapipes.iter.IterableWrapper(iterable=[gdf_kpgayer])
dp_clipped = dp_vector.clip_vector_with_rectangle(mask_datapipe=dp_xbatcher)
dp_clipped
```

```{important}
When using {py:class}`zen3geo.datapipes.GeoPandasRectangleClipper` 💇, there
should only be one 'global' 🌐 vector {py:class}`geopandas.GeoSeries` or
{py:class}`geopandas.GeoDataFrame`.

If your raster DataPipe has chips 🍕 with different coordinate reference
systems (e.g. multiple UTM Zones 🌏🌍🌎),
{py:class}`zen3geo.datapipes.GeoPandasRectangleClipper` will actually reproject
🔄 the 'global' vector to the coordinate reference system of each chip, and
clip ✂️ the geometries accordingly to the chip's bounding box extent 😎.
```

This ``dp_clipped`` DataPipe will yield 🤲 a tuple of ``(vector, raster)``
objects for each 128x128 chip. Let's inspect 🧐 one to see how they look like.

```{code-cell}
# Get one chip with over 10 building footprint geometries
for vector, raster in dp_clipped:
    if len(vector) > 10:
        break
```

These are the spatially subsetted vector geometries 🌙 in one 128x128 chip.

```{code-cell}
vector
```

This is the raster chip/mask 🤿 used to clip the vector.

```{code-cell}
raster
```

And here's a side by side visualization of the 🌈 RGB chip image (left) and
🔷 vector building footprint polygons (right).

```{code-cell}
fig, ax = plt.subplots(ncols=2, figsize=(18, 9), sharex=True, sharey=True)
raster.plot.imshow(ax=ax[0])
vector.plot(ax=ax[1])
```

Cool, these buildings are part of the 🏬
[Yayasan Shopping Complex](https://web.archive.org/web/20220906020248/http://www.yayasancomplex.com)
in Bandar Seri Begawan 🌆. We can see that the raster image 🖼️ on the left
aligns ok with the vector polygons 💠 on the right.

```{note}
The optical 🛰️ imagery shown here is **not** the imagery used to digitize the
[building footprints](https://planetarycomputer.microsoft.com/dataset/ms-buildings)
🏢! This is an example tutorial using two different data sources, that we just
so happened to have plotted in the same geographic space 😝.
```

### From polygons in geographic coordinates to boxes in image coordinates ↕️

Up to this point, we still have the actual 🛖 building footprint polygons. In
this step 📶, we'll convert these polygons into a format suitable for 'basic'
object detection 🥅 models in computer vision. Specifically:

1. The polygons 🌙 (with multiple vertices) will be simplified to a horizontal
   bounding box 🔲 with 4 corner vertices only.
2. The 🌐 geographic coordinates of the box which use lower left corner and
   upper right corner (i.e. y increases from South to North ⬆️) will be
   converted to 🖼️ image coordinates (0-128) which use the top left corner and
   bottom right corner (i.e y increases from Top to Bottom ⬇️).

Let's start by using {py:attr}`geopandas.GeoSeries.bounds` to get the
geographic bounds 🗺️ of each building footprint geometry 📐 in each 128x128
chip.

```{code-cell}
def polygon_to_bbox(geom_and_chip) -> (gpd.GeoDataFrame, xr.DataArray):
    """
    Get bounding box (minx, miny, maxx, maxy) coordinates for each geometry in
    a geopandas.GeoDataFrame.

                          (maxx,maxy)
               ul-------ur
             ^  |       |
             |  |  geo  |    y increases going up, x increases going right
             y  |       |
               ll-------lr
    (minx,miny)    x-->

    """
    gdf, chip = geom_and_chip
    bounds: gpd.GeoDataFrame = gdf.bounds
    assert tuple(bounds.columns) == ("minx", "miny", "maxx", "maxy")

    return bounds, chip
```

```{code-cell}
dp_bbox = dp_clipped.map(fn=polygon_to_bbox)
```

Next, the geographic 🗺️ bounding box coordinates (in EPSG:3857) will be
converted to image 🖼️ or pixel coordinates (0-128 scale). The y-direction will
be flipped 🤸 upside down, and we'll be using the spatial bounds (or corner
coordinates) of the 128x128 image chip as a reference 📍.

```{code-cell}
def geobox_to_imgbox(bbox_and_chip) -> (pd.DataFrame, xr.DataArray):
    """
    Convert bounding boxes in a pandas.DataFrame from geographic coordinates
    (minx, miny, maxx, maxy) to image coordinates (x1, y1, x2, y2) based on the
    spatial extent of a raster image chip.

        (x1,y1)
               ul-------ur
             y  |       |
             |  |  img  |    y increases going down, x increases going right
             v  |       |
               ll-------lr
                   x-->    (x2,y2)

    """
    geobox, chip = bbox_and_chip

    x_res, y_res = chip.rio.resolution()
    assert y_res < 0

    left, bottom, right, top = chip.rio.bounds()
    assert top > bottom

    imgbox = pd.DataFrame()
    imgbox["x1"] = (geobox.minx - left) / x_res  # left
    imgbox["y1"] = (top - geobox.maxy) / -y_res  # top
    imgbox["x2"] = (geobox.maxx - left) / x_res  # right
    imgbox["y2"] = (top - geobox.miny) / -y_res  # bottom

    assert all(imgbox.x2 > imgbox.x1)
    assert all(imgbox.y2 > imgbox.y1)

    return imgbox, chip
```

```{code-cell}
dp_ibox = dp_bbox.map(fn=geobox_to_imgbox)
```

Now to plot 🎨 and double check that the boxes are positioned correctly in
0-128 image space 🌌.

```{code-cell}
# Get one chip with over 10 building footprint geometries
for ibox, ichip in dp_ibox:
    if len(ibox) > 10:
        break
ibox
```

```{code-cell}
fig, ax = plt.subplots(ncols=2, figsize=(18, 9), sharex=True, sharey=True)
ax[0].imshow(X=ichip.transpose("y", "x", "band"))
for i, row in ibox.iterrows():
    rectangle = matplotlib.patches.Rectangle(
        xy=(row.x1, row.y1),
        width=row.x2 - row.x1,
        height=row.y2 - row.y1,
        edgecolor="blue",
        linewidth=1,
        facecolor="none",
    )
    ax[1].add_patch(rectangle)
```

Cool, the 🟦 bounding boxes on the right subplot are correctly positioned 🧭
(compare it with the figure in the previous subsection).

```{hint}
Instead of a bounding box 🥡 object detection task, you can also use the
building polygons 🏘️ for a segmentation task 🧑‍🎨 following
{doc}`./vector-segmentation-masks`.

If you still prefer doing object detection 🕵️, but want a different box format
(see options in {py:func}`torchvision.ops.box_convert`),
like 🎌 centre-based coordinates with width and height (`cxcywh`), or
📨 oriented/rotated bounding box coordinates, feel free to implement your own
function and DataPipe for it 🤗!
```


## 2️⃣ There and back again 🧙

What follows on from here requires focus 🤫. To start, we'll pool the hundred
💯 128x128 chips into 10 batches (10 chips per batch) using
{py:class}`torchdata.datapipes.iter.Batcher` (functional name: `batch`).

```{code-cell}
dp_batch = dp_ibox.batch(batch_size=10)
print(f"Number of items in first batch: {len(list(dp_batch)[0])}")
```

### Batch boxes with variable lengths 📏

Next, we'll stack 🥞 all the image chips into a single tensor (recall
{doc}`./chipping`), and concatenate 📚 the bounding boxes into a list of
tensors using {py:class}`torchdata.datapipes.iter.Collator` (functional name:
`collate`).

```{code-cell}
def boximg_collate_fn(samples) -> (list[torch.Tensor], torch.Tensor, list[dict]):
    """
    Converts bounding boxes and raster images to tensor objects and keeps
    geographic metadata (spatial extent, coordinate reference system and
    spatial resolution).

    Specifically, the bounding boxes in pandas.DataFrame format are each
    converted to a torch.Tensor and collated into a list, while the raster
    images in xarray.DataArray format are converted to a torch.Tensor (int16
    dtype) and stacked into a single torch.Tensor.
    """
    box_tensors: list[torch.Tensor] = [
        torch.as_tensor(sample[0].to_numpy(dtype=np.float32)) for sample in samples
    ]

    tensors: list[torch.Tensor] = [
        torch.as_tensor(data=sample[1].data.astype(dtype="int16")) for sample in samples
    ]
    img_tensors = torch.stack(tensors=tensors)

    metadata: list[dict] = [
        {
            "bbox": sample[1].rio.bounds(),
            "crs": sample[1].rio.crs,
            "resolution": sample[1].rio.resolution(),
        }
        for sample in samples
    ]

    return box_tensors, img_tensors, metadata
```

```{code-cell}
dp_collate = dp_batch.collate(collate_fn=boximg_collate_fn)
print(f"Number of mini-batches: {len(dp_collate)}")
mini_batch_box, mini_batch_img, mini_batch_metadata = list(dp_collate)[1]
print(f"Mini-batch image tensor shape: {mini_batch_img.shape}")
print(f"Mini-batch box tensors: {mini_batch_box}")
print(f"Mini-batch metadata: {mini_batch_metadata}")
```

The DataPipe is complete 🙌, let's visualize the entire data pipeline graph.

```{code-cell}
torchdata.datapipes.utils.to_graph(dp=dp_collate)
```

### Into a DataLoader 🏋️

Loop over the DataPipe using {py:class}`torch.utils.data.DataLoader` ⚙️!

```{code-cell}
dataloader = torchdata.dataloader2.DataLoader2(datapipe=dp_collate)
for i, batch in enumerate(dataloader):
    box, img, metadata = batch
    print(f"Batch {i} - img: {img.shape}, box sizes: {[len(b) for b in box]}")
```

There's probably hundreds of models you can 🍜 feed this data into, from
mmdetection's {doc}`mmdetection:model_zoo` 🐼 to torchvision's
{doc}`torchvision:models`). But are we out of the woods yet?

### Georeference image boxes 📍

To turn the model's predicted bounding boxes in image space 🌌 back to
geographic coordinates 🌐, you'll need to use an
[affine transform](https://web.archive.org/web/20210506173651/https://www.perrygeo.com/python-affine-transforms.html).
Assuming you've kept your 🏷️ metadata intact, here's an example on how to do
the georeferencing:

```{code-cell}
for batch in dataloader:
    pred_boxes, images, metadata = batch

    objs: list = []
    for idx in range(0, len(images)):
        left, bottom, right, top = metadata[idx]["bbox"]
        crs = metadata[idx]["crs"]
        x_res, y_res = metadata[idx]["resolution"]

        gdf = gpd.GeoDataFrame(
            geometry=[
                shapely.affinity.affine_transform(
                    geom=shapely.geometry.box(*coords),
                    matrix=[x_res, 0, 0, y_res, left, top],
                )
                for coords in pred_boxes[idx]
            ],
            crs=crs,
        )
        objs.append(gdf.to_crs(crs=crs))

    geodataframe: gpd.GeoDataFrame = pd.concat(objs=objs, ignore_index=True)
    geodataframe.set_crs(crs=crs, inplace=True)
    break

geodataframe
```

Back at square one, or are we?
