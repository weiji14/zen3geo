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

# Vector segmentation masks

> *Clouds float by, water flows on;
> in movement there is no grasping, in Chan there is no settling*

For 🧑‍🏫 supervised machine learning, labels 🏷️ are needed in addition to the
input image 🖼️. Here, we'll step through an example workflow on matching vector
🚏 label data (points, lines, polygons) to 🛰️ Earth Observation data inputs.
Specifically, this tutorial will cover:

- Reading shapefiles 📁 directly from the web via {doc}`pyogrio <pyogrio:index>`
- Rasterizing vector polygons from a {py:class}`geopandas.GeoDataFrame` to an {py:class}`xarray.DataArray`
- Pairing 🛰️ satellite images with the rasterized label masks and feeding them into a DataLoader


## 🎉 **Getting started**

These are the tools 🛠️ you'll need.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pyogrio
import pystac
import torch
import torchdata
import xarray as xr
import zen3geo
```

## 0️⃣ Find cloud-hosted raster and vector data ⛳

In this case study, we'll look at the flood water extent over Johor,
Malaysia 🇲🇾 on 15 Dec 2019 that were digitized by 🇺🇳 UNITAR-UNOSAT's rapid
mapping service over Synthetic Aperture Radar (SAR) 🛰️ images. Specifically,
we'll be using the 🇪🇺 Sentinel-1 Ground Range Detected (GRD) product's VV
polarization channel.

🔗 Links:
- https://www.unitar.org/maps/unosat-rapid-mapping-service
- https://unitar.org/maps/countries
- [Microsoft Planetary Computer STAC Explorer](https://planetarycomputer.microsoft.com/explore?c=103.6637%2C2.1494&z=8.49&v=2&d=sentinel-1-grd&s=false%3A%3A100%3A%3Atrue&ae=0&m=cql%3Afc3d85b6ab43d3e8ebe168da0206f2cf&r=VV%2C+VH+False-color+composite)

To start, let's get the 🛰️ satellite scene we'll be using for this tutorial.

```{code-cell}
item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20191215T224757_20191215T224822_030365_037955"

# Load the individual item metadata and sign the assets
item = pystac.Item.from_file(item_url)
signed_item = planetary_computer.sign(item)
signed_item
```

This is how the Sentinel-1 🩻 image looks like over Johor in Peninsular
Malaysia on 15 Dec 2019.

![Sentinel-1 image over Johor, Malaysia on 20191215](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-1-grd&item=S1A_IW_GRDH_1SDV_20191215T224757_20191215T224822_030365_037955&assets=vv&assets=vh&expression=vv%2Cvh%2Cvv%2Fvh&rescale=0%2C500&rescale=0%2C300&rescale=0%2C7&tile_format=png)

### Load and reproject image data 🔄

To keep things simple, we'll load just the VV channel into a DataPipe via
{py:class}`zen3geo.datapipes.RioXarrayReader` (functional name:
`read_from_rioxarray`) 😀.

```{code-cell}
url = signed_item.assets["vv"].href
dp = torchdata.datapipes.iter.IterableWrapper(iterable=[url])
# Reading lower resolution grid using overview_level=3
dp_rioxarray = dp.read_from_rioxarray(overview_level=3)
dp_rioxarray
```

The Sentinel-1 image from Planetary Computer comes in longitude/latitude 🌐
geographic coordinates by default (OGC:CRS84). To make the pixels more equal 🔲
area, we can project it to a 🌏 local projected coordinate system instead.

```{code-cell}
def reproject_to_local_utm(dataarray: xr.DataArray, resolution: float=80.0) -> xr.DataArray:
    """
    Reproject an xarray.DataArray grid from OGC:CRS84 to a local UTM coordinate
    reference system.
    """
    # Estimate UTM coordinate reference from a single pixel
    pixel = dataarray.isel(y=slice(0, 1), x=slice(0,1))
    new_crs = dataarray.rio.reproject(dst_crs="OGC:CRS84").rio.estimate_utm_crs()

    return dataarray.rio.reproject(dst_crs=new_crs, resolution=resolution)
```

```{code-cell}
dp_reprojected = dp_rioxarray.map(fn=reproject_to_local_utm)
```

```{note}
Universal Transverse Mercator (UTM) isn't actually an equal-area projection
system. However, Sentinel-1 🛰️ satellite scenes from Copernicus are usually
distributed in a UTM coordinate reference system, and UTM is typically a close
enough 🤏 approximation to the local geographic area, or at least it won't
matter much when we're looking at spatial resolutions over several 10s of
metres 🙂.
```

```{hint}
For those wondering what `OGC:CRS84` is, it is the longitude/latitude version
of [`EPSG:4326`](https://epsg.io/4326) 🌐 (latitude/longitude). I.e., it's a
matter of axis order, with `OGC:CRS84` being x/y and `EPSG:4326` being y/x.

🔖 References:
- https://gis.stackexchange.com/questions/54073/what-is-crs84-projection
- https://github.com/opengeospatial/geoparquet/issues/52
```

### Transform and visualize raster data 🔎

Let's visualize 👀 the Sentinel-1 image, but before that, we'll transform 🔄
the VV data from linear to [decibel](https://en.wikipedia.org/wiki/Decibel)
scale.

```{code-cell}
def linear_to_decibel(dataarray: xr.DataArray) -> xr.DataArray:
    """
    Transforming the input xarray.DataArray's VV or VH values from linear to
    decibel scale using the formula ``10 * log_10(x)``.
    """
    # Mask out areas with 0 so that np.log10 is not undefined
    da_linear = dataarray.where(cond=dataarray != 0)
    da_decibel = 10 * np.log10(da_linear)
    return da_decibel
```

```{code-cell}
dp_decibel = dp_reprojected.map(fn=linear_to_decibel)
dp_decibel
```

As an aside, we'll be using the Sentinel-1 image datapipe twice later, once as
a template to create a blank canvas 🎞️, and another time by itself 🪞. This
requires forking 🍴 the DataPipe into two branches, which can be achieved using
{py:class}`torchdata.datapipes.iter.Forker` (functional name: `fork`).

```{code-cell}
dp_decibel_canvas, dp_decibel_image = dp_decibel.fork(num_instances=2)
dp_decibel_canvas, dp_decibel_image
```

Now to visualize the transformed Sentinel-1 image 🖼️. Let's zoom in 🔭 to one
of the analysis extent areas we'll be working on later.

```{code-cell}
it = iter(dp_decibel_image)
dataarray = next(it)

da_clip = dataarray.rio.clip_box(minx=371483, miny=190459, maxx=409684, maxy=229474)
da_clip.isel(band=0).plot.imshow(figsize=(11.5, 9), cmap="Blues_r", vmin=18, vmax=26)
```

Notice how the darker blue areas 🔵 tend to correlate more with water features
like the meandering rivers and the 🐚 sea on the NorthEast. This is because the
SAR 🛰️ signal which is side looking reflects off flat water bodies like a
mirror 🪞, with little energy getting reflected 🙅 back directly to the sensor
(hence why it looks darker ⚫).

### Load and visualize cloud-hosted vector files 💠

Let's now load some vector data from the web 🕸️. These are polygons of the
segmented 🌊 water extent digitized by UNOSAT's AI Based Rapid Mapping Service.
We'll be converting these vector polygons to 🌈 raster masks later.

🔗 Links:
- https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service
- [Humanitarian Data Exchange link to polygon dataset](https://data.humdata.org/dataset/waters-extents-as-of-15-december-2019-over-kota-tinggi-and-mersing-district-johor-state-of)
- [Disaster Risk Monitoring Using Satellite Imagery online course](https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1)

```{code-cell}
# https://gdal.org/user/virtual_file_systems.html#vsizip-zip-archives
shape_url = "/vsizip/vsicurl/https://unosat-maps.web.cern.ch/MY/FL20191217MYS/FL20191217MYS_SHP.zip/ST1_20191215_WaterExtent_Johor_AOI2.shp"
```

This is a shapefile containing 🔷 polygons of the mapped water extent. Let's
put it into a DataPipe called {py:class}`zen3geo.datapipes.PyogrioReader`
(functional name: ``read_from_pyogrio``).

```{code-cell}
dp_shapes = torchdata.datapipes.iter.IterableWrapper(iterable=[shape_url])
dp_pyogrio = dp_shapes.read_from_pyogrio()
dp_pyogrio
```

This will take care of loading the shapefile into a
{py:class}`geopandas.GeoDataFrame` object. Let's take a look at the data table
📊 to see what attributes are inside.

```{code-cell}
it = iter(dp_pyogrio)
geodataframe = next(it)
geodataframe.dropna(axis="columns")
```

Cool, and we can also visualize the polygons 🔷 on a 2D map. To align the
coordinates with the 🛰️ Sentinel-1 image above, we'll first use
{py:meth}`geopandas.GeoDataFrame.to_crs` to reproject the vector from 🌐
EPSG:9707 (WGS 84 + EGM96 height, latitude/longitude) to 🌏 EPSG:32648 (UTM
Zone 48N).

```{code-cell}
print(f"Original bounds in EPSG:9707:\n{geodataframe.bounds}")
gdf = geodataframe.to_crs(crs="EPSG:32648")
print(f"New bounds in EPSG:32648:\n{gdf.bounds}")
```

Plot it with {py:meth}`geopandas.GeoDataFrame.plot`. This vector map 🗺️ should
correspond to the zoomed in Sentinel-1 image plotted earlier above.

```{code-cell}
gdf.plot(figsize=(11.5, 9))
```

```{tip}
Make sure to understand your raster and vector datasets well first! Open the
files up in your favourite 🌐 Geographic Information System (GIS) tool, see how
they actually look like spatially. Then you'll have a better idea to decide on
how to create your data pipeline. The zen3geo way puts you as the Master 🧙 in
control.
```


## 1️⃣ Create a canvas to paint on 🎨

In this section, we'll work on converting the flood water 🌊 polygons above
from a 🚩 vector to a 🌈 raster format, i.e. rasterization. This will be done
in two steps 📶:

1. Defining a blank canvas 🎞️
2. Paint the polygons onto this blank canvas 🧑‍🎨

For this, we'll be using tools from {py:meth}`zen3geo.datapipes.datashader`.
Let's see how this can be done.

### Blank canvas from template raster 🖼️

A canvas represents a 2D area with a height and a width 📏. For us, we'll be
using a {py:class}`datashader.Canvas`, which also defines the range of y-values
(ymin to ymax) and x-values (xmin to xmax), essentially coordinates for
every unit 🇾 height and 🇽 width.

Since we already have a Sentinel-1 🛰️ raster grid with defined height/width
and y/x coordinates, let's use it as a 📄 template to define our canvas. This
is done via {py:class}`zen3geo.datapipes.XarrayCanvas` (functional name:
``canvas_from_xarray``).

```{code-cell}
dp_canvas = dp_decibel_canvas.canvas_from_xarray()
dp_canvas
```

Cool, and here's a quick inspection 👀 of the canvas dimensions and metadata.

```{code-cell}
it = iter(dp_canvas)
canvas = next(it)
print(f"Canvas height: {canvas.plot_height}, width: {canvas.plot_width}")
print(f"Y-range: {canvas.y_range}")
print(f"X-range: {canvas.x_range}")
print(f"Coordinate reference system: {canvas.crs}")
```

This information should match the template the Sentinel-1 dataarray 🏁.

```{code-cell}
print(f"Dimensions: {dict(dataarray.sizes)}")
print(f"Affine transform: {dataarray.rio.transform()}")
print(f"Bounding box: {dataarray.rio.bounds()}")
print(f"Coordinate reference system: {dataarray.rio.crs}")
```

### Rasterize vector polygons onto canvas 🖌️

Now's the time to paint or rasterize the
vector {py:class}`geopandas.GeoDataFrame` polygons 🔷 onto the blank
{py:class}`datashader.Canvas`! This would enable us to have a direct pixel-wise
X -> Y mapping ↔️ between the Sentinel-1 image (X) and target flood label (Y).

The vector polygons can be rasterized or painted 🖌️ onto the template canvas
using {py:class}`zen3geo.datapipes.DatashaderRasterizer` (functional name:
``rasterize_with_datashader``).

```{code-cell}
dp_datashader = dp_canvas.rasterize_with_datashader(vector_datapipe=dp_pyogrio)
dp_datashader
```

This will turn the vector {py:class}`geopandas.GeoDataFrame` into a
raster {py:class}`xarray.DataArray` grid, with the spatial coordinates and
bounds matching exactly with the template Sentinel-1 image 😎.

```{note}
Since we have just one Sentinel-1 🛰️ image and one raster 💧 flood
mask, we have an easy 1:1 mapping. There are two other scenarios supported by
{py:class}`zen3geo.datapipes.DatashaderRasterizer`:

1. N:1 - Many {py:class}`datashader.Canvas` objects to one vector
   {py:class}`geopandas.GeoDataFrame`. The single vector geodataframe will be
   broadcasted to match the length of the canvas list. This is useful for
   situations when you have a 🌐 'global' vector database that you want to pair
   with multiple 🛰️ satellite images.
2. N:N - Many {py:class}`datashader.Canvas` objects to many vector
   {py:class}`geopandas.GeoDataFrame` objects. In this case, the list of grids
   **must** ❗ have the same length as the list of vector geodataframes. E.g.
   if you have 5 grids, there must also be 5 vector files. This is so that a
   1:1 pairing can be done, useful when each raster tile 🖽 has its own
   associated vector annotation.
```

```{seealso}
For more details on how rasterization of polygons work behind the scenes 🎦,
check out {doc}`Datashader <datashader:index>`'s documentation on:

- {doc}`The datashader pipeline <datashader:getting_started/Pipeline>`
  (especially the section on Aggregation).
- {doc}`Rendering large collections of polygons <datashader:user_guide/Polygons>`
```


## 2️⃣ Combine and conquer ⚔️

So far, we've got two datapipes that should be 🧑‍🤝‍🧑 paired up in an X -> Y
manner:

1. The pre-processed Sentinel-1 🌈 raster image in ``dp_decibel_image``
2. The rasterized 💧 flood segmentation masks in ``dp_datashader``

One way to get these two pieces in a Machine Learning ready chip format is via
a stack, slice and split ™️ approach. Think of it like a sandwich 🥪, we first
stack the bread 🍞 and lettuce 🥬, and then slice the pieces 🍕 through the
layers once. Ok, that was a bad analogy, let's just stick with tensors 🤪.

### Stacking the raster layers 🥞

Each of our 🌈 raster inputs are {py:class}`xarray.DataArray` objects with the
same spatial resolution and extent 🪟, so these can be stacked into an
{py:class}`xarray.Dataset` with multiple data variables. First, we'll zip 🤐
the two datapipes together using {py:class}`torchdata.datapipes.iter.Zipper`
(functional name: ``zip``)

```{code-cell}
dp_zip = dp_decibel_image.zip(dp_datashader)
dp_zip
```

This will result in a DataPipe where each item is a tuple of (X, Y) pairs 🧑‍🤝‍🧑.
Just to illustrate what we've done so far, we can use
{py:class}`torchdata.datapipes.utils.to_graph` to visualize the data pipeline
⛓️.

```{code-cell}
torchdata.datapipes.utils.to_graph(dp=dp_zip)
```

Next, let's combine 🖇️ the two (X, Y) {py:class}`xarray.DataArray` objects in
the tuple into an {py:class}`xarray.Dataset` using
{py:class}`torchdata.datapipes.iter.Collator` (functional name: `collate`).
We'll also ✂️ clip the dataset to a bounding box area where the target water
mask has no 0 or NaN values.

```{code-cell}
def xr_collate_fn(image_and_mask: tuple) -> xr.Dataset:
    """
    Combine a pair of xarray.DataArray (image, mask) inputs into an
    xarray.Dataset with two data variables named 'image' and 'mask'.
    """
    # Turn 2 xr.DataArray objects into 1 xr.Dataset with multiple data vars
    image, mask = image_and_mask
    dataset: xr.Dataset = xr.merge(
        objects=[image.isel(band=0).rename("image"), mask.rename("mask")],
        join="override",
    )

    # Clip dataset to bounding box extent of where labels are
    mask_extent: tuple = mask.where(cond=mask == 1, drop=True).rio.bounds()
    clipped_dataset: xr.Dataset = dataset.rio.clip_box(*mask_extent)

    return clipped_dataset
```

```{code-cell}
dp_dataset = dp_zip.collate(collate_fn=xr_collate_fn)
dp_dataset
```

Double check to see that resulting {py:class}`xarray.Dataset`'s image and mask
looks ok 🙆‍♂️.

```{code-cell}
it = iter(dp_dataset)
dataset = next(it)

# Create subplot with VV image on the left and Water mask on the right
fig, axs = plt.subplots(ncols=2, figsize=(11.5, 4.5), sharey=True)
dataset.image.plot.imshow(ax=axs[0], cmap="Blues_r")
axs[0].set_title("Sentinel-1 VV channel")
dataset.mask.plot.imshow(ax=axs[1], cmap="Blues")
axs[1].set_title("Water mask")
plt.show()
```

### Slice into chips and turn into tensors 🗡️

To cut 🔪 the {py:class}`xarray.Dataset` into 128x128 sized chips, we'll use
{py:class}`zen3geo.datapipes.XbatcherSlicer` (functional name:
`slice_with_xbatcher`). Refer to {doc}`./chipping` if you need a 🧑‍🎓 refresher.

```{code-cell}
dp_xbatcher = dp_dataset.slice_with_xbatcher(input_dims={"y": 128, "x": 128})
dp_xbatcher
```

Next step is to convert the 128x128 chips into a {py:class}`torch.Tensor` via
{py:class}`torchdata.datapipes.iter.Mapper` (functional name: `map`). The 🛰️
Sentinel-1 image and 💧 water mask will be split out at this point too.

```{code-cell}
def dataset_to_tensors(chip: xr.Dataset) -> (torch.Tensor, torch.Tensor):
    """
    Converts an xarray.Dataset into to two torch.Tensor objects, the first one
    being the satellite image, and the second one being the target mask.
    """
    image: torch.Tensor = torch.as_tensor(chip.image.data)
    mask: torch.Tensor = torch.as_tensor(chip.mask.data.astype("uint8"))

    return image, mask
```

```{code-cell}
dp_map = dp_xbatcher.map(fn=dataset_to_tensors)
dp_map
```

At this point, we could do some batching and collating, but we'll point you
again to {doc}`./chipping` to figure it out 😝. Let's take a look at a graph
of the complete data pipeline.

```{code-cell}
torchdata.datapipes.utils.to_graph(dp=dp_map)
```

Sweet, time for the final step ⏩.

### Into a DataLoader 🏋️

Pass the DataPipe into {py:class}`torch.utils.data.DataLoader` 🤾!

```{code-cell}
dataloader = torch.utils.data.DataLoader(dataset=dp_map)
for i, batch in enumerate(dataloader):
    image, mask = batch
    print(f"Batch {i} - image: {image.shape}, mask: {mask.shape}")
```

Now go train some flood water detection models 🌊🌊🌊

```{seealso}
To learn more about AI-based flood mapping with SAR, check out these resources:

- [UNOSAT/NVIDIA Disaster Risk Monitoring Using Satellite Imagery online course](https://event.unitar.org/full-catalog/disaster-risk-monitoring-using-satellite-imagery)
- [Code to train a Convolutional Neural Network for flood segmentation](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb)
```
