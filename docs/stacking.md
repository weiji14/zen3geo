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
import numpy as np
import planetary_computer
import pystac
import rasterio
import torchdata
import xarray as xr
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
- [UNOSAT satellite-detected landslide maps over Pasaman, Indonesia](https://unosat.org/products/3064)
- [Humanitarian Data Exchange link](https://data.humdata.org/dataset/landslide-analysis-in-mount-talakmau-in-pasaman-pasaman-barat-districts-indonesia-as-of-04)
- [Microsoft Planetary Computer STAC Explorer](https://planetarycomputer.microsoft.com/explore?c=99.9823%2C0.0564&z=11.34&v=2&d=sentinel-1-grd%7C%7Ccop-dem-glo-30&m=cql%3A99108f1228c2543e04cad62d0a795c1a%7C%7CMost+recent&r=VV%2C+VH+False-color+composite%7C%7CElevation+%28terrain%29&s=false%3A%3A100%3A%3Atrue%7C%7Ctrue%3A%3A100%3A%3Atrue&ae=0)

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
{py:class}`zen3geo.datapipes.PySTACAPISearcher` (functional name:
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
   [`search_for_pystac_item()`](zen3geo.datapipes.PySTACAPISearcher), e.g. the
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

This is one of the two DEM tiles 🀫 that will be returned from the query.

![Copernicus 30m DEM over Sumatra Barat, Indonesia](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=cop-dem-glo-30&item=Copernicus_DSM_COG_10_N00_00_E099_00_DEM&assets=data&colormap_name=terrain&rescale=-1000%2C4000)

### Landslide extent vector polygons 🔶

Now for the target labels 🏷️. Following {doc}`./vector-segmentation-masks`,
we'll first load the digitized landslide polygons from a vector file 📁 using
{py:class}`zen3geo.datapipes.PyogrioReader` (functional name:
``read_from_pyogrio``).

```{code-cell}
# https://gdal.org/user/virtual_file_systems.html#vsizip-zip-archives
shape_url = "/vsizip/vsicurl/https://unosat-maps.web.cern.ch/ID/LS20220308IDN/LS20220308IDN_SHP.zip/LS20220308IDN_SHP/S2_20220304_LandslideExtent_MountTalakmau.shp"

dp_shapes = torchdata.datapipes.iter.IterableWrapper(iterable=[shape_url])
dp_pyogrio = dp_shapes.read_from_pyogrio()
dp_pyogrio
```

Let's take a look at the {py:class}`geopandas.GeoDataFrame` data table
📊 to see the attributes inside.

```{code-cell}
it = iter(dp_pyogrio)
geodataframe = next(it)
print(geodataframe.bounds)
geodataframe.dropna(axis="columns")
```

We'll show you what the landslide segmentation masks 😷 look like after it's
been rasterized later 😉.


## 1️⃣ Stack bands, append variables 📚

There are now three layers 🍰 to handle, two rasters and a vector. This section
will show you step by step 📶 instructions to
{doc}`combine them using xarray <xarray:user-guide/combining>` like so:

1. Stack the Sentinel-1 🛰️ time-series STAC Items (GeoTIFFs) into an
   {py:class}`xarray.DataArray`.
2. Combine the Sentinel-1 and Copernicus DEM ⛰️ {py:class}`xarray.DataArray`
   layers into a single {py:class}`xarray.Dataset`.
3. Using the {py:class}`xarray.Dataset` as a canvas template, rasterize the
   landslide 🛝 polygon extents, and append the resulting segmentation mask as
   another data variable 🗃️ in the {py:class}`xarray.Dataset`.

### Stack multi-channel time-series GeoTIFFs 🗓️

Each {py:class}`pystac.Item` in a {py:class}`pystac.ItemCollection` represents
a 🛰️ Sentinel-1 GRD image captured at a particular datetime ⌚. Let's subset
the data to just the mountain area, and stack 🥞 all the STAC items into a 4D
time-series tensor using {py:class}`zen3geo.datapipes.StackSTACStacker`
(functional name: `stack_stac_items`)!

```{code-cell}
dp_sen1_stack = dp_sen1_items.stack_stac_items(
    assets=["vh", "vv"],  # SAR polarizations
    epsg=32647,  # UTM Zone 47N
    resolution=30,  # Spatial resolution of 30 metres
    bounds_latlon=[99.933681, -0.009951, 100.065765, 0.147054], # W, S, E, N
    xy_coords="center",  # pixel centroid coords instead of topleft corner
    dtype=np.float16,  # Use a lightweight data type
)
dp_sen1_stack
```

The keyword arguments are 📨 passed to {py:func}`stackstac.stack` behind the
scenes. The important❕parameters to set in this case are:

- **assets**: The STAC item assets 🍱 (typically the 'band' names)
- **epsg**: The 🌐 EPSG projection code, best if you know the native projection
- **resolution**: Spatial resolution 📏. The Sentinel-1 GRD is actually at 10m,
  but we'll resample to 30m to keep things small 🤏 and match the Copernicus
  DEM.

The result is a single {py:class}`xarray.DataArray` 'datacube' 🧊 with
dimensions (time, band, y, x).

```{code-cell}
it = iter(dp_sen1_stack)
dataarray = next(it)
dataarray
```

### Append single-band DEM to datacube 🧊

Time for layer number 2 💕. Let's read the Copernicus DEM ⛰️ STAC Item into an
{py:class}`xarray.DataArray` first, again via
{py:class}`zen3geo.datapipes.StackSTACStacker` (functional name:
`stack_stac_items`). We'll need to ensure ✔️ that the DEM is reprojected to the
same 🌏 coordinate reference system and 📐 aligned to the same spatial extent
as the Sentinel-1 time-series.

```{code-cell}
dp_copdem_stack = dp_copdem30_items.stack_stac_items(
    assets=["data"],
    epsg=32647,  # UTM Zone 47N
    resolution=30,  # Spatial resolution of 30 metres
    bounds_latlon=[99.933681, -0.009951, 100.065765, 0.147054], # W, S, E, N
    xy_coords="center",  # pixel centroid coords instead of topleft corner
    dtype=np.float16,  # Use a lightweight data type
    resampling=rasterio.enums.Resampling.bilinear,  # Bilinear resampling
)
dp_copdem_stack
```

```{code-cell}
it = iter(dp_copdem_stack)
dataarray = next(it)
dataarray
```

Why are there 2 ⏳ time layers? Actually, the STAC query had returned two DEM
tiles 🀫, and {py:func}`stackstac.stack` has stacked both of them along a
dimension name 'time' (probably better named 'tile'). Fear not, the tiles can
be joined 🍒 into a single terrain mosaic layer with dimensions ("band", "y",
"x") using {py:class}`zen3geo.datapipes.StackSTACMosaicker` (functional name:
`mosaic_dataarray`).

```{code-cell}
dp_copdem_mosaic = dp_copdem_stack.mosaic_dataarray()
dp_copdem_mosaic
```

Great! The two {py:class}`xarray.DataArray` objects (Sentinel-1 and Copernicus
DEM mosaic) can now be combined 🪢. First, use
{py:class}`torchdata.datapipes.iter.Zipper` (functional name: `zip`) to put the
two {py:class}`xarray.DataArray` objects into a tuple 🎵.

```{code-cell}
dp_sen1_copdem = dp_sen1_stack.zip(dp_copdem_mosaic)
dp_sen1_copdem
```

Next, use {py:class}`torchdata.datapipes.iter.Collator` (functional name:
`collate`) to convert 🤸 the tuple of {py:class}`xarray.DataArray` objects into
an {py:class}`xarray.Dataset` 🧊, similar to what was done in
{doc}`./object-detection-boxes`.

```{code-cell}
def sardem_collate_fn(sar_and_dem: tuple) -> xr.Dataset:
    """
    Combine a pair of xarray.DataArray (SAR, DEM) inputs into an
    xarray.Dataset with data variables named 'vh', 'vv' and 'dem'.
    """
    # Turn 2 xr.DataArray objects into 1 xr.Dataset with multiple data vars
    sar, dem = sar_and_dem

    # Initialize xr.Dataset with VH and VV channels
    dataset: xr.Dataset = sar.sel(band="vh").to_dataset(name="vh")
    dataset["vv"] = sar.sel(band="vv")

    # Add Copernicus DEM mosaic as another layer
    dataset["dem"] = dem.squeeze()

    return dataset
```

```{code-cell}
dp_vhvvdem_dataset = dp_sen1_copdem.collate(collate_fn=sardem_collate_fn)
dp_vhvvdem_dataset
```

Here's how the current {py:class}`xarray.Dataset` 🧱 is structured. Notice that
VH and VV polarization channels 📡 are now two separate data variables, each
with dimensions (time, y, x). The DEM ⛰️ data is not a time-series, so it has
dimensions (y, x) only. All the 'band' dimensions have been removed ❌ and are
now data variables within the {py:class}`xarray.Dataset` 😎.

```{code-cell}
it = iter(dp_vhvvdem_dataset)
dataset = next(it)
dataset
```

Visualize the DataPipe graph ⛓️ too for good measure.

```{code-cell}
torchdata.datapipes.utils.to_graph(dp=dp_vhvvdem_dataset)
```


### Rasterize target labels to datacube extent 🏷️

The landslide polygons 🔶 can now be rasterized and added as another layer to
our datacube 🧊. Following {doc}`./vector-segmentation-masks`, we'll first fork
the DataPipe into two branches 🫒 using
{py:class}`torchdata.datapipes.iter.Forker` (functional name: `fork`).

```{code-cell}
dp_vhvvdem_canvas, dp_vhvvdem_datacube = dp_vhvvdem_dataset.fork(num_instances=2)
dp_vhvvdem_canvas, dp_vhvvdem_datacube
```

Next, create a blank canvas 📃 using
{py:class}`zen3geo.datapipes.XarrayCanvas` (functional name:
``canvas_from_xarray``) and rasterize 🖌 the vector polygons onto the template
canvas using {py:class}`zen3geo.datapipes.DatashaderRasterizer` (functional
name: ``rasterize_with_datashader``)

```{code-cell}
dp_datashader = dp_vhvvdem_canvas.canvas_from_xarray().rasterize_with_datashader(
    vector_datapipe=dp_pyogrio
)
dp_datashader
```

As promised, here's the landslide mask 🏁 visualized.

```{code-cell}
it = iter(dp_datashader)
mask = next(it)
mask.plot.imshow(cmap="binary_r")
```

Cool, and this layer can be added as another data variable in the datacube.

```{code-cell}
def cubemask_collate_fn(cube_and_mask: tuple) -> xr.Dataset:
    """
    Insert target 'mask' (xarray.DataArray) as a data variable in an existing
    datacube (xarray.Dataset).
    """
    # Turn 2 xr.DataArray objects into 1 xr.Dataset with multiple data var
    datacube, mask = cube_and_mask
    datacube["mask"] = mask

    return datacube
```

```{code-cell}
dp_datacube = dp_vhvvdem_datacube.zip(dp_datashader).collate(
    collate_fn=cubemask_collate_fn
)
dp_datacube
```

Preview the final datacube 🧊 and DataPipe graph ⛓️.

```{code-cell}
it = iter(dp_datacube)
datacube = next(it)
datacube
```

```{code-cell}
torchdata.datapipes.utils.to_graph(dp=dp_datacube)
```
