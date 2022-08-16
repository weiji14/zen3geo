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

# Vector segmentation labels

> *Clouds float by, water flows on;
> in movement there is no grasping, in Chan there is no settling*

For supervised machine learning, labels are needed in addition to the input
image. Here, we'll step through an example workflow on matching vector label
data (points, lines, polygons) to Earth Observation data inputs. Specifically,
this tutorial will cover:

- Reading shapefiles directly from the web via pyogrio
- Rasterizing vector polygons from a geopandas.GeoDataFrame to an xarray.DataArray
- Pairing satellite images with the rasterized labels followed by chipping


## 🎉 **Getting started**

These are the tools 🛠️ you'll need.

```{code-cell}
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
Malaysia 🇲🇾 on 15 Dec 2019 that were digitized by UNITAR-UNOSAT's rapid mapping
service over Synthetic Aperture Radar (SAR) images. Specifically, we'll be
using the Sentinel-1 Ground Range Detected (GRD) product's VV polarization
channel.

🔗 Links:
- https://www.unitar.org/maps/unosat-rapid-mapping-service
- https://unitar.org/maps/countries
- [Microsoft Planetary Computer STAC Explorer](https://planetarycomputer.microsoft.com/explore?c=103.6637%2C2.1494&z=8.49&v=2&d=sentinel-1-grd&s=false%3A%3A100%3A%3Atrue&ae=0&m=cql%3Afc3d85b6ab43d3e8ebe168da0206f2cf&r=VV%2C+VH+False-color+composite)

```{code-cell}
item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/S1A_IW_GRDH_1SDV_20191215T224757_20191215T224822_030365_037955"

# Load the individual item metadata and sign the assets
item = pystac.Item.from_file(item_url)
signed_item = planetary_computer.sign(item)
signed_item
```

This is how the Sentinel-1 image looks like over Johor in Peninsular Malaysia
on 15 Dec 2019.

![Sentinel-1 image over Johor, Malaysia on 20191215](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-1-grd&item=S1A_IW_GRDH_1SDV_20191215T224757_20191215T224822_030365_037955&assets=vv&assets=vh&expression=vv%2Cvh%2Cvv%2Fvh&rescale=0%2C500&rescale=0%2C300&rescale=0%2C7&tile_format=png)

### Load and reproject image data 🔄

To keep things simple, we'll load just the VV channel into a DataPipe via
{py:class}`zen3geo.datapipes.rioxarray.RioXarrayReaderIterDataPipe`

```{code-cell}
url = signed_item.assets["vv"].href
dp = torchdata.datapipes.iter.IterableWrapper(iterable=[url])
# Reading lower resolution grid using overview_level=2
dp_rioxarray = dp.read_from_rioxarray(overview_level=2)
dp_rioxarray
```

The Sentinel-1 image from Planetary Computer comes in longitude/latitude
geographic coordinates by default (EPSG:4326). To make the pixels more equal
area, we can project it to a local projected coordinate system instead.

```{code-cell}
def reproject_to_local_utm(dataarray: xr.DataArray, resolution: float=80.0) -> xr.DataArray:
    """
    Reproject an xarray.DataArray grid from EPSG;4326 to a local UTM coordinate
    reference system.
    """
    # Estimate UTM coordinate reference from a single pixel
    pixel = dataarray.isel(y=slice(0, 1), x=slice(0,1))
    new_crs = dataarray.rio.reproject(dst_crs="EPSG:4326").rio.estimate_utm_crs()

    return dataarray.rio.reproject(dst_crs=new_crs, resolution=resolution)
```

```{code-cell}
dp_reprojected = dp_rioxarray.map(fn=reproject_to_local_utm)
```

```{note}
Note that Universal Transverse Mercator (UTM) isn't an equal-area projection
system. However, Sentinel-1 satellite scenes from Copernicus are usually
distributed in a UTM coordinate reference system, and UTM is typically a close
enough approximation to the local geographic area, or at least it won't matter
much when we're looking at spatial resolutions over several 10s of metres.
```

### Transform and visualize raster data 🔎

Let's visualize the Sentinel-1 image, but before that, we'll transform the VV
data from linear to [decibel](https://en.wikipedia.org/wiki/Decibel) units.

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

Now to visualize the transformed Sentinel-1 image. Let's zoom in to one of the
analysis extent areas we'll be working on later.

```{code-cell}
it = iter(dp_decibel)
dataarray = next(it)

da_clip = dataarray.rio.clip_box(minx=371483, miny=190459, maxx=409684, maxy=229474)
da_clip.isel(band=0).plot.imshow(figsize=(11.5, 9), cmap="Blues_r", vmin=18, vmax=26)
```

Notice how the darker blue areas tend to correlate more with water features
like the meandering rivers and the sea on the NorthEast. This is because the
SAR signal which is side looking reflects off flat water bodies like a mirror,
with very little energy getting reflected back directly to the sensor (hence
it looks darker).

### Load and visualize cloud-hosted vector files 💠

Let's now load some vector data from the web. These are polygons of the
segmented water extent digitized by UNOSAT's AI Based Rapid Mapping Service.
We'll be converting these vector polygons to raster masks later.

🔗 Links:
- https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service
- [Humanitarian Data Exchange link to polygon dataset](https://data.humdata.org/dataset/waters-extents-as-of-15-december-2019-over-kota-tinggi-and-mersing-district-johor-state-of)
- [Disaster Risk Monitoring Using Satellite Imagery online course](https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1)

```{code-cell}
# https://gdal.org/user/virtual_file_systems.html#vsizip-zip-archives
shape_urls = [
    "/vsizip/vsicurl/https://unosat-maps.web.cern.ch/MY/FL20191217MYS/FL20191217MYS_SHP.zip/ST1_20191215_WaterExtent_Johor_AOI1.shp",
    "/vsizip/vsicurl/https://unosat-maps.web.cern.ch/MY/FL20191217MYS/FL20191217MYS_SHP.zip/ST1_20191215_WaterExtent_Johor_AOI2.shp",
]
```

So there are two shapefiles containing polygons of the mapped water extent.
Let's put this list into a DataPipe called
{py:class}`zen3geo.datapipes.PyogrioReader`.

```{code-cell}
dp = torchdata.datapipes.iter.IterableWrapper(iterable=shape_urls)
dp_pyogrio = dp.read_from_pyogrio()
```

This will take care of loading each shapefile into a
{py:class}`geopandas.GeoDataFrame` object. Let's take a look at the data table
to see what attributes are inside.

```{code-cell}
# Iterate through the datapipe one by one
it = iter(dp_pyogrio)
geodataframe0 = next(it)  # 1st shapefile
geodataframe1 = next(it)  # 2nd shapefile
```

```{code-cell}
geodataframe0.dropna(axis="columns")
```

```{code-cell}
geodataframe1.dropna(axis="columns")
```

Cool, and we can also visualize the polygons 🔷 on a 2D map. To align the
coordinates with the Sentinel-1 image above, we'll first use
{py:meth}`geopandas.GeoDataFrame.to_crs` to reproject the vector from EPSG:4326
(longitude/latitude) to EPSG:32649 (UTM Zone 49N).

```{code-cell}
print(f"Original bounds in EPSG:4326:\n{geodataframe1.bounds}")
gdf = geodataframe1.to_crs(crs="EPSG:32649")
print(f"New bounds in EPSG:32649:\n{gdf.bounds}")
```

Plot it with {py:meth}`geopandas.GeoDataFrame.plot`. This vector map should
correspond to the zoomed in Sentinel-1 image plotted earlier above.

```{code-cell}
gdf.plot(figsize=(11.5, 9))
```
