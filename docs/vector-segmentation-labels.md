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
import pystac
import planetary_computer
import pyogrio

import torch
import torchdata
import zen3geo
```

## 0️⃣ Loading cloud-hosted vector files 💠

Let's load some vector data first from the web. As a case study, we'll look at
the flood water extent over Johor, Malaysia 🇲🇾 on 15 Dec 2019 that were
digitized by UNITAR-UNOSAT's rapid mapping service over Sentinel-1 imagery.

🔗 Links:
- https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service
- https://www.unitar.org/maps/unosat-rapid-mapping-service
- https://unitar.org/maps/countries
- https://data.humdata.org/dataset/waters-extents-as-of-15-december-2019-over-kota-tinggi-and-mersing-district-johor-state-of


```{code-cell}
# https://gdal.org/user/virtual_file_systems.html#vsizip-zip-archives
shape_urls = [
    "/vsizip/vsicurl/https://unosat-maps.web.cern.ch/MY/FL20191217MYS/FL20191217MYS_SHP.zip/ST1_20191215_WaterExtent_Johor_AOI1.shp",
    "/vsizip/vsicurl/https://unosat-maps.web.cern.ch/MY/FL20191217MYS/FL20191217MYS_SHP.zip/ST1_20191215_WaterExtent_Johor_AOI2.shp",
]
```

## Inspect the shapefiles

```{code-cell}
geodataframe0 = pyogrio.read_dataframe(shape_urls[0])
geodataframe0
```

```{code-cell}
geodataframe1 = pyogrio.read_dataframe(shape_urls[1])
geodataframe1
```
