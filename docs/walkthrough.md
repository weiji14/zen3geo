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

# Walkthrough

> *To get it, you first see it, and then let it go*

In this tutorial 🧑‍🏫, we'll step through an Earth Observation 🛰️ data pipeline
using ``torchdata`` and by the end of this lesson, you should be able to:
- Find Cloud-Optimized GeoTIFFs (COGs) from STAC catalogs 🥞
- Construct a DataPipe that iteratively reads several COGs in a stream 🌊
- Loop through batches of images in a DataPipe with a DataLoader 🏋️

## 🎉 **Getting started**

These are the tools 🛠️ you'll need.

```{code-cell}
# Geospatial libraries
import pystac
import planetary_computer
import rioxarray
# Deep Learning libraries
import torch
import torchdata
import zen3geo
```

Just to make sure we’re on the same page 📃,
let’s check that we’ve got compatible versions installed.

```{code-cell}
print(f"pystac version: {pystac.__version__}")
print(f"planetary-computer version: {planetary_computer.__version__}")
print(f"torch version: {torch.__version__}")

print(f"torchdata version: {torchdata.__version__}")
print(f"zen3geo version: {zen3geo.__version__}")
rioxarray.show_versions()
```

## 0️⃣ Find [Cloud-Optimized GeoTIFFs](https://www.cogeo.org) 🗺️

Let's get some optical satellite data using [STAC](https://stacspec.org)!
How about Sentinel-2 L2A data over Singapore 🇸🇬?

🔗 Links:
- [Official Sentinel-2 description page at ESA](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [Microsoft Planetary Computer STAC Explorer](https://planetarycomputer.microsoft.com/explore?c=103.8152%2C1.3338&z=10.08&v=2&d=sentinel-2-l2a&s=false%3A%3A100%3A%3Atrue&ae=0&m=cql%3A2ff1401acb50731fa0a6d1e2a46f3064&r=Natural+color)
- [AWS Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs)


```{code-cell}
item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_20220115T032101_R118_T48NUG_20220115T170435"

# Load the individual item metadata and sign the assets
item = pystac.Item.from_file(item_url)
signed_item = planetary_computer.sign(item)
signed_item
```

### Inspect one of the data assets 🍱

The Sentinel-2 STAC item contains several assets.
These include different 🌈 bands (e.g. 'B02', 'B03', 'B04').
Let's just use the 'visual' product for now which includes the RGB bands.

```{code-cell}
url: str = signed_item.assets["visual"].href
da = rioxarray.open_rasterio(filename=url)
da
```

This is how the Sentinel-2 image looks like over Singapore on 15 Jan 2022.

![Sentinel-2 image over Singapore on 20220115](https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-2-l2a&item=S2A_MSIL2A_20220115T032101_R118_T48NUG_20220115T170435&assets=visual&asset_bidx=visual%7C1%2C2%2C3&nodata=0)

## 1️⃣ Construct [DataPipe](https://github.com/pytorch/data/tree/v0.3.0#what-are-datapipes) 📡

A torch `DataPipe` is a way of composing data (rather than inheriting data).
Yes, I don't know what it really means either, so here's some extra reading.

🔖 References:
- https://pytorch.org/blog/pytorch-1.11-released/#introducing-torchdata
- https://github.com/pytorch/data/tree/v0.3.0#what-are-datapipes
- https://realpython.com/inheritance-composition-python

### Create an Iterable 📏

Start by wrapping a list of URLs to the Cloud-Optimized GeoTIFF files.
We only have 1 item so we'll use ``[url]``, but if you have more, you can do
``[url1, url2, url3]``, etc. Pass this iterable list into
[`torchdata.datapipes.iter.IterableWrapper`](https://pytorch.org/data/0.4.0/generated/torchdata.datapipes.iter.IterableWrapper.html):

```{code-cell}
dp = torchdata.datapipes.iter.IterableWrapper(iterable=[url])
dp
```

The ``dp`` variable is the DataPipe!
Now to apply some more transformations/functions on it.

### Read using RioXarrayReader 🌐

This is where ``zen3geo`` comes in. We'll be using the
{py:class}`zen3geo.datapipes.RioXarrayReaderIterDataPipe` class, or rather,
the short alias  ``zen3geo.RioXarrayReader``.

Confusingly, there are two ways or forms of applying ``RioXarrayReader``,
a class-based method and a functional method.

```{code-cell}
# Using class constructors
dp_rioxarray = zen3geo.RioXarrayReader(source_datapipe=dp)
dp_rioxarray
```

```{code-cell}
# Using functional form (recommended)
dp_rioxarray = dp.read_from_rioxarray()
dp_rioxarray
```

Note that both ways are equivalent (they produce the same IterDataPipe output),
but the latter (functional) form is preferred, see also
https://pytorch.org/data/0.4.0/tutorial.html#registering-datapipes-with-the-functional-api

What if you don't want the whole Sentinel-2 scene at the full 10m resolution?
Since we're using Cloud-Optimized GeoTIFFs, you could set an ``overview_level``
(following https://corteva.github.io/rioxarray/stable/examples/COG.html).

```{code-cell}
dp_rioxarray_zoom3 = dp.read_from_rioxarray(overview_level=3)
dp_rioxarray_zoom3
```

Extra keyword arguments will be handled by
[``rioxarray.open_rasterio``](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray-open-rasterio)
or [``rasterio.open``](https://rasterio.readthedocs.io/en/stable/api/rasterio.html#rasterio.open).

```{note}
Other DataPipe classes/functions can be stacked or joined to this basic GeoTIFF
reader. For example, clipping by bounding box or reprojecting to a certain
Coordinate Reference System. If you would like to implement this, check out the
[Contributing Guidelines](./CONTRIBUTING) to get started!
```

## 2️⃣ Loop through DataPipe ⚙️

A DataPipe describes a flow of information.
Through a series of steps it goes,
as one piece comes in, another might follow.

At the most basic level, you could iterate through the DataPipe like so:

```{code-cell}
it = iter(dp_rioxarray_zoom3)
filename, dataarray = next(it)
dataarray
```

Or if you're more familiar with a for-loop, here it is:

```{code-cell}
for filename, dataarray in dp_rioxarray_zoom3:
    print(dataarray)
    # Run model on this data batch
```

For the deep learning folks though, you'll probably want to use
[``torch.utils.data.DataLoader``](https://pytorch.org/docs/1.11/data.html#torch.utils.data.DataLoader):

```{code-cell}
dataloader = torch.utils.data.DataLoader(dataset=dp_rioxarray_zoom3)
dataloader
```

And so it begins 🌄

---

That’s all 🎉! For more information on how to use DataPipes, check out:

- Tutorial at https://pytorch.org/data/0.4.0/tutorial.html
- Usage examples at https://pytorch.org/data/0.4.0/examples.html

If you have any questions 🙋, feel free to ask us anything at
https://github.com/weiji14/zen3geo/discussions or visit the Pytorch forums at
https://discuss.pytorch.org/c/data/37.

Cheers!
