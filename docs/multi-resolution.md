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

# Multi-resolution

> On top of a hundred foot pole you linger
>
> Clinging to the first mark of the scale
>
> How do you proceed higher?
>
> It will take more than a leap of faith

Earth Observation 🛰️ and climate projection 🌡️ data can be captured at
different levels of detail. In this lesson, we'll work with a multitude of
spatial resolutions 📏, learning to respect the ground sampling distance or
native resolution 🔬 of the physical variable being measured, while 🪶
minimizing memory usage. By the end of the lesson, you should be able to:

- Find 🔍 low and high spatial resolution climate datasets and load them from
  Zarr stores
- Stack 🥞 different spatial resolution datasets into an xarray.DataTree
- Slice 🔪 the multi-resolution dataset into chips and pass into a DataLoader

🔗 Links:
- https://carbonplan.org/research/cmip6-downscaling-explainer
- https://github.com/carbonplan/cmip6-downscaling/blob/1.0/notebooks/accessing_data_example.ipynb
- https://github.com/xarray-contrib/xbatcher/issues/93

## 🎉 **Getting started**

These are the tools 🛠️ you'll need.

```{code-cell}
import matplotlib.pyplot as plt
import torchdata
import xarray as xr
import xpystac
import zen3geo

from datatree import DataTree
```

## 0️⃣ Find climate model datasets 🪸

The two datasets we'll be working with are 🌐 gridded climate projections, one
that is in its original low 🔅 spatial resolution, and another one of a
higher 🔆 spatial resolution. Specifically, we'll be looking at the maximum
temperature 🌡️ (tasmax) variable from one of the Coupled Model Intercomparison
Project Phase 6 (CMIP6) global coupled ocean-atmosphere general circulation
model (GCM) 💨 outputs that is of low resolution (67.5 arcminute), and a
super-resolution product from DeepSD 🤔 that is of a higher resolution (15
arcminute).

```{note}
The following tutorial will mostly use the term super-resolution 🔭 from
Computer Vision instead of downscaling ⏬. It's just that the term
downscaling ⏬ (going from low to high resolution) can get confused with
downsampling 🙃 (going from high to low resolution), whereas
super-resolution 🔭 is unambiguously about going from low 🔅 to high 🔆
resolution.
```

🔖 References:
- https://carbonplan.org/research/cmip6-downscaling
- https://github.com/tjvandal/deepsd
- https://tutorial.xarray.dev/intermediate/cmip6-cloud.html

```{code-cell}
lowres_raw = "https://cpdataeuwest.blob.core.windows.net/cp-cmip/cmip6/ScenarioMIP/MRI/MRI-ESM2-0/ssp585/r1i1p1f1/Amon/tasmax/gn/v20191108"
highres_deepsd = "https://cpdataeuwest.blob.core.windows.net/cp-cmip/version1/data/DeepSD/ScenarioMIP.MRI.MRI-ESM2-0.ssp585.r1i1p1f1.month.DeepSD.tasmax.zarr"
```

This is how the projected maximum temperature 🥵 for August 2089 looks like over
South Asia 🪷 for a low resolution 🔅 Global Climate Model (left) and a
high resolution 🔆 downscaled product (right).

```{code-cell}
:tags: [hide-input]
# Zarr datasets from https://github.com/carbonplan/research/blob/d05d148fd716ba6304e3833d765069dd890eaf4a/articles/cmip6-downscaling-explainer/components/downscaled-data.js#L97-L122
ds_gcm = xr.open_dataset(filename_or_obj="https://cmip6downscaling.blob.core.windows.net/vis/article/fig1/regions/india/gcm-tasmax.zarr")
ds_gcm -= 273.15  # convert from Kelvin to Celsius
ds_downscaled = xr.open_dataset(filename_or_obj="https://cmip6downscaling.blob.core.windows.net/vis/article/fig1/regions/india/downscaled-tasmax.zarr")
ds_downscaled -= 273.15  # convert from Kelvin to Celsius

# Plot projected maximum temperature over South Asia from GCM and GARD-MV
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 3), sharey=True)

img1 = ds_gcm.tasmax.plot.imshow(ax=ax[0], cmap="inferno", vmin=16, vmax=48, add_colorbar=False)
ax[0].set_title("Global Climate Model (67.5 arcminute)")

img2 = ds_downscaled.tasmax.plot.imshow(ax=ax[1], cmap="inferno", vmin=16, vmax=48, add_colorbar=False)
ax[1].set_title("Downscaled result (15 arcminute)")

cbar = fig.colorbar(mappable=img1, ax=ax.ravel().tolist(), extend="both")
cbar.set_label(label="Daily Max Near-Surface Air\nTemperature in Aug 2089 (°C)")

plt.show()
```

### Load Zarr stores 📦

The {doc}`Zarr <zarr:index>` stores 🧊 can be loaded into an
{py:class}`xarray.Dataset` via {py:class}`zen3geo.datapipes.XpySTACAssetReader`
(functional name: ``read_from_xpystac``) with the `engine="zarr"` keyword
argument.

```{code-cell}
dp_lowres = torchdata.datapipes.iter.IterableWrapper(iterable=[lowres_raw])
dp_highres = torchdata.datapipes.iter.IterableWrapper(iterable=[highres_deepsd])

dp_lowres_dataset = dp_lowres.read_from_xpystac(engine="zarr", chunks="auto")
dp_highres_dataset = dp_highres.read_from_xpystac(engine="zarr", chunks="auto")
```

### Inspect the climate datasets 🔥

Let's now preview 👀 the low-resolution 🔅 and high-resolution 🔆 temperature
datasets.

```{code-cell}
it = iter(dp_lowres_dataset)
ds_lowres = next(it)
print(ds_lowres)
```

```{code-cell}
it = iter(dp_highres_dataset)
ds_highres = next(it)
print(ds_highres)
```

Notice that the low-resolution 🔅 dataset has lon/lat pixels of shape
(320, 160), whereas the high-resolution 🔆 dataset is of shape (1440, 720). So
there has been a 4.5x increase 📈 in spatial resolution going from the raw GCM
🌐 grid to the super-resolution 🔭 DeepSD grid.
