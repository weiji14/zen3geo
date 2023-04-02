# <center> ☯ *zen3geo* - The 🌏 data science library you've been waiting for <center>

## Installation

Get what you need, not more, not less:

| Command                        |  Dependencies |
|:-------------------------------|---------------|
| `pip install zen3geo`          | rioxarray, torchdata |
| `pip install zen3geo[raster]`  | rioxarray, torchdata, xbatcher, zarr |
| `pip install zen3geo[spatial]` | rioxarray, torchdata, datashader, spatialpandas |
| `pip install zen3geo[stac]`    | rioxarray, torchdata, pystac, pystac-client, stackstac, xpystac |
| `pip install zen3geo[vector]`  | rioxarray, torchdata, pyogrio[geopandas] |

Retrieve more ['extras'](https://github.com/weiji14/zen3geo/blob/main/pyproject.toml) using

    pip install zen3geo[raster,spatial,stac,vector]

To install the development version from [TestPyPI](https://test.pypi.org/project/zen3geo), do:

    pip install --pre --extra-index-url https://test.pypi.org/simple/ zen3geo

May [conda-forge](https://anaconda.org/conda-forge/zen3geo) be with you,
though optional dependencies it has not.

    mamba install --channel conda-forge zen3geo

For the eager ones, {ref}`contributing <contributing:running:locally>` will take you further.
