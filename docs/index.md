# <center> ☯ *zen3geo* - The 🌏 data science library you've been waiting for <center>

## Installation

Get what you need, not more, not less:

| Command                        |  Dependencies |
|:-------------------------------|---------------|
| `pip install zen3geo`          | rioxarray, torchdata |
| `pip install zen3geo[raster]`  | rioxarray, torchdata, xbatcher |
| `pip install zen3geo[spatial]` | rioxarray, torchdata, datashader, spatialpandas |
| `pip install zen3geo[vector]`  | rioxarray, torchdata, pyogrio[geopandas] |

Retrieve more 'extras' using

    pip install zen3geo[raster,spatial,vector]

To install the development version from [TestPyPI](https://test.pypi.org/project/zen3geo), do:

    pip install --pre --extra-index-url https://test.pypi.org/simple/ zen3geo

For the eager ones, {ref}`contributing <contributing:running:locally>` will take you further.
