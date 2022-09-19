"""
Iterable-style DataPipes for geospatial raster 🌈 and vector 🚏 data.
"""

from zen3geo.datapipes.datashader import (
    DatashaderRasterizerIterDataPipe as DatashaderRasterizer,
    XarrayCanvasIterDataPipe as XarrayCanvas,
)
from zen3geo.datapipes.geopandas import (
    GeoPandasRectangleClipperIterDataPipe as GeoPandasRectangleClipper,
)
from zen3geo.datapipes.pyogrio import PyogrioReaderIterDataPipe as PyogrioReader
from zen3geo.datapipes.pystac import PySTACItemReaderIterDataPipe as PySTACItemReader
from zen3geo.datapipes.pystac_client import (
    PySTACAPISearchIterDataPipe as PySTACAPISearch,
)
from zen3geo.datapipes.rioxarray import RioXarrayReaderIterDataPipe as RioXarrayReader
from zen3geo.datapipes.stackstac import StackSTACStackerIterDataPipe as StackSTACStacker
from zen3geo.datapipes.xbatcher import XbatcherSlicerIterDataPipe as XbatcherSlicer
