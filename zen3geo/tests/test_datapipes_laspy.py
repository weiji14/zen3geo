"""
Tests for laspy datapipes.
"""
import tempfile
import urllib

import numpy as np
import numpy.testing as npt
import pytest
from torchdata.datapipes.iter import IterableWrapper

from zen3geo.datapipes import LaspyReader

laspy = pytest.importorskip("laspy")


# %%
def test_laspy_reader_las_local():
    """
    Ensure that LaspyReader works to read in a LAS file (on disk) and outputs a
    laspy.lasdata.LasData object.
    """
    with tempfile.NamedTemporaryFile(suffix=".las") as tmpfile:
        urllib.request.urlretrieve(
            url="https://github.com/laz-rs/laz-rs/raw/0.8.3/tests/data/point-time-color.las",
            filename=tmpfile.name,
        )
        dp = IterableWrapper(iterable=[tmpfile.name])

        # Using class constructors
        dp_laspy = LaspyReader(source_datapipe=dp)
        # Using functional form (recommended)
        dp_laspy = dp.read_from_laspy()

        assert len(dp_laspy) == 1
        it = iter(dp_laspy)
        lasdata = next(it)

    assert lasdata.header.version == laspy.header.Version(major=1, minor=2)
    assert lasdata.header.point_format == laspy.point.PointFormat(point_format_id=3)
    assert lasdata.points.array.shape == (1065,)
    assert lasdata.xyz.shape == (1065, 3)
    npt.assert_allclose(
        actual=lasdata.xyz.mean(axis=0),
        desired=[494494.6635117371, 4878134.831230047, 132.31299530516432],
    )
    npt.assert_allclose(actual=np.unique(lasdata.classification.array), desired=[1, 2])


def test_laspy_reader_laz_http():
    """
    Ensure that LaspyReader works to read in a LAZ file (from a HTTP byte stream)
    and outputs a laspy.lasdata.LasData object.
    """
    file_url: str = "https://github.com/laz-rs/laz-rs/raw/0.8.3/tests/data/point-version-1-point-wise.laz"
    dp = IterableWrapper(iterable=[file_url])
    _, dp_stream = (
        dp.read_from_http()
        .read_from_stream()
        .set_length(length=1)
        .unzip(sequence_length=2)
    )

    # Using class constructors
    dp_laspy = LaspyReader(source_datapipe=dp_stream)
    # Using functional form (recommended)
    dp_laspy = dp_stream.read_from_laspy()

    assert len(dp_laspy) == 1
    it = iter(dp_laspy)
    lasdata = next(it)

    assert lasdata.header.version == laspy.header.Version(major=1, minor=0)
    assert lasdata.header.point_format == laspy.point.PointFormat(point_format_id=0)
    assert lasdata.points.array.shape == (11781,)
    assert lasdata.xyz.shape == (11781, 3)
    npt.assert_allclose(
        actual=lasdata.xyz.mean(axis=0),
        desired=[2483799.026934895, 366405.56612511666, 1511.9428214922332],
    )
    npt.assert_allclose(
        actual=np.unique(lasdata.classification), desired=[1, 2, 8, 9, 12, 15]
    )


def test_laspy_reader_copc_http():
    """
    Ensure that LaspyReader works to read in a COPC file (from a HTTP byte stream) and
    outputs a laspy.lasdata.LasData object.
    """
    file_url: str = (
        "https://github.com/laspy/laspy/raw/2.5.3/tests/data/simple_with_page.copc.laz"
    )
    dp = IterableWrapper(iterable=[file_url])
    _, dp_stream = (
        dp.read_from_http()
        .read_from_stream()
        .set_length(length=1)
        .unzip(sequence_length=2)
    )

    # Using class constructors
    dp_laspy = LaspyReader(source_datapipe=dp_stream)
    # Using functional form (recommended)
    dp_laspy = dp_stream.read_from_laspy()

    assert len(dp_laspy) == 1
    it = iter(dp_laspy)
    lasdata = next(it)

    assert lasdata.header.version == laspy.header.Version(major=1, minor=4)
    assert lasdata.header.point_format == laspy.point.PointFormat(point_format_id=7)
    assert lasdata.points.array.shape == (1065,)
    assert lasdata.xyz.shape == (1065, 3)
    npt.assert_allclose(
        actual=lasdata.xyz.mean(axis=0),
        desired=[637296.7351830985, 851249.5384882629, 434.0978403755869],
    )
    npt.assert_allclose(actual=np.unique(lasdata.classification), desired=[1, 2])
