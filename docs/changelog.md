# Changelog

## Release v0.6.0 (2023/04/18)

### 💫 Highlights

* 🎉 **Sixth release of zen3geo** 🎉
* 🚸 Walkthrough on handling multi-resolution climate data ([#91](https://github.com/weiji14/zen3geo/pull/91))

### 🚀 Features

* ✨ XpySTACAssetReader for reading COG, NetCDF & Zarr STAC assets ([#87](https://github.com/weiji14/zen3geo/pull/87))
* ✨ Implement len function for XbatcherSlicerIterDataPipe ([#75](https://github.com/weiji14/zen3geo/pull/75))

### 📖 Documentation

* ♻️ Use xarray.merge with join="override" in collate functions ([#72](https://github.com/weiji14/zen3geo/pull/72))

### 🧰 Maintenance

* ⬆️ Bump jupyter-book from 0.14.0 to 0.15.1 ([#94](https://github.com/weiji14/zen3geo/pull/94))
* 📦️ Publish to TestPyPI and PyPI via OpenID Connect token ([#90](https://github.com/weiji14/zen3geo/pull/90))
* 👷 NEP29: Run Continuous Integration on Python 3.11 ([#89](https://github.com/weiji14/zen3geo/pull/89))
* ⬆️ Bump jupyter-book from 0.13.0 to 0.14.0 ([#85](https://github.com/weiji14/zen3geo/pull/85))
* 📌 Pin maximum python version to <4.0 ([#78](https://github.com/weiji14/zen3geo/pull/78))
* ⬆️ Bump poetry from 1.2.0 to 1.3.0 ([#77](https://github.com/weiji14/zen3geo/pull/77))
* 📌 Pin minimum xbatcher version to 0.2.0 ([#73](https://github.com/weiji14/zen3geo/pull/73))

### 🧑‍🤝‍🧑 Contributors

[@dependabot[bot]](https://github.com/dependabot-bot) and [@weiji14](https://github.com/weiji14)

---

## Release v0.5.0 (2022/09/26)

### 💫 Highlights

* 🎉 **Fifth release of zen3geo** 🎉
* 🚸 Walkthrough on stacking time-series earth observation data ([#62](https://github.com/weiji14/zen3geo/pull/62))

### 🚀 Features

* ✨ StackSTACMosaicIterDataPipe to mosaic tiles into one piece ([#63](https://github.com/weiji14/zen3geo/pull/63))
* ✨ StackSTACStackerIterDataPipe for stacking STAC items ([#61](https://github.com/weiji14/zen3geo/pull/61))
* ✨ PySTACAPISearchIterDataPipe to query dynamic STAC Catalogs ([#59](https://github.com/weiji14/zen3geo/pull/59))
* ✨ PySTACItemReaderIterDataPipe for reading STAC Items ([#46](https://github.com/weiji14/zen3geo/pull/46))

### 📖 Documentation

* 🚚 Rename to PySTACAPISearcher and StackSTACMosaicker ([#64](https://github.com/weiji14/zen3geo/pull/64))

### 🧰 Maintenance

* 📌 Pin min pystac-client and stackstac to v0.4.0, pystac to 1.4.0 ([#66](https://github.com/weiji14/zen3geo/pull/66))
* 📦️ Exclude tests from source distribution and binary wheel ([#58](https://github.com/weiji14/zen3geo/pull/58))

### 🧑‍🤝‍🧑 Contributors

[@dependabot[bot]](https://github.com/dependabot-bot) and [@weiji14](https://github.com/weiji14)

---

## Release v0.4.0 (2022/09/08)

### 💫 Highlights

* 🎉 **Fourth release of zen3geo** 🎉
* 🚸 Walkthrough on object detection with bounding boxes ([#49](https://github.com/weiji14/zen3geo/pull/49))

### 🚀 Features

* ✨ GeoPandasRectangleClipper for spatially subsetting vectors ([#52](https://github.com/weiji14/zen3geo/pull/52))

### 📖 Documentation

* 📝 Add install from conda-forge instructions ([#55](https://github.com/weiji14/zen3geo/pull/55))
* ✏️ Edit docs to use OGC:CRS84 lon/lat instead of EPSG:4326 ([#45](https://github.com/weiji14/zen3geo/pull/45))
* 💡 Warn about overlapping strides if followed by train/val split ([#43](https://github.com/weiji14/zen3geo/pull/43))

### 🧰 Maintenance

* ⬆️ Bump poetry from 1.2.0rc1 to 1.2.0 ([#47](https://github.com/weiji14/zen3geo/pull/47))
* ⬆️ Bump poetry from 1.2.0b3 to 1.2.0rc1 ([#44](https://github.com/weiji14/zen3geo/pull/44))

### 🧑‍🤝‍🧑 Contributors

[@dependabot[bot]](https://github.com/dependabot-bot) and [@weiji14](https://github.com/weiji14)

---

## Release v0.3.0 (2022/08/19)

### 💫 Highlights

* 🎉 **Third release of zen3geo** 🎉
* 🚸 Walkthrough on rasterizing vector polygons into label masks ([#31](https://github.com/weiji14/zen3geo/pull/31))

### 🚀 Features

* ✨ DatashaderRasterizer for burning vector shapes to xarray grids ([#35](https://github.com/weiji14/zen3geo/pull/35))
* ✨ XarrayCanvasIterDataPipe for creating blank datashader canvas ([#34](https://github.com/weiji14/zen3geo/pull/34))
* ♻️ Let PyogrioReader return geodataframe only instead of tuple ([#33](https://github.com/weiji14/zen3geo/pull/33))

### 🐛 Bug Fixes

* ♻️ Refactor DatashaderRasterizer to be up front about datapipe lengths ([#39](https://github.com/weiji14/zen3geo/pull/39))
* 🩹 Raise ModuleNotFoundError when xbatcher not installed ([#37](https://github.com/weiji14/zen3geo/pull/37))

### 📖 Documentation

* 📝 Improve pip install zen3geo instructions with extras dependencies ([#40](https://github.com/weiji14/zen3geo/pull/40))
* 🔍 Show more levels for the in-page table of contents ([#36](https://github.com/weiji14/zen3geo/pull/36))

### 🧑‍🤝‍🧑 Contributors

[@weiji14](https://github.com/weiji14)

---

## Release v0.2.0 (2022/07/17)

### 💫 Highlights

* 🎉 **Second release of zen3geo** 🎉
* 🚸 Walkthrough on creating batches of data chips ([#20](https://github.com/weiji14/zen3geo/pull/20))

### 🚀 Features

* ♻️ Let RioXarrayReader return dataarray only instead of tuple ([#24](https://github.com/weiji14/zen3geo/pull/24))
* ✨ XbatcherSlicerIterDataPipe for slicing xarray.DataArray ([#22](https://github.com/weiji14/zen3geo/pull/22))
* ✨ PyogrioReaderIterDataPipe for reading vector OGR files ([#19](https://github.com/weiji14/zen3geo/pull/19))

### 📖 Documentation

* 🎨 Extra subsection for rioxarray datapipes ([#18](https://github.com/weiji14/zen3geo/pull/18))

### 🧰 Maintenance

* 👷 NEP29: Run CI and Docs build on Python 3.10 ([#29](https://github.com/weiji14/zen3geo/pull/29))
* ⬆️ Bump poetry from 1.2.0b2 to 1.2.0b3 ([#28](https://github.com/weiji14/zen3geo/pull/28))
* 📌 Pin minimum torchdata version to 0.4.0 ([#25](https://github.com/weiji14/zen3geo/pull/25))
* 📌 Pin minimum pyogrio version to 0.4.0 ([#21](https://github.com/weiji14/zen3geo/pull/21))

### 🧑‍🤝‍🧑 Contributors

[@weiji14](https://github.com/weiji14)

---

## Release v0.1.0 (2022/06/08)

### 💫 Highlights

* 🎉 **First release of zen3geo** 🎉
* 🚸 Walkthrough on using RioXarray IterDataPipes at https://zen3geo.readthedocs.io/en/latest/walkthrough.html ([#8](https://github.com/weiji14/zen3geo/pull/8))

### 🚀 Features

* ✨ Introducing RioXarrayReaderIterDataPipe for reading GeoTIFFs ([#6](https://github.com/weiji14/zen3geo/pull/6))

### 📖 Documentation

* 🔧 Configure readthedocs documentation build ([#13](https://github.com/weiji14/zen3geo/pull/13))
* 💬 Show how to convert xarray.DataArray to torch.Tensor ([#9](https://github.com/weiji14/zen3geo/pull/9))
* 📝 Add basic installation instructions ([#7](https://github.com/weiji14/zen3geo/pull/7))
* 👥 Healthy community standards ([#4](https://github.com/weiji14/zen3geo/pull/4))

### 🧰 Maintenance

* 📦 Publish to TestPyPI and PyPI using GitHub Actions ([#14](https://github.com/weiji14/zen3geo/pull/14))
* 🧑‍💻 Draft changelog with Release Drafter GitHub Actions ([#11](https://github.com/weiji14/zen3geo/pull/11))
* 👷 Setup GitHub Actions Continuous Integration tests ([#2](https://github.com/weiji14/zen3geo/pull/2))
* 🌱 Initialize pyproject.toml file ([#1](https://github.com/weiji14/zen3geo/pull/1))

### 🧑‍🤝‍🧑 Contributors

[@weiji14](https://github.com/weiji14)
