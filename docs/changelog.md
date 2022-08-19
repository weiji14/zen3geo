# Changelog

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

@weiji14

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

@weiji14

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
