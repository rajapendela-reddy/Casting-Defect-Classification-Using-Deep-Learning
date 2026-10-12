# Image dataset

Download [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle. Extract it and locate the directory that directly contains both `train/` and `test/`. The download may contain nested `casting_data/casting_data` folders.

Copy those two folders beneath `data/casting_data`, or pass that existing directory directly using `--data-root`. The required directories are:

| Directory relative to data root | Meaning |
|---|---|
| `train/def_front/` | Defective images in the supplied training partition |
| `train/ok_front/` | Acceptable images in the supplied training partition |
| `test/def_front/` | Defective images in the supplied test partition |
| `test/ok_front/` | Acceptable images in the supplied test partition |

The data root is the **parent of `train/` and `test/`**, not the top-level directory of an unrelated image-only variant. Use the dataset's supplied test split. Do not merge or manually reshuffle its train/test folders if you want the original protocol.

The supplied experiment has 6,633 images under `train/` and 715 under `test/`. Stratified validation uses 995 images from the supplied training folder, leaving 5,638 for optimization.

Original image files were not included in the uploaded project/output archives. This repository contains the original split manifest and sample figures, but does not substitute those figures for the source dataset. Downloaded images are ignored by Git.

Optional official Kaggle CLI download, if it is installed and authenticated:

```bash
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/downloads --unzip
```

Check the extracted directory layout before selecting `--data-root`. Credentials are not included in this repository.
