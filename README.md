# HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods
This repository contains code of the paper [`HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods`](https://arxiv.org/abs/2306.00385) presented at the IEEE International Geoscience and Remote Sensing Symposium (IGARSS) in July 2023. This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Martin Hermann Paul Fuchs](https://rsim.berlin/team/members/martin-hermann-paul-fuchs) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> M. H. P. Fuchs and B. Demir, "[HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods,](https://arxiv.org/abs/2306.00385)" IEEE International Geoscience and Remote Sensing Symposium, Pasadena, CA, USA, 2023, pp. 1779-1782, doi: 10.1109/IGARSS52108.2023.10283385.
```
@INPROCEEDINGS{10283385,
  author={Fuchs, Martin Hermann Paul and Demir, Begüm},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
  year={2023},
  volume={},
  number={},
  pages={1779-1782},
  doi={10.1109/IGARSS52108.2023.10283385}}
```
This repository contains code that has been adapted from the CompressAI [\[2\]](#2-compressai) framework https://github.com/InterDigitalInc/CompressAI/.

## Description
The development of learning-based hyperspectral image compression methods has recently attracted great attention in remote sensing. Such methods require a high number of hyperspectral images to be used during training to optimize all parameters and reach a high compression performance. However, existing hyperspectral datasets are not sufficient to train and evaluate learning-based compression methods, which hinders the research in this field. To address this problem, in this paper we present HySpecNet-11k that is a large-scale hyperspectral benchmark dataset made up of 11,483 nonoverlapping image patches. Each patch is a portion of 128 × 128 pixels with 224 spectral bands and a ground sample distance of 30 m. We exploit HySpecNet-11k to benchmark the current state of the art in learning-based hyperspectral image compression by focussing our attention on various 1D, 2D and 3D convolutional autoencoder architectures. Nevertheless, HySpecNet-11k can be used for any unsupervised learning task in the framework of hyperspectral image analysis. The dataset, our code and the pre-trained weights are publicly available at [https://hyspecnet.rsim.berlin](https://hyspecnet.rsim.berlin).

## Setup
The code in this repository is tested with `Ubuntu 22.04 LTS` and `Python 3.10.6`.

### Dependencies
All dependencies are listed in the [`requirements.txt`](requirements.txt) and can be installed via the following command:
```
pip install -r requirements.txt
```

### Dataset
HySpecNet-11k is made up of image patches acquired by the Environmental Mapping and Analysis Program (EnMAP) [\[1\]](#1-environmental-mapping-and-analysis-program-enmap) satellite.

Follow the instructions on [https://hyspecnet.rsim.berlin](https://hyspecnet.rsim.berlin) to download, extract and preprocess the HySpecNet-11k dataset.

The folder structure should be as follows:
```
┗ 📂 hsi-compression/
  ┗ 📂 datasets/
    ┗ 📂 hyspecnet-11k/
      ┣ 📂 patches/
      ┃ ┣ 📂 tile_001/
      ┃ ┃ ┣ 📂 tile_001-patch_01/
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-DATA.npy
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_PIXELMASK.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLASSES.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLOUD.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_HAZE.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_SNOW.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_SWIR.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-QL_VNIR.TIF
      ┃ ┃ ┃ ┣ 📜 tile_001-patch_01-SPECTRAL_IMAGE.TIF
      ┃ ┃ ┃ ┗ 📜 tile_001-patch_01-THUMBNAIL.jpg
      ┃ ┃ ┣ 📂 tile_001-patch_02/
      ┃ ┃ ┃ ┗ 📜 ...
      ┃ ┃ ┗ 📂 ...
      ┃ ┣ 📂 tile_002/
      ┃ ┃ ┗ 📂 ...
      ┃ ┗ 📂 ...
      ┗ 📂 splits/
        ┣ 📂 easy/
        ┃ ┣ 📜 test.csv
        ┃ ┣ 📜 train.csv
        ┃ ┗ 📜 val.csv
        ┣ 📂 hard/
        ┃ ┣ 📜 test.csv
        ┃ ┣ 📜 train.csv
        ┃ ┗ 📜 val.csv
        ┗ 📂 ...
```

## Usage

### Train
The [`train.py`](train.py) expects the following command line arguments:
| Parameter | Description | Default |
| :- | :- | :- |
| `--devices` | Devices to use, e.g. `cpu` or `0` or `0,2,5,7` | `0` |
| `--train-batch-size` | Training batch size | `2` |
| `--val-batch-size` | Validation batch size | `4` |
| `-n` | Data loaders threads | `4` |
| `-d` | Path to dataset | `./datasets/hyspecnet-11k/` |
| `--mode` | Dataset split difficulty | `easy` |
| `-m` | Model architecture | `cae1d` |
| `--loss` | Loss | `mse` |
| `-e` | Number of epochs | `500` |
| `-lr` | Learning rate | `1e-4` |
| `--save-dir` | Directory to save results | `./results/trains/` |
| `--seed` | Set random seed for reproducibility | `10587` |
| `--clip-max-norm` | Gradient clipping max norm | `1.0` |
| `--checkpoint` | Path to a checkpoint to resume training | `None` |

Specify the parameters in the [`train.sh`](train.sh) file and then execute the following command:
```console
./train.sh
```
Or run the python code directly through the console:
```console
python train.py \
    --devices 0 \
    --train-batch-size 2 \
    --val-batch-size 4 \
    --num-workers 4 \
    --learning-rate 1e-4 \
    --mode easy \
    --model cae1d \
    --loss mse \
    --epochs 500
```
### Test
The [`test.py`](test.py) expects the following command line arguments:
| Parameter | Description | Default |
| :- | :- | :- |
| `--device` | Device to use (default: 0), e.g. `cpu` or `0` | `0` |
| `--batch-size` | Test batch size | `64` |
| `-n` | Data loaders threads | `0` |
| `-d` | Path to dataset | `./datasets/hyspecnet-11k/` |
| `--mode` | Dataset split difficulty | `easy` |
| `-m` | Model architecture | `cae1d` |
| `--checkpoint` | Path to the checkpoint to evaluate | `None` |
| `--half` | Convert model to half floating point (fp16) | `False` |
| `--save-dir` | Directory to save results | `./results/tests/` |
| `--seed` | Set random seed for reproducibility | `10587` |

Specify the parameters in the [`test.sh`](test.sh) file and then execute the following command:
```console
./test.sh
```
Or run the python code directly through the console:
```console
python test.py \
    --device 0 \
    --batch-size 4 \
    --num-workers 4 \
    --mode easy \
    --model cae1d \
    --checkpoint ./results/weights/cae1d_8bpppc.pth.tar
```

## Pre-Trained Weights
Pre-trained weights are publicly available and should be downloaded into the [`./results/weights/`](results/weights/) folder.

| Method | Model | Rate | PSNR | Download Link |
| :----- | :---- | :--- | :--- | :------------ |
| 1D-CAE [\[3\]](#3-1d-convolutional-autoencoder-1d-cae) | `cae1d_1bpppc` | 1.11 bpppc | 48.95 dB | [cae1d_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/ew2jr67yro7cj3x/download/cae1d_1bpppc.pth.tar) |
| | `cae1d_2bpppc` | 2.06 bpppc | 52.38 dB | [cae1d_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/Ae35EBRado8QSmk/download/cae1d_2bpppc.pth.tar) |
| | `cae1d_4bpppc` | 4.12 bpppc | 53.90 dB | [cae1d_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/ZNeXycsssRdYZ5m/download/cae1d_4bpppc.pth.tar) |
| | `cae1d` | 8.08 bpppc | 54.85 dB | [cae1d_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/GpmXDAWEeo2nG5w/download/cae1d_8bpppc.pth.tar) |
| 1D-CAE-Adv [\[4\]](#4-advanced-1d-convolutional-autoencoder-1d-cae-adv) | `cae1da` | 8.08 bpppc | 54.18 dB | [cae1da_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/zYagFRJjTrgNQwJ/download/cae1da_8bpppc.pth.tar) |
| 1D-CAE-Ext [\[5\]](#5-extended-1d-convolutional-autoencoder-1d-cae-ext) | `cae1de` | 8.08 bpppc | 43.08 dB | [cae1de_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/3xF7QrDz3JQJpg6/download/cae1de_8bpppc.pth.tar) |
| SSCNet [\[6\]](#6-spectral-signals-compressor-network-sscnet) | `sscnet_1bpppc` | 1.00 bpppc | 43.24 dB | [sscnet_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/wPwbMKYJAmXxLRX/download/sscnet_1bpppc.pth.tar) |
| | `sscnet_2bpppc` | 2.02 bpppc | 43.60 dB | [sscnet_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/H9Yg8n8rzxGMe2Z/download/sscnet_2bpppc.pth.tar) |
| | `sscnet` | 2.53 bpppc | 43.64 dB | [sscnet_2point5bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/igLNmTG9kExngz9/download/sscnet_2point5bpppc.pth.tar) |
| | `sscnet_4bpppc` | 4.00 bpppc | 43.69 dB | [sscnet_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/WQ65aCDxgedQYxZ/download/sscnet_4bpppc.pth.tar) |
| | `sscnet_6bpppc` | 6.00 bpppc | 43.52 dB | [sscnet_6bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/gAyxFNSbx4aA5sa/download/sscnet_6bpppc.pth.tar) |
| | `sscnet_8bpppc` | 8.08 bpppc | 43.29 dB | [sscnet_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/5kiQ8ZLRnkpbSg6/download/sscnet_8bpppc.pth.tar) |
| 3D-CAE [\[7\]](#7-3d-convolutional-auto-encoder-3d-cae) | `cae3d_1bpppc` | 1.01 bpppc | 39.06 dB | [cae3d_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/QDfARfWL3Pab3xK/download/cae3d_1bpppc.pth.tar) |
| | `cae3d` | 2.02 bpppc | 39.54 dB | [cae3d_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/dD3qtjrgzJxmymP/download/cae3d_2bpppc.pth.tar) |
| | `cae3d_4bpppc` | 4.04 bpppc | 39.69 dB | [cae3d_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/CmTdQzcE3x9pEEJ/download/cae3d_4bpppc.pth.tar) |
| | `cae3d_8bpppc` | 8.08 bpppc | 39.94 dB | [cae3d_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/DpqKJdMbojF3CLx/download/cae3d_8bpppc.pth.tar) |

## Authors
**Martin Hermann Paul Fuchs**
https://rsim.berlin/team/members/martin-hermann-paul-fuchs

## License
The code in this repository is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2023 Martin Hermann Paul Fuchs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## References
### [1] [Environmental Mapping and Analysis Program (EnMAP)](https://doi.org/10.3390/rs70708830)

### [2] [CompressAI](https://doi.org/10.48550/arXiv.2011.03029)

### [3] [1D-Convolutional Autoencoder (1D-CAE)](https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021)

### [4] [Advanced 1D-Convolutional Autoencoder (1D-CAE-Adv)](https://doi.org/10.1109/WHISPERS56178.2022.9955109)

### [5] [Extended 1D-Convolutional Autoencoder (1D-CAE-Ext)](https://doi.org/10.1117/12.2636129)

### [6] [Spectral Signals Compressor Network (SSCNet)](https://doi.org/10.3390/rs14102472)

### [7] [3D Convolutional Auto-Encoder (3D-CAE)](https://doi.org/10.1117/1.JEI.30.4.041403)
