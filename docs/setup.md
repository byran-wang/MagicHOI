# Getting Started

General Requirements:

- Ubuntu 20.04.6 LTS
- Python 3.9
- torch 2.0.1
- CUDA 11.8 (check nvcc --version)
- pytorch3d 0.7.4
- pytorch-lightning 2.3.0
- aitviewer 1.13.0

These requirements are non-strict. However, they have been tested on our end. Therefore, it is a good starting point. 

## Preliminary

```bash
sudo apt-get install ffmpeg # needed to convert rendered files to mp4
```

### CUDA

Before starting, check your CUDA `nvcc` version:

```bash
nvcc --version # should be 11.8
```

You can install nvcc and cuda via [runfile](https://developer.nvidia.com/cuda-11-8-0-download-archive). If `nvcc --version` is still not `11.8`, check whether you are referring the right nvcc with `which nvcc`. Assuming you have an NVIDIA driver installed, usually, you only need to run the following command to install `nvcc` (as an example):

```bash
# Install TOOLKIT ONLY (no driver)
sudo bash cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
```

After the installation, make sure the paths pointing to the current cuda toolkit location. For example:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export CPATH="/usr/local/cuda-11.8/include:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64/"
```

## MagicHOI environment

Install packages: 

```shell
# -- conda
ENV_NAME=MagicHOI
conda create -n $ENV_NAME python=3.9
conda activate $ENV_NAME

# -- torch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# -- requirements
pip install -r requirements.txt
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub




# --- pytroch3d
# ref https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
!ipython  # run on the terminal
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html

# # --- pytroch3d
# git clone https://github.com/facebookresearch/pytorch3d.git 
# cd pytorch3d
# git checkout 35badc08
# python setup.py install
# cd ..
# --- kaolin
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html

# # --- kaolin
# git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
# cd kaolin
# git checkout v0.10.0
# python setup.py install
# cd ../../


# --- smplx (custom)
git clone https://github.com/zc-alexfan/smplx.git
cd smplx
python setup.py install
cd ..

# --- hloc (custom)
cd third_party/Hierarchical-Localization_hold/
python -m pip install -e .
cd ../../


# -- override non-compatible packages
pip install setuptools==75.1.0
pip install numpy==1.23.1
pip install scipy==1.10.1
pip install scikit-image==0.21.0
```

Setup viewer:

```bash
# aitviewer
pip install aitviewer==1.13.0
```