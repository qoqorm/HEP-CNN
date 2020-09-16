#!/bin/bash

source /apps/compiler/intel/19.0.5/impi/2019.5.281/intel64/bin/mpivars.sh release_mt
source /apps/applications/miniconda3/etc/profile.d/conda.sh
module load git craype-mic-knl
module load gcc/8.3.0
export USE_CUDA=0
export USE_MKLDNN=1
export USE_OPENMP=1
export USE_TBB=0

## Just for once
rm -rf /scratch/$USER/conda/nurion_torch
conda create -p /scratch/$USER/conda/nurion_torch
##### Proceed ([y]/n)? y

## Every time
conda activate /scratch/$USER/conda/nurion_torch

## Just for once
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -y h5py tqdm scikit-learn matplotlib pandas

conda install -y -c intel mkl-dnn
conda install -y -c pytorch pytorch=1.6.0 cpuonly
##### Proceed ([y]/n)? y

HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 pip install horovod

