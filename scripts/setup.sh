#!/bin/bash

source /apps/applications/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_v1.1.0

export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="hfi1_0:1"
export LD_LIBRARY_PATH=/opt/pbs/lib:$LD_LIBRARY_PATH

