#!/bin/bash

#PBS -V
#PBS -N hep-cnn
#PBS -q normal
#PBS -W sandbox=PRIVATE
#PBS -A etc
#PBS -l select=1:ncpus=68:mpiprocs=1:ompthreads=64
#PBS -l walltime=20:00:00

export HDF5_USE_FILE_LOCKING='FALSE'
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_SETTINGS=1
export OMP_NUM_THREADS=64
export CUDA_VISIBLE_DEVICES=""

module load gcc/7.2.0 openmpi/3.1.0 craype-mic-knl tensorflow/1.12.0 hdf5-parallel/1.10.2

BATCH=128
OUTDIR=perf_nurion_KNL/KMPBLOCKTIME_${KMP_BLOCKTIME}__OMPNUMTHREADS_${OMP_NUM_THREADS}__batch_${BATCH}
mkdir -p `dirname $OUTDIR`

[ _$PBS_O_WORKDIR != _ ] && cd $PBS_O_WORKDIR
mpirun -np 1 python train_keras.py -o $OUTDIR \
                      --epoch 5 --batch $BATCH --ntrain 8192 --ntest 1024 \
                      -t ../data/NERSC_preproc/train.h5 -v ../data/NERSC_preproc/val.h5 \


