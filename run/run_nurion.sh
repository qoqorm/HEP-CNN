#!/bin/bash

#PBS -V
#PBS -N hep-cnn
#PBS -q normal
#PBS -W sandbox=PRIVATE
#PBS -A etc
#PBS -l select=2:ncpus=68:mpiprocs=2:ompthreads=64
#PBS -l walltime=20:00:00

export HDF5_USE_FILE_LOCKING='FALSE'
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_SETTINGS=1
export CUDA_VISIBLE_DEVICES=""

#module load gcc/7.2.0 openmpi/3.1.0 craype-mic-knl tensorflow/1.12.0 hdf5-parallel/1.10.2
source /apps/compiler/intel/19.0.4/impi/2019.4.243/intel64/bin/mpivars.sh relase_mt
source /apps/applications/miniconda3/etc/profile.d/conda.sh
conda activate tf_v1.13

export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=64
MPIPROC=2
NTHREAD=8
BATCH=32
OUTDIR=perf_nurion_KNL/MPIPROC_${MPIPROC}__THREADS_${NTHREAD}__BATCH_${BATCH}

[ _$PBS_O_WORKDIR != _ ] && cd $PBS_O_WORKDIR
[ -d $OUTDIR ] || mkdir -p $OUTDIR
mpirun -np $MPIPROC -env OMP_NUM_THREADS $NTHREAD \
    python train_keras.py -o $OUTDIR \
           --epoch 5 --batch $BATCH \
           -t ../data/NERSC_preproc_base/train.h5 -v ../data/NERSC_preproc_base/val.h5 \


