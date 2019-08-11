#!/bin/bash
#PBS -V
#PBS -N pytorch_mpitest
#PBS -q normal
####PBS -l select=16:ncpus=66:mpiprocs=64
#PBS -l select=64:ncpus=16:mpiprocs=16:ompthreads=16
#PBS -W sandbox=PRIVATE
#PBS -A etc

env
cat /proc/cpuinfo | grep process
echo "NCPUS=" `nproc` $NCPUS
cd $PBS_O_WORKDIR

if [ $# -ge 1 ]; then
  NTOTALMPI=$1
else
  NTOTALMPI=$((64*16))
fi

source ../scripts/setup.sh
mpirun -np $NTOTALMPI python pytorch-mpicall-only.py
