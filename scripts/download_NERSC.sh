#!/bin/bash

[ -d data/NERSC ] || mkdir data/NERSC
cd data/NERSC
[ -f val.h5   ] || wget https://portal.nersc.gov/project/mpccc/wbhimji/RPVSusyData/val.h5
[ -f train.h5 ] || wget https://portal.nersc.gov/project/mpccc/wbhimji/RPVSusyData/train.h5
[ -f test.h5  ] || wget https://portal.nersc.gov/project/mpccc/wbhimji/RPVSusyData/test.h5
cd ../..
