#!/bin/bash

CMSSW_VERSION=CMSSW_10_6_1
[ -d $CMSSW_VERSION ] || scram project CMSSW $CMSSW_VERSION
cd $CMSSW_VERSION/src
eval `scram runtime -sh`
cd ../..

SRCURL=http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.4.1.tar.gz
TEMP=`basename $SRCURL`

[ -f $TEMP ] || wget $SRCURL
tar xzf $TEMP
TEMP=`readlink -f ${TEMP/.tar.gz/}`
ln -sf $TEMP Delphes

cd Delphes
./configure
sed -ie 's;c++0x;c++17;g' Makefile
make -j $(nproc)
cd ..
