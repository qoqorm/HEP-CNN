#!/bin/bash

if [ ! -d /cvmfs ]; then
	echo "Cannot find CVMFS. Stop."
	exit 1
fi

CMSSW_VERSION=CMSSW_10_6_12
[ -d $CMSSW_VERSION ] || scram project CMSSW $CMSSW_VERSION
cd $CMSSW_VERSION/src
eval `scram runtime -sh`
cd ../..

SRCURL=http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.4.2.tar.gz
TEMP=`basename $SRCURL`
TEMP=`readlink -f ${TEMP/.tar.gz/}`
[ -f ${TEMP}.tar.gz ] || wget $SRCURL
[ -d ${TEMP} ] || tar xzf ${TEMP}.tar.gz
[ -f Delphes ] || ln -sf $TEMP Delphes

cd Delphes
./configure
sed -ie 's;c++0x;c++17;g' Makefile
make -j $(nproc)
cd ..
