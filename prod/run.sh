#!/bin/bash

## Find out the job index and which file to process
I=$((${2}+1))
[ $I -gt `cat files.txt | wc -l` ] && exit ## Skip if job index exceeds number of files to process
FILEIN=`cat files.txt | sed -n "${I}p"`

## Set basic envvars - we need cmssw FWLite
DATASET=${1}
WORKDIR=$_CONDOR_JOB_IWD
cd /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_10_6_12
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`
cd $TEMP
OUTDIR=$WORKDIR/../data/CMSDelphes/$DATASET
if [ ! -d $OUTDIR ]; then
    mkdir -p $OUTDIR/64x64_noPU
    mkdir -p $OUTDIR/64x64_32PU
    mkdir -p $OUTDIR/224x224_noPU
    mkdir -p $OUTDIR/224x224_32PU
fi

## Print out what to do in this job.
echo "+ Job index=$I"
echo "+ CMSSW=$CMSSW_VERSION"
echo "+ FILEIN=$FILEIN"
echo "+ OUTDIR=$OUTDIR"

## Move to the Delphes directory
df -h $WORKDIR
df -h $TEMP

## Run Delphes and do the simulation
cd $WORKDIR/Delphes
./DelphesCMSFWLite $WORKDIR/cards/CMS_noPU.tcl $TEMP/delphes_noPU_${I}.root $FILEIN
./DelphesCMSFWLite $WORKDIR/cards/CMS_32PU.tcl $TEMP/delphes_32PU_${I}.root $FILEIN

## Run the scripts to produce images
HEPCNNARGS="--input-type delphes --output-h5"
cd $WORKDIR/atlas_dl/scripts

echo $TEMP/delphes_noPU_${I}.root > $TEMP/filelist.txt
python ./prepare_data.py $HEPCNNARGS --bins  64 $OUTDIR/64x64_noPU/hepcnn_${I}.h5 $TEMP/filelist.txt
python ./prepare_data.py $HEPCNNARGS --bins 224 $OUTDIR/224x224_noPU/hepcnn_${I}.h5 $TEMP/filelist.txt

echo $TEMP/delphes_32PU_${I}.root > $TEMP/filelist.txt
python ./prepare_data.py $HEPCNNARGS --bins  64 $OUTDIR/64x64_32PU/hepcnn_${I}.h5 $TEMP/filelist.txt
python ./prepare_data.py $HEPCNNARGS --bins 224 $OUTDIR/224x224_32PU/hepcnn_${I}.h5 $TEMP/filelist.txt

rm -f $TEMP/*
