#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--type', action='store', type=str, choices=('trackpt', 'trackcount'), default='trackcount', help='image type')
parser.add_argument('--device', action='store', type=int, default=-1, help='device name')
args = parser.parse_args()

xmaxs = [5e3, 5e3, 20]
nbinsx = [50, 50, 20]
units = [1e-3, 1e-3, 1]
if args.type == 'trackpt':
    xmaxs[2] = xmaxs[0]
    nbinsx[2] = nbinsx[0]
    units[2] = units[0]

import torch
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    if args.device >= 0: torch.cuda.set_device(args.device)

import sys, os
sys.path.append("../python")
from HEPCNN.dataset_hepcnn import HEPCNNDataset as MyDataset

myDataset = MyDataset()
basedir = os.environ['SAMPLEDIR'] if 'SAMPLEDIR' in  os.environ else "../data/hdf5_32PU_224x224/"
myDataset.addSample("RPV_1400", basedir+"/RPV/Gluino1400GeV/*.h5", weight=0.013/330599)
#myDataset.addSample("QCD_HT700to1000" , basedir+"/QCD/HT700to1000/*/*.h5", weight=???)
myDataset.addSample("QCD_HT1000to1500", basedir+"/QCDBkg/HT1000to1500/*.h5", weight=1094./15466225)
myDataset.addSample("QCD_HT1500to2000", basedir+"/QCDBkg/HT1500to2000/*.h5", weight=99.16/3368613)
myDataset.addSample("QCD_HT2000toInf" , basedir+"/QCDBkg/HT2000toInf/*.h5", weight=20.25/3250016)
myDataset.setProcessLabel("RPV_1400", 1)
myDataset.setProcessLabel("QCD_HT1000to1500", 0) ## This is not necessary because the default is 0
myDataset.setProcessLabel("QCD_HT1500to2000", 0) ## This is not necessary because the default is 0
myDataset.setProcessLabel("QCD_HT2000toInf", 0) ## This is not necessary because the default is 0
myDataset.initialize()

from torch.utils.data import DataLoader
lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(123456)
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())

kwargs = {'pin_memory':True, 'num_workers':4}
trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=False, **kwargs)
valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=False, **kwargs)
testLoader = DataLoader(testDataset, batch_size=args.batch, shuffle=False, **kwargs)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

bins = [None, None, None]
imgHist_val_sig = [np.zeros(nbinsx[i]) for i in range(3)]
imgHist_val_bkg = [np.zeros(nbinsx[i]) for i in range(3)]
imgSum_val_sig, imgSum_val_bkg = None, None
for i, (data, label, weight, rescale) in enumerate(tqdm(valLoader)):
    ws = (weight*rescale).float()

    for image, w in zip(data[label==1], ws[label==1]):
        if imgSum_val_sig == None:
            imgSum_val_sig = torch.zeros(image.shape)
        imgSum_val_sig += image*w

        ww = np.ones(image.shape)*w.numpy()
        for c in range(3):
            y, b = np.histogram(image[c]*units[c], weights=ww[c], bins=nbinsx[c], range=(0.,xmaxs[c]))
            if bins[c] is None: bins[c] = b
            imgHist_val_sig[c] += y

    for image, w in zip(data[label==0], ws[label==0]):
        if imgSum_val_bkg == None:
            imgSum_val_bkg = torch.zeros(image.shape)
        imgSum_val_bkg += image*w

        ww = np.ones(image.shape)*w.numpy()
        for c in range(3):
            y, b = np.histogram(image[c]*units[c], weights=ww[c], bins=nbinsx[c], range=(0.,xmaxs[c]))
            if bins[c] is None: bins[c] = b
            imgHist_val_bkg[c] += y

fig, ax = plt.subplots(1, 3, figsize=(9,3))
for c in range(3):
    ax[c].set_yscale('log')
    ax[c].plot(bins[c][2:], imgHist_val_sig[c][1:], drawstyle='steps-post', label="signal")
    ax[c].plot(bins[c][2:], imgHist_val_bkg[c][1:], drawstyle='steps-post', label="background")
fig.show()

maxZ = [max([x.max() for x in (imgSum_val_sig[c], imgSum_val_bkg[c])]+[2.0]) for c in range(3)]
minZ = [min([x.min()+1e-15 for x in (imgSum_val_sig[c], imgSum_val_bkg[c])]+[1.0, maxZ[c]]) for c in range(3)]
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(2,3, figsize=(9,6))
for c in range(3):
    ax[0,c].imshow(imgSum_val_sig[c], cmap='gist_heat', norm=LogNorm(vmin=minZ[c], vmax=maxZ[c]))
    ax[1,c].imshow(imgSum_val_bkg[c], cmap='gist_heat', norm=LogNorm(vmin=minZ[c], vmax=maxZ[c]))
fig.show()
