#!/usr/bin/env python
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--type', action='store', type=str, choices=('trackpt', 'trackcount'), default='trackcount', help='image type')
parser.add_argument('--device', action='store', type=int, default=-1, help='device name')
parser.add_argument('--noimage', action='store_true', default=False, help='draw event image')
parser.add_argument('-c', '--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')

args = parser.parse_args()
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

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
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    myDataset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    myDataset.setProcessLabel(name, sampleInfo['label'])
myDataset.initialize()

procNames = myDataset.sampleInfo['procName'].unique()

from torch.utils.data import DataLoader
lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())

kwargs = {'num_workers':config['training']['nDataLoaders']}
allLoader = DataLoader(myDataset, batch_size=args.batch, shuffle=False, **kwargs)
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
sumE_sig, sumR_sig, sumW_sig = 0., 0., 0.
sumE_bkg, sumR_bkg, sumW_bkg = 0., 0., 0.
sumEs, sumRs, sumWs = {}, {}, {}
for procName in procNames:
    sumEs[procName] = 0.
    sumRs[procName] = 0.
    sumWs[procName] = 0.

for i, (data, labels, weights, rescales, procIdxs) in enumerate(tqdm(allLoader)):
    ws = (weights*rescales).float()

    for procIdx, procName in enumerate(procNames):
        ww = ws[procIdxs==procIdx]
        sumEs[procName] += len(ww)
        sumRs[procName] += ww.sum()
        sumWs[procName] += weights[procIdxs==procIdx].sum()

        ww_sig = ws[(procIdxs==procIdx) & (labels==1)]
        ww_bkg = ws[(procIdxs==procIdx) & (labels==0)]
        sumE_sig += len(ww_sig)
        sumE_bkg += len(ww_bkg)
        sumR_sig += ww_sig.sum()
        sumR_bkg += ww_bkg.sum()
        sumW_sig += weights[(procIdxs==procIdx) & (labels==1)].sum()
        sumW_bkg += weights[(procIdxs==procIdx) & (labels==0)].sum()

    if args.noimage: continue

    for image, w in zip(data[labels==1], ws[labels==1]):
        if imgSum_val_sig == None:
            imgSum_val_sig = torch.zeros(image.shape)
        imgSum_val_sig += image*w

        ww = np.ones(image.shape)*w.numpy()
        for c in range(3):
            y, b = np.histogram(image[c]*units[c], weights=ww[c], bins=nbinsx[c], range=(0.,xmaxs[c]))
            if bins[c] is None: bins[c] = b
            imgHist_val_sig[c] += y

    for image, w in zip(data[labels==0], ws[labels==0]):
        if imgSum_val_bkg == None:
            imgSum_val_bkg = torch.zeros(image.shape)
        imgSum_val_bkg += image*w

        ww = np.ones(image.shape)*w.numpy()
        for c in range(3):
            y, b = np.histogram(image[c]*units[c], weights=ww[c], bins=nbinsx[c], range=(0.,xmaxs[c]))
            if bins[c] is None: bins[c] = b
            imgHist_val_bkg[c] += y

print("-"*80)
print("sumEvent : signal=%d bkg=%d" % (sumE_sig, sumE_bkg))
print("sumResWgt: signal=%g bkg=%g" % (sumR_sig, sumR_bkg))
print("sumWeight: signal=%g bkg=%g" % (sumW_sig, sumW_bkg))
for procName in procNames:
    print("proc=%s sumE=%d sumR=%g sumW=%g" % (procName, sumEs[procName], sumRs[procName], sumWs[procName]))
print("-"*80)
print("sum=", sum(sumEs.values()), sum(sumWs.values()).item())
print("="*80)
exit()

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
