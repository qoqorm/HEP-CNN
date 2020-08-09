#!/usr/bin/env python
import numpy as np
import argparse
import sys, os
import subprocess
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
torch.set_num_threads(nthreads)

parser = argparse.ArgumentParser()
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-d', '--input', action='store', type=str, required=True, help='directory with pretrained model parameters')
parser.add_argument('--model', action='store', choices=('none', 'default', 'log3ch', 'log5ch', 'original', 'circpad', 'circpadlog3ch', 'circpadlog5ch'),
                               default='none', help='choice of model')
parser.add_argument('--device', action='store', type=int, default=-1, help='device name')

args = parser.parse_args()

predFile = args.input+'/prediction.csv'
import pandas as pd

sys.path.append("../python")

print("Load data", end='')
from HEPCNN.dataset_hepcnn import HEPCNNDataset as MyDataset

myDataset = MyDataset()
basedir = os.environ['SAMPLEDIR'] if 'SAMPLEDIR' in  os.environ else "../data/hdf5_32PU_224x224/"
basedir+='/'
myDataset.addSample("RPV_1400", basedir+"RPV/Gluino1400GeV/*.h5", weight=0.013/330599)
#myDataset.addSample("QCD_HT700to1000" , basedir+"QCD/HT700to1000/*/*.h5", weight=???)
myDataset.addSample("QCD_HT1000to1500", basedir+"QCDBkg/HT1000to1500/*.h5", weight=1094./15466225)
myDataset.addSample("QCD_HT1500to2000", basedir+"QCDBkg/HT1500to2000/*.h5", weight=99.16/3199737)
myDataset.addSample("QCD_HT2000toInf" , basedir+"QCDBkg/HT2000toInf/*.h5", weight=20.25/1520178)
myDataset.setProcessLabel("RPV_1400", 1)
myDataset.setProcessLabel("QCD_HT1000to1500", 0) ## This is not necessary because the default is 0
myDataset.setProcessLabel("QCD_HT1500to2000", 0) ## This is not necessary because the default is 0
myDataset.setProcessLabel("QCD_HT2000toInf", 0) ## This is not necessary because the default is 0
myDataset.initialize()
print("done")

print("Split data", end='')
lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(123456)
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())
print("done")

if args.device >= 0: torch.cuda.set_device(args.device)
kwargs = {'num_workers':min(4, nthreads)}
if torch.cuda.is_available():
    #if hvd: kwargs['num_workers'] = 1
    kwargs['pin_memory'] = True

testLoader = DataLoader(testDataset, batch_size=args.batch, shuffle=False, **kwargs)

print("Load model", end='')
if args.model == 'none':
    print("Load saved model from", (args.input+'/model.pkl'))
    model = torch.load(args.input+'/model.pkl', map_location='cpu')
else:
    print("Load the model", args.model)
    if args.model == 'original':
        from HEPCNN.torch_model_original import MyModel
    elif 'circpad' in args.model:
        from HEPCNN.torch_model_circpad import MyModel
    else:
        from HEPCNN.torch_model_default import MyModel
    model = MyModel(testDataset.width, testDataset.height, model=args.model)

device = 'cpu'
if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
print('done')

model.load_state_dict(torch.load(args.input+'/weight_0.pkl', map_location='cpu'))
print('modify model', end='')
model.fc.add_module('output', torch.nn.Sigmoid())
model.eval()
print('done')

from tqdm import tqdm
labels, preds = [], []
weights, scaledWeights = [], []
for i, (data, label, weight, rescale) in enumerate(tqdm(testLoader)):
    data = data.float().to(device)
    weight = weight.float()
    pred = model(data).detach().to('cpu').float()

    labels.extend([x.item() for x in label])
    preds.extend([x.item() for x in pred.view(-1)])
    weights.extend([x.item() for x in weight.view(-1)])
    scaledWeights.extend([x.item() for x in (weight*rescale).view(-1)])
df = pd.DataFrame({'label':labels, 'prediction':preds,
                   'weight':weights, 'scaledWeight':scaledWeights})
df.to_csv(predFile, index=False)

from sklearn.metrics import roc_curve, roc_auc_score
df = pd.read_csv(predFile)
tpr, fpr, thr = roc_curve(df['label'], df['prediction'], sample_weight=df['weight'], pos_label=0)
auc = roc_auc_score(df['label'], df['prediction'], sample_weight=df['weight'])

import matplotlib.pyplot as plt
print(df.keys())
df_bkg = df[df.label==0]
df_sig = df[df.label==1]

hbkg1 = df_bkg['prediction'].plot(kind='hist', histtype='step', weights=df_bkg['weight'], bins=50, alpha=0.7, color='red', label='QCD')
hsig1 = df_sig['prediction'].plot(kind='hist', histtype='step', weights=df_sig['weight'], bins=50, alpha=0.7, color='blue', label='RPV')
plt.yscale('log')
plt.ylabel('Events/(100)/(fb-1)')
plt.legend()
plt.show()

hbkg2 = df_bkg['prediction'].plot(kind='hist', histtype='step', weights=df_bkg['scaledWeight'], bins=50, alpha=0.7, color='red', label='QCD')
hsig2 = df_sig['prediction'].plot(kind='hist', histtype='step', weights=df_sig['scaledWeight'], bins=50, alpha=0.7, color='blue', label='RPV')
#plt.yscale('log')
plt.ylabel('Arbitrary Unit')
plt.legend()
plt.show()

plt.plot(fpr, tpr, '.-', label='%s %.3f' % (args.input, auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.xlim(0, 0.001)
plt.xlim(0, 1.000)
plt.ylim(0, 1.000)
plt.legend()
plt.show()

