#!/usr/bin/env python
import h5py
import numpy as np
import argparse
import sys, os
import subprocess
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
#num_workers = os.cpu_count()
#torch.omp_set_num_threads(nthreads)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', action='store', type=int, default=50, help='Number of epochs')
parser.add_argument('--ntrain', action='store', type=int, default=-1, help='Number of events for training')
parser.add_argument('--ntest', action='store', type=int, default=-1, help='Number of events for test/validation')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-t', '--trndata', action='store', type=str, required=True, help='input file for training')
parser.add_argument('-v', '--valdata', action='store', type=str, required=True, help='input file for validation')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--lr', action='store', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--noEarlyStopping', action='store_true', help='do not apply Early Stopping')

args = parser.parse_args()

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
weightFile = os.path.join(args.outdir, 'weight.h5')
predFile = os.path.join(args.outdir, 'predict.npy')
historyFile = os.path.join(args.outdir, 'history.csv')
batchHistoryFile = os.path.join(args.outdir, 'batchHistory.csv')
usageHistoryFile = os.path.join(args.outdir, 'usageHistory.csv')

proc = subprocess.Popen(['python', '../scripts/monitor_proc.py', '-t', '1',
                        '-o', usageHistoryFile, '%d' % os.getpid()],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

import time
class TimeHistory():#tf.keras.callbacks.Callback):
    def on_train_begin(self):
        self.times = []
    def on_epoch_begin(self):
        self.epoch_time_start = time.time()
    def on_epoch_end(self):
        self.times.append(time.time() - self.epoch_time_start)

sys.path.append("../scripts")
from monitor_proc import SysStat
sysstat = SysStat(os.getpid(), fileName=batchHistoryFile)
sysstat.update(annotation="start_loggin")

class H5Dataset(Dataset):
    def __init__(self, fileName, nEvent):
        super(H5Dataset, self).__init__()
        print("Opening", fileName, "nEvent=", nEvent)
        self.data = h5py.File(args.trndata, 'r')
        self.images  = self.data['all_events']['images']#[()]
        self.labels  = self.data['all_events']['labels']#[()]
        self.weights = self.data['all_events']['weights']#[()]
        if nEvent > 0:
            self.images  = self.images[:nEvent]
            self.labels  = self.labels[:nEvent]
            self.weights = self.weights[:nEvent]
        print("Convert data to Tensors")
        self.images = torch.Tensor(self.images)
        self.labels = torch.Tensor(self.labels)
        self.weights = torch.Tensor(self.weights)
        self.shape = self.images.shape

        width, height, channel = self.shape[1:]
        self.data_format = 'NHWC'
        if channel > 5:
            channel, width, height = width, height, channel
            self.data_format = 'NCHW'
        self.channel, self.width, self.height = channel, width, height

    def __getitem__(self, idx):
        return (self.images[idx] if self.data_format == 'NCHW' else np.transpose(self.images[idx], (2,1,0)),
                self.labels[idx], self.weights[idx])

    def __len__(self):
        return self.shape[0]

sysstat.update(annotation="open_trn")
trnDataset = H5Dataset(args.trndata, args.ntrain)
sysstat.update(annotation="read_trn")

sysstat.update(annotation="open_val")
valDataset = H5Dataset(args.valdata, args.ntest)
sysstat.update(annotation="read_val")

#if torch.cuda.is_available():
#    num_workers = 1

trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=True, num_workers=num_workers)
valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=True, num_workers=num_workers)

## Build model
sys.path.append("../models")
#if torch.cuda.is_available: kwargs = {'num_workers': 1, 'pin_memory': True}
from HEPCNN.torch_default import MyModel
model = MyModel(trnDataset.width, trnDataset.height)
optm = optim.Adam(model.parameters(), lr=args.lr)
crit = torch.nn.BCELoss()

device = 'cpu'
if torch.cuda.is_available():
    model = model.cuda()
    crit = crit.cuda()
    device = 'cuda'
    #kwargs = {'num_workers': 1, 'pin_memory': True}

sysstat.update(annotation="modelsetup_done")

from tqdm import tqdm
from sklearn.metrics import accuracy_score
bestModel, bestAcc = {}, -1
if not os.path.exists(weightFile):
    try:
        timeHistory = TimeHistory()
        timeHistory.on_train_begin()
        sysstat.update(annotation="train_start")
        history = {'time':[], 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

        for epoch in range(args.epoch):
            timeHistory.on_epoch_begin()

            model.train()
            trn_loss, trn_acc = 0., 0.
            for i, (data, label, weight) in enumerate(tqdm(trnLoader)):
                data = data.float().to(device)
                weight = weight.float().to(device)

                pred = model(data).to('cpu').float()
                loss = crit(pred.view(-1), label.float())
                optm.zero_grad()
                loss.backward()
                optm.step()

                trn_loss += loss.item()
                trn_acc += accuracy_score(label, np.where(pred > 0.5, 1, 0))

                sysstat.update()
            trn_loss /= (i+1)
            trn_acc /= (i+1)

            model.eval()
            val_loss, val_acc = 0., 0.
            for i, (data, label, weight) in enumerate(tqdm(valLoader)):
                data = data.float().to(device)
                weight = weight.float().to(device)

                pred = model(data).to('cpu').float()
                loss = crit(pred.view(-1), label.float())

                val_loss += loss.item()
                val_acc += accuracy_score(label, np.where(pred > 0.5, 1, 0))
            val_loss /= (i+1)
            val_acc /= (i+1)

            if bestAcc < val_acc:
                bestModel = model.state_dict()
                bestAcc = val_acc

            timeHistory.on_epoch_end()
            sysstat.update(annotation='epoch_end')
            history['loss'].append(trn_loss)
            history['acc'].append(trn_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        sysstat.update(annotation="train_end")

        history['time'] = timeHistory.times[:]
        with open(historyFile, 'w') as f:
            writer = csv.writer(f)
            keys = history.keys()
            writer.writerow(keys)
            for row in zip(*[history[key] for key in keys]):
                writer.writerow(row)
        sysstat.update(annotation="wrote_logs")

    except KeyboardInterrupt:
        print("Training finished early")

#model.load_weights(weightFile)
#pred = model.predict(val_images, verbose=1, batch_size=args.batch)

#np.save(predFile, pred)
sysstat.update(annotation="saved_model")

