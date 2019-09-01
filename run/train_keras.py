#!/usr/bin/env python
import h5py
import numpy as np
import argparse
import sys, os
import subprocess
import csv

import tensorflow as tf

config = tf.ConfigProto()
nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
## From Nurion user guide, intra=1, inter=n_physical_core
config.intra_op_parallelism_threads = 1 ## for independent graph computations
config.inter_op_parallelism_threads = nthreads ## for operations which can run in parallel such as matmul or reduction
## From TF performance manual page, intra=inter=n_physical_core or n_logical_core
#config.intra_op_parallelism_threads = nthreads
#config.inter_op_parallelism_threads = nthreads
tf.Session(config=config)

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
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.times = []
    def on_epoch_begin(self, batch, logs):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs):
        self.times.append(time.time() - self.epoch_time_start)

sys.path.append("../scripts")
from monitor_proc import SysStat
class SysStatHistory(tf.keras.callbacks.Callback, SysStat):
    def __init__(self, pid):
        SysStat.__init__(self, pid, fileName=batchHistoryFile)
    def on_epoch_end(self, batch, logs):
        self.update(annotation='epoch_end')
    def on_batch_end(self, batch, logs):
        self.update()
sysstat = SysStatHistory(os.getpid())
sysstat.update(annotation="start_logging")

sys.path.append("../python")
from HEPCNN.keras_dataGenerator import HEPCNNDataGenerator as DataLoader
trn_dataLoader = DataLoader(args.trndata, args.batch, shuffle=True, nEvent=args.ntrain, syslogger=sysstat)
val_dataLoader = DataLoader(args.valdata, args.batch, shuffle=True, nEvent=args.ntest, syslogger=sysstat)

## Build model
from HEPCNN.keras_model_default import MyModel
model = MyModel(trn_dataLoader.shape[1:])

optm = tf.keras.optimizers.Adam(args.lr)

model.compile(
      optimizer=optm,
      loss='binary_crossentropy',
      metrics=['accuracy']
)
model.summary()
sysstat.update(annotation="modelsetup_done")

if not os.path.exists(weightFile):
    try:
        timeHistory = TimeHistory()
        sysstat.update(annotation="train_start")
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=args.outdir, histogram_freq=1, write_graph=True, write_images=True),
            tf.keras.callbacks.ModelCheckpoint(weightFile, monitor='val_loss', verbose=True, save_best_only=True),
            timeHistory, sysstat,
        ]
        if not args.noEarlyStopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'))
        #history = model.fit(trn_dataLoader.images, trn_dataLoader.labels, sample_weight=trn_dataLoader.weights,
        #                    validation_data = (val_dataLoader.images, val_dataLoader.labels, val_dataLoader.weights),
        #                    epochs=args.epoch, batch_size=args.batch,
        #                    verbose=1,
        #                    shuffle=False,
        #                    #shuffle='batch',
        #                    #shuffle=True,
        #                    callbacks = callbacks)
        history = model.fit_generator(generator=trn_dataLoader,
                                      validation_data = val_dataLoader,
                                      epochs=args.epoch, verbose=1, workers=4,# use_multiprocessing=True,
                                      callbacks = callbacks)
        sysstat.update(annotation="train_end")

        history.history['time'] = timeHistory.times[:]
        with open(historyFile, 'w') as f:
            writer = csv.writer(f)
            keys = history.history.keys()
            writer.writerow(keys)
            for row in zip(*[history.history[key] for key in keys]):
                writer.writerow(row)
        sysstat.update(annotation="wrote_logs")

    except KeyboardInterrupt:
        print("Training finished early")

model.load_weights(weightFile)
pred = model.predict(val_dataLoader.images, verbose=1, batch_size=args.batch)

np.save(predFile, pred)
sysstat.update(annotation="saved_model")
