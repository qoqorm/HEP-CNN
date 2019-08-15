#!/usr/bin/env python
import pandas as pd
import h5py
import numpy as np
import argparse
import sys, os

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', action='store', type=int, default=50, help='Number of epochs')
parser.add_argument('--ntrain', action='store', type=int, default=-1, help='Number of events for training')
parser.add_argument('--ntest', action='store', type=int, default=-1, help='Number of events for test/validation')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-t', '--trndata', action='store', type=str, required=True, help='input file for training')
parser.add_argument('-v', '--valdata', action='store', type=str, required=True, help='input file for validation')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')

args = parser.parse_args()

trn_data = h5py.File(args.trndata, 'r')
trn_images = trn_data['all_events']['images']#[()]
trn_labels = trn_data['all_events']['labels']#[()]
trn_weights = trn_data['all_events']['weights']#[()]

val_data = h5py.File(args.valdata, 'r')
if 'images_val' in val_data['all_events']:
    val_images = val_data['all_events']['images_val']#[()]
    val_labels = val_data['all_events']['labels_val']#[()]
    val_weights = val_data['all_events']['weights_val']#[()]
else:
    val_images = val_data['all_events']['images']#[()]
    val_labels = val_data['all_events']['labels']#[()]
    val_weights = val_data['all_events']['weights']#[()]

if args.ntrain > 0:
    trn_images = trn_images[:args.ntrain]
    trn_labels = trn_labels[:args.ntrain]
    trn_weights = trn_weights[:args.ntrain]

if args.ntest > 0:
    val_images = val_images[:args.ntest]
    val_labels = val_labels[:args.ntest]
    val_weights = val_weights[:args.ntest]

shape = trn_images.shape

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
weightFile = os.path.join(args.outdir, 'weight.h5')
predFile = os.path.join(args.outdir, 'predict.npy')
historyFile = os.path.join(args.outdir, 'history.csv')

#from keras.utils.io_utils import HD5Matrix ## available from TF2.X
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

## Build model
model = tf.keras.Sequential([
    Conv2D(64, input_shape=shape[1:], kernel_size=(3,3), activation='relu', strides=1, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),

    Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same'),
    BatchNormalization(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy']
)
model.summary()

import time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.times = []

    def on_epoch_begin(self, batch, logs):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs):
        self.times.append(time.time() - self.epoch_time_start)

if not os.path.exists(weightFile):
    try:
        timeHistory = TimeHistory()
        history = model.fit(trn_images, trn_labels, sample_weight=trn_weights,
                            validation_data = (val_images, val_labels, val_weights),
                            epochs=args.epoch, batch_size=args.batch,
                            verbose=1,
                            #shuffle='batch',
                            shuffle=False,
                            callbacks = [
                                tf.keras.callbacks.TensorBoard(log_dir=args.outdir, histogram_freq=1, write_graph=True, write_images=True),
                                tf.keras.callbacks.ModelCheckpoint(weightFile, monitor='val_loss', verbose=True, save_best_only=True),
                                tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                                timeHistory,
                            ])

        history.history['time'] = timeHistory.times[:]
        df = pd.DataFrame(history.history)
        df.index.name = "epoch"
        df.to_csv(historyFile)

    except KeyboardInterrupt:
        print("Training finished early")

model.load_weights(weightFile)
pred = model.predict(val_images, verbose=1, batch_size=args.batch)

np.save(predFile, pred)
