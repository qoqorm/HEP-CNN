#!/usr/bin/env python
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
trn_images = trn_data['all_events']['images']
trn_labels = trn_data['all_events']['labels']
trn_weights = trn_data['all_events']['weights']

val_data = h5py.File(args.valdata, 'r')
val_images = val_data['all_events']['images']
val_labels = val_data['all_events']['labels']
val_weights = val_data['all_events']['weights']

shape = trn_images.shape

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
weightFile = os.path.join(args.outdir, 'weight.h5')
predFile = os.path.join(args.outdir, 'predict.npy')

#from keras.utils.io_utils import HD5Matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

try:
    model.fit(trn_images, trn_labels, sample_weight=trn_weights,
              validation_data = (val_images, val_labels, val_weights),
              shuffle='batch',
              epochs=args.epoch, batch_size=args.batch,
              verbose=1,
              callbacks = [
                  EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                  ModelCheckpoint(weightFile, monitor='val_loss', verbose=True, save_best_only=True),
              ])
except KeyboardInterrupt:
    print("Training finished early")

model.load_weights(weightFile)
pred = model.predict(val_images, verbose=1, batch_size=args.batch)

np.save(predFile, pred)
