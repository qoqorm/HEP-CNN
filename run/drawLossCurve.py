#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

#plt.rcParams['figure.figsize'] = (7, len(dirs)*1.2)
plt.rcParams['figure.figsize'] = (7, 2*2)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5

for d in sys.argv[1:]:
    df = pd.read_csv(d)

    ax = plt.subplot(2, 1, 1)
    plt.plot(df['loss'], '.-', label='train')
    plt.plot(df['val_loss'], '.-', label='val')
    plt.ylabel('Loss')

    plt.yscale('log')

    ax = plt.subplot(2, 1, 2)
    plt.plot(df['acc'], '.-', label='train')
    plt.plot(df['val_acc'], '.-', label='val')
    ax.set_ylim([0,1])
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')

    plt.tight_layout()
    plt.show()
