#!/usr/bin/env python
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', action='store', type=str, help='input file name')
parser.add_argument('-o', action='store', type=str, help='output file name')
parser.add_argument('-n', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('--suffix', action='store', type=str, default='', help='suffix for output ("" for train, "val" for validation set)')
parser.add_argument('--format', action='store', choices=('NHWC', 'NCHW'), default='NHWC', help='image format for output (NHWC for TF default, HCHW for pytorch default)')
parser.add_argument('--circpad', action='store', type=int, default=0, help='padding size for the circular padding (0 for no-padding)')
parser.add_argument('--logscale', action='store', type=bool, default=False, help='apply log scaling to images')
args = parser.parse_args()

srcFileName = args.i
outFileName = args.o
if args.suffix != '': args.suffix = '_'+args.suffix

print("Opening input dataset %s..." % srcFileName)
data = h5py.File(srcFileName, 'r')

print("Loading data...")
if args.n == -1:
    labels = data['all_events']['y']
    print("  total event to preprocess=", labels.shape[0])
    weights = data['all_events']['weight']
    image_h = data['all_events']['hist'][()]
    image_e = data['all_events']['histEM'][()]
    image_t = data['all_events']['histtrack'][()]
else:
    labels = data['all_events']['y'][:args.n]
    print("  total event to preprocess=", labels.shape[0])
    weights = data['all_events']['weight'][:args.n]
    image_h = data['all_events']['hist'][:args.n]
    image_e = data['all_events']['histEM'][:args.n]
    image_t = data['all_events']['histtrack'][:args.n]

print("Normalizing EM, track histograms...")
image_e /= image_e.max()
image_t /= image_t.max()

if args.logscale:
    print("Apply log scaling to images...")
    print("!!!NOT IMPLEMENTED YET!!!!")

if args.circpad > 0:
    print("Apply circular padding, size=", args.circpad, "...")
    print("  Note for the next step: circular padding size depend on the CNN structure (layers, kernel, maxpool...).")
    print("  Please double check with your model.")

    print("    input image shape=", image_h.shape)
    image_h = np.concatenate([image_h, image_h[:,:,:args.circpad]], axis=-1)
    image_e = np.concatenate([image_e, image_e[:,:,:args.circpad]], axis=-1)
    image_t = np.concatenate([image_t, image_t[:,:,:args.circpad]], axis=-1)
    print("    output image shape=", image_h.shape)

print("Joining channels...")
print("  Input image shape=", image_h.shape, image_e.shape, image_t.shape)
if args.format == 'NHWC':
    image_h = np.expand_dims(image_h, -1)
    image_e = np.expand_dims(image_e, -1)
    image_t = np.expand_dims(image_t, -1)
    image = np.concatenate([image_h, image_e, image_t], axis=-1)
else:
    image_h = np.expand_dims(image_h, 1)
    image_e = np.expand_dims(image_e, 1)
    image_t = np.expand_dims(image_t, 1)
    image = np.concatenate([image_h, image_e, image_t], axis=1)
print("  Output image format=", args.format)
print("  Output image shape=", image.shape)

print("Writing output file %s..." % outFileName)
with h5py.File(outFileName, 'w') as outFile:
    g = outFile.create_group('all_events')
    g.create_dataset('images'+args.suffix, data=image, 
                     chunks=True, compression='gzip', compression_opts=9)
    g.create_dataset('labels'+args.suffix, data=labels, chunks=True)
    g.create_dataset('weights'+args.suffix, data=weights, chunks=True)

with h5py.File(outFileName, 'r') as outFile:
    print("  created", outFileName)
    print("  keys=", list(outFile.keys()))
    print("  shape=", outFile['all_events']['images'].shape)
