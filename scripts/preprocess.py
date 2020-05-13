#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import sys

if sys.version_info[0] < 3: sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, help='input file name', required=True)
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
parser.add_argument('-n', '--nevent', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('--format', action='store', choices=('NHWC', 'NCHW'), default='NHWC', help='image format for output (NHWC for TF default, NCHW for pytorch default)')
parser.add_argument('-c', '--chunk', action='store', type=int, default=1024, help='chunk size')
parser.add_argument('--nocompress', dest='nocompress', action='store_true', default=False, help='disable gzip compression')
parser.add_argument('-s', '--split', action='store_true', default=False, help='split output file')
args = parser.parse_args()

srcFileName = args.input
outFileName = args.output

print("Opening input dataset %s..." % srcFileName)
data = h5py.File(srcFileName, 'r')
nEventsTotal = len(data['all_events']['weight'])
if not args.split:
    if args.nevent >= 0: nEventsTotal = args.nevent
    else: args.nevent = nEventsTotal
print("  total event to preprocess=", nEventsTotal)

for i, begin in enumerate(range(0, nEventsTotal, args.nevent)):
    end = min(begin+args.nevent, nEventsTotal)

    if args.split: outFileName = '%s_%d.h5' % (args.output.rsplit('.', 1)[0], i)
    else: outFileName = args.output

    labels = data['all_events']['y'][begin:end]
    weights = data['all_events']['weight'][begin:end]
    image_h = data['all_events']['hist'][begin:end]
    image_e = data['all_events']['histEM'][begin:end]
    image_t = data['all_events']['histtrack'][begin:end]

    #print("Normalizing EM, track histograms...")
    image_e /= image_e.max()
    image_t /= image_t.max()

    if i == 0:
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
    if i == 0:
        print("  Output image format=", args.format)
        print("  Output image shape=", image.shape)

    print("Writing output file %s..." % outFileName)
    nEvent = image.shape[0]
    chunkSize = min(args.chunk, nEvent)
    if outFileName.endswith('.h5'):
        with h5py.File(outFileName, 'w', libver='latest') as outFile:
            g = outFile.create_group('all_events')
            kwargs = {} if args.nocompress else {'compression':'gzip', 'compression_opts':9}
            g.create_dataset('images', data=image, chunks=((chunkSize,)+image.shape[1:]), **kwargs)
            g.create_dataset('labels', data=labels, chunks=(chunkSize,))
            g.create_dataset('weights', data=weights, chunks=(chunkSize,))
            outFile.swmr_mode = True
            print("  done")

        with h5py.File(outFileName, 'r') as outFile:
            print("  created", outFileName)
            print("  keys=", list(outFile.keys()))
            print("  shape=", outFile['all_events']['images'].shape)
    elif outFileName.endswith('npz'):
        args = {
            'images': image,
            'labels': labels,
            'weights': weights,
        }
        np.savez_compressed(outFileName, **args)
        print("  done")
