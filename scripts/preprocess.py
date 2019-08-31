#!/usr/bin/env python
import h5py
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, help='input file name', required=True)
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
parser.add_argument('-n', '--nevent', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('--suffix', action='store', type=str, default='', help='suffix for output ("" for train, "val" for validation set)')
parser.add_argument('--format', action='store', choices=('NHWC', 'NCHW'), default='NHWC', help='image format for output (NHWC for TF default, HCHW for pytorch default)')
parser.add_argument('--circpad', action='store', type=int, default=0, help='padding size for the circular padding (0 for no-padding)')
parser.add_argument('--logscale', action='store', type=bool, default=False, help='apply log scaling to images')
args = parser.parse_args()

srcFileName = args.input
outFileName = args.output
if args.suffix != '': args.suffix = '_'+args.suffix

print("Opening input dataset %s..." % srcFileName)
data = h5py.File(srcFileName, 'r')

print("Loading data...")
if args.nevent == -1:
    labels = data['all_events']['y'][()]
    print("  total event to preprocess=", labels.shape[0])
    weights = data['all_events']['weight'][()]
    image_h = data['all_events']['hist'][()]
    image_e = data['all_events']['histEM'][()]
    image_t = data['all_events']['histtrack'][()]
else:
    labels = data['all_events']['y'][:args.nevent]
    print("  total event to preprocess=", labels.shape[0])
    weights = data['all_events']['weight'][:args.nevent]
    image_h = data['all_events']['hist'][:args.nevent]
    image_e = data['all_events']['histEM'][:args.nevent]
    image_t = data['all_events']['histtrack'][:args.nevent]

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
nEvent = image.shape[0]
chunkSize = min(1024, nEvent)
if outFileName.endswith('.h5'):
    with h5py.File(outFileName, 'w') as outFile:
        g = outFile.create_group('all_events')
        g.create_dataset('images'+args.suffix, data=image,
                         chunks=((chunkSize,)+image.shape[1:]), compression='gzip', compression_opts=9)
        g.create_dataset('labels'+args.suffix, data=labels, chunks=(chunkSize,))
        g.create_dataset('weights'+args.suffix, data=weights, chunks=(chunkSize,))
        print("  done")

    with h5py.File(outFileName, 'r') as outFile:
        print("  created", outFileName)
        print("  keys=", list(outFile.keys()))
        print("  shape=", outFile['all_events']['images'+args.suffix].shape)
elif outFileName.endswith('npz'):
    args = {
        'images'+args.suffix: image,
        'labels'+args.suffix: labels,
        'weights'+args.suffix: weights,
    }
    np.savez_compressed(outFileName, **args)
    print("  done")
elif outFileName.endswith('.tfrecords'):
    import tensorflow as tf
    options = tf.python_io.TFRecordOptions(
        compression_method=tf.python_io.TFRecordCompressionType.GZIP,
        compression_level=9,
    )
    with tf.python_io.TFRecordWriter(outFileName, options=options) as writer:
        nsplit = int(np.ceil(1.0*nEvent/chunkSize))
        ximage = np.array_split(image, nsplit)
        xlabels = np.array_split(labels, nsplit)
        xweights = np.array_split(weights, nsplit)
        for i in range(nsplit):
            print("  Write chunk (%d/%d)" % (i, nsplit), end='\r')
            sys.stdout.flush()
            ex = tf.train.Example(features=tf.train.Features(feature={
                'all_events/images'+args.suffix: tf.train.Feature(bytes_list=tf.train.BytesList(value=[ximage[i].tobytes()])),
                #'all_events/images'+args.suffix: tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(ximage[i].tostring())])),
                #'all_events/images'+args.suffix: tf.train.Feature(float_list=tf.train.FloatList(value=ximage[i].astype(np.float32).reshape(-1))),
                'all_events/labels'+args.suffix: tf.train.Feature(int64_list=tf.train.Int64List(value=xlabels[i].astype(np.int32))),
                'all_events/weights'+args.suffix: tf.train.Feature(float_list=tf.train.FloatList(value=xweights[i].astype(np.float32))),
            }))
            writer.write(ex.SerializeToString())
        print("\n  done")
