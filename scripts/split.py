import h5py
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, help='input file name', required=True)
args = parser.parse_args()

srcFileName = args.input
outFileName = args.output

data = h5py.File(srcFileName, 'r')

sub_data = list(data['all_events'].keys())


hist_images = data['all_events']['hist']
histEM_images = data['all_events']['histEM']
histtrack_images = data['all_events']['histtrack']
normwght_images = data['all_events']['normalized_weight']
passSR_images = data['all_events']['passSR']
passSR4J_images = data['all_events']['passSR4J']
passSR5J_images = data['all_events']['passSR5J']
weight_images = data['all_events']['weight']
label = data['all_events']['y']

dataSet = [hist_images, histEM_images, histtrack_images, normwght_images, 
           passSR_images, passSR4J_images, passSR5J_images, weight_images, label]


print("Spliting the file...")

cnt = 1
fileName_Num = 0

data1 = []

for i in dataSet:
  for matrix in i:
    fileName = "train_" + str(fileName_Num) + ".h5"
    groupName = str(i).split('_')[0]


    hf = h5py.File(fileName, "w")
    g1 = hf.create_group('all_events')
    data1.append(matrix)
  
    g1.create_dataset('%s' % groupName, data = data1)
    hf.close()

    if cnt == 1024:
      fileName_Num += 1
      cnt = 0
    cnt = cnt+1
