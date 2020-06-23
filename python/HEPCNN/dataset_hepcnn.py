#!/usr/bin/env pythnon
import h5py
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from glob import glob

class HEPCNNDataset(Dataset):
    def __init__(self, **kwargs):
        super(HEPCNNDataset, self).__init__()
        self.procFiles = {}
        self.procLabels = {}

        self.maxEventsList = [0,]
        self.imagesList = []
        self.labelsList = []
        self.weightsList = []
        self.rescaleList = []

    def __getitem__(self, idx):
        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = idx - offset

        image  = self.imagesList[fileIdx][idx]
        label  = self.labelsList[fileIdx][idx]
        weight = self.weightsList[fileIdx][idx]
        rescale = self.rescaleList[fileIdx][idx]

        return (image, label, weight, rescale)

    def __len__(self):
        return self.maxEventsList[-1]

    def addSample(self, procName, fileNames, weight=None, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fileNames))
        weightValue = weight ## Rename it just to be clear in the codes

        if procName not in self.procFiles:
            self.procFiles[procName] = [] ## this list keeps image index - later we will use this info to get total event and update weights, etc

        for fileName in glob(fileNames):
            data = h5py.File(fileName, 'r', libver='latest', swmr=True)['all_events']
            suffix = "_val" if 'images_val' in data else ""

            images  = data['images'+suffix]
            size = images.shape[0]

            if weightValue is None: weights = data['weights'+suffix]
            else: weights = torch.ones(size, dtype=torch.float32, requires_grad=False)*weightValue

            nEventsInFile = len(weights)
            self.maxEventsList.append(self.maxEventsList[-1]+nEventsInFile)

            labels  = torch.zeros(size, dtype=torch.float32, requires_grad=False) ## Put dummy labels, to set later by calling setProcessLabel()
            ## We will do this step for images later

            fileIdx = len(self.imagesList)
            self.procFiles[procName].append(fileIdx)
            self.imagesList.append(images)
            self.labelsList.append(labels)
            self.weightsList.append(weights)
            self.rescaleList.append(torch.ones(size, dtype=torch.float32, requires_grad=False))

    def setProcessLabel(self, procName, label):
        for i in self.procFiles[procName]:
            size = self.labelsList[i].shape[0]
            self.labelsList[i] = torch.ones(size, dtype=torch.float32, requires_grad=False)*label
            self.procLabels[procName] = label

    def initialize(self, logger=None):
        if logger: logger.update(annotation='Reweights by category imbalance')
        ## Compute sum of weights for each label categories
        sumWByLabel = {}
        sumEByLabel = {}
        for procName, fileIdxs in self.procFiles.items():
            label = self.procLabels[procName]
            if label not in sumWByLabel: sumWByLabel[label] = 0.
            if label not in sumEByLabel: sumEByLabel[label] = 0.
            procSumW = sum([sum(self.weightsList[i]) for i in fileIdxs])
            procSumE = sum([len(self.weightsList[i]) for i in fileIdxs])
            print("@@@ Process=%s nEvent=%d sumW=%.3f events/fb-1" % (procName, procSumE, procSumW.item()))
            sumWByLabel[label] += procSumW
            sumEByLabel[label] += procSumE

        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for procName, fileIdxs in self.procFiles.items():
            label = self.procLabels[procName]
            for i in fileIdxs: self.rescaleList[i] *= sumEByLabel[label]/sumWByLabel[label]

        ## Find overall rescale for the data imbalancing problem - fit to the category with maximum entries
        maxSumELabel = max(sumEByLabel, key=lambda key: sumEByLabel[key])
        for procName, fileIdxs in self.procFiles.items():
            label = self.procLabels[procName]
            if label == maxSumELabel: continue
            sf = sumEByLabel[maxSumELabel]/sumEByLabel[label]
            print("@@@ Scale up the sample", label, "->", maxSumELabel, sf)
            for i in fileIdxs: self.rescaleList[i] *= sf

        self.shape = self.imagesList[0].shape[1:]
        if self.shape[-1] <= 5: ## actual format was NHWC
            self.format = 'NHWC'
            self.height, self.width, self.channel = self.shape
        else:
            self.format = 'NCHW'
            self.channel, self.height, self.width= self.shape

