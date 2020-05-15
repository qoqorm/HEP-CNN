#!/usr/bin/env pythnon
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from os import environ
import concurrent.futures as futures
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
        self.cache_fileIdx = -1

    def __getitem__(self, idx):
        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]

        if self.cache_fileIdx != fileIdx:
            self.cache_fileIdx = fileIdx

            self.cache_images  = self.imagesList[fileIdx]
            self.cache_labels  = self.labelsList[fileIdx]
            self.cache_weights = self.weightsList[fileIdx]
            self.cache_rescale = self.rescaleList[fileIdx]

        idx = idx - offset
        return (self.cache_images[idx], self.cache_labels[idx], self.cache_weights[idx], self.cache_rescale[idx])

    def __len__(self):
        return self.maxEventsList[-1]

    def addSample(self, procName, fileNames, weight=None, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fileNames))
        weightValue = weight ## Rename it just to be clear in the codes

        if procName not in self.procFiles:
            self.procFiles[procName] = [] ## this list keeps image index - later we will use this info to get total event and update weights, etc

        for fileName in glob(fileNames):
            data = h5py.File(fileName, 'r')['all_events']
            suffix = "_val" if 'images_val' in data else ""

            images  = (fileName, 'images'+suffix) ## Keep the filename and image path only, and load them later with multiproc.
            weights = data['weights'+suffix]
            size = weights.shape[0]

            if weightValue is None: weights = data['weights'+suffix]
            else: weights = torch.ones(size)*weightValue

            nEventsInFile = len(weights)
            self.maxEventsList.append(self.maxEventsList[-1]+nEventsInFile)

            labels  = torch.zeros(size) ## Put dummy labels, to set later by calling setProcessLabel()
            weights = torch.Tensor(weights[()])
            ## We will do this step for images later

            fileIdx = len(self.imagesList)
            self.procFiles[procName].append(fileIdx)
            self.imagesList.append(images)
            self.labelsList.append(labels)
            self.weightsList.append(weights)
            self.rescaleList.append(torch.ones(size))

    def setProcessLabel(self, procName, label):
        for i in self.procFiles[procName]:
            size = self.labelsList[i].shape[0]
            self.labelsList[i] = torch.ones(size)*label
            self.procLabels[procName] = label

    def imageToTensor(self, fileIdx):
        fileName, imagesName = self.imagesList[fileIdx]
        data = h5py.File(fileName, 'r')['all_events']
        images = data[imagesName]
        t = torch.Tensor(images[()])
        data = None
        return fileIdx, t

    def initialize(self, nWorkers=1, logger=None):
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

        if logger: logger.update(annotation='Convert images to Tensor')

        env_kmp = environ['KMP_AFFINITY'] if 'KMP_AFFINITY' in environ else None
        environ['KMP_AFFINITY'] = 'none'
        jobs = []
        with futures.ProcessPoolExecutor(max_workers=nWorkers) as pool:
            for fileIdx in range(len(self.maxEventsList)-1):
                job = pool.submit(self.imageToTensor, fileIdx)
                jobs.append(job)

            for job in futures.as_completed(jobs):
                fileIdx, images = job.result()
                self.imagesList[fileIdx] = images
        if env_kmp != None: environ['KMP_AFFINITY'] = env_kmp

        for fileIdx in range(len(self.maxEventsList)-1):
            #images  = torch.Tensor(self.imagesList[fileIdx][()])
            images = self.imagesList[fileIdx]
            self.shape = images.shape

            if self.shape[-1] <= 5:
                ## actual format was NHWC. convert to pytorch native format, NCHW
                images = images.permute(0,3,1,2)
                self.shape = images.shape
                if logger: logger.update(annotation="Convert image format")

            self.imagesList[fileIdx] = images
            self.channel, self.height, self.width = self.shape[1:]

