#!/usr/bin/env pythnon
from torch.utils.data import Dataset, DataLoader

class HEPCNNDataset(Dataset):
    def __init__(self, fileName, nEvent):
        super(H5Dataset, self).__init__()
        print("Opening", fileName, "nEvent=", nEvent)

        if fileName.endswith('h5'):
            self.data = h5py.File(fileName, 'r')
        elif fileName.endswith('npz'):
            self.data = {'all_events':np.load(fileName)}

        suffix = "_val" if 'images_val' in self.data['all_events'] else ""
        if nEvent < 0:
            self.images  = self.data['all_events']['images'+suffix][()]
            self.labels  = self.data['all_events']['labels'+suffix][()]
            self.weights = self.data['all_events']['weights'+suffix][()]
        else:
            self.images  = self.images[:nEvent]
            self.labels  = self.labels[:nEvent]
            self.weights = self.weights[:nEvent]

        print("Convert data to Tensors")
        self.images = torch.Tensor(self.images)
        self.labels = torch.Tensor(self.labels)
        self.weights = torch.Tensor(self.weights)
        self.shape = self.images.shape

        width, height, channel = self.shape[1:]
        self.data_format = 'NHWC'
        if channel > 5:
            channel, width, height = width, height, channel
            self.data_format = 'NCHW'
            self.images = np.transpose(self.images[idx], (2,1,0))
        self.channel, self.width, self.height = channel, width, height

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx], self.weights[idx])

    def __len__(self):
        return self.shape[0]

