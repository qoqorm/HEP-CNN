import h5py
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm



# -- parser == #
def get_parser():
    parser = argparse.ArgumentParser(
        description='Run SUSY RPV training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--train-events', action='store', type=int,
                        default=412416, help='Number of events to train on.')
    parser.add_argument('--test-events', action='store', type=int,
                        default=137471, help='Number of events to test on.')
    parser.add_argument('--batch-size', action='store', type=int, default=256,
                        help='batch size per update')
    parser.add_argument('--lr', action='store', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--ls', action='store', type=bool, default=False,
                        help='log scaling')
    parser.add_argument('--patch-size', action='store', type=int, default=0,
                        help='circular padding size')
    parser.add_argument('--base-value', action='store', type=float,
                        default=1e-6, help='non-zero base value for log')
    parser.add_argument('train_data', action='store', type=str,
                        help='path to HDF5 file to train on')
    parser.add_argument('val_data', action='store', type=str,
                        help='path to HDF5 file to validate on')
    parser.add_argument('model', action='store', type=str,
                        help='one of: "CNN", "3ch-CNN"')
    return parser

if torch.cuda.is_available(): print("\nGPU Acceleration Available")
else: print("\nGPU Acceleration Unavailable")

parser = get_parser()
args = parser.parse_args()



# -- load data -- #
class H5Dataset(Dataset):
    def __init__(self, filePath, evtnum):
        super(H5Dataset, self).__init__()
        h5File = h5py.File(filePath)
        evtnum = evtnum
        im = h5File['all_events']['hist'][:evtnum]
        im = np.expand_dims(im, 1)
        if args.model == '3ch-CNN':
            layer_em = h5File['all_events']['histEM'][:evtnum]
            layer_em = np.expand_dims(layer_em, 1)
            layer_track = h5File['all_events']['histtrack'][:evtnum]
            layer_track = np.expand_dims(layer_track, 1)
            layer_em = layer_em / layer_em.max()
            layter_track = layer_track / layer_track.max()
            im = np.concatenate(
                (np.concatenate((im, layer_em), axis=1),
                 layer_track), axis=1
            )

        if args.patch_size:
            imp = np.empty((im.shape[0],                    # N
                            im.shape[1],                    # C
                            im.shape[2],                    # H
                            im.shape[3] + args.patch_size)) # W
            for idx in range(im.shape[0]):
                patch = im[idx][:, :, :args.patch_size]
                img = np.concatenate((im[idx], patch), axis=2)
                imp[idx] = img
            im = imp
        if args.ls:
            im = np.log10(im + args.base_value) - np.log10(args.base_value)
        self.data = im
        lb = h5File['all_events']['y'][:args.train_events]
        self.label = torch.Tensor(lb)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]), self.label[idx])
    def __len__(self):
        return self.data.shape[0]
if torch.cuda.is_available: kwargs = {'num_workers': 1, 'pin_memory': True}

trainDataset = H5Dataset(args.train_data, args.train_events)
trainLoader = DataLoader(trainDataset, batch_size=args.batch_size)

valDataset = H5Dataset(args.val_data, args.test_events)
valLoader = DataLoader(valDataset, batch_size=args.batch_size)



# -- image visualization for data check -- #
h, w = 3, 7
cmap0 = 'viridis'
cmap1 = 'viridis'
chs = 3 if args.model == '3ch-CNN' else 1
for ch in range(chs):
    fig, axes = plt.subplots(nrows=h, ncols=w, figsize=(20, 6))
    for ax in axes.flat:
        ax.set_axis_off()
        idx = random.randint(0, len(trainDataset) + 1)
        label = int(trainDataset[idx][1])
        im = ax.imshow(trainDataset[idx][0][ch, :, :],
                       cmap=cmap1 if label else cmap0)
    cb = fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()



# -- model and train -- #
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.widthFinal = 8 if args.patch_size else 5
        self.channels = 3 if args.model == '3ch-CNN' else 1
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, kernel_size=(2, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            )
        self.fc = nn.Sequential(
            nn.Linear(5*self.widthFinal*256, 512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
            )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 5*self.widthFinal*256)
        x = self.fc(x)
        return x

model = CNN()
if torch.cuda.is_available(): model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCELoss()

def train(lossList):
    model.train()
    for batch_idx, (data, label) in enumerate(trainLoader):
        if torch.cuda.is_available():
            data, label = data.float().cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        #print('Train Epoch: {}\t[{}/{}\t({:.2f}%)]\tLoss: {:.6f}'.format(
        #    epoch, batch_idx * len(data), len(trainDataset),
        #    100. * batch_idx / len(trainLoader), loss.item()
        #))
    lossList.append(loss.item())

def evaluation(accList):
    model.eval()
    acc = 0
    for batch_idx, (data, label) in enumerate(valLoader):
        if torch.cuda.is_available():
            data, label = data.float().cuda(), label.cuda()
        prediction = model(data.float().to('cuda')).cpu().detach().numpy()
        acc += accuracy_score(label.cpu().detach().numpy(), np.where(prediction > 0.5, 1, 0))
    acc /= (batch_idx + 1)
    print(acc)
    accList.append(acc)



# -- train -- #
from sklearn.metrics import roc_curve, accuracy_score, auc
lossList, accList = [], []
for epoch in tqdm(range(1, args.epochs + 1)):
    train(lossList)
    evaluation(accList)
from matplotlib.ticker import MultipleLocator, PercentFormatter

# -- train result -- #
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(121)
ax1.set_title('Loss', fontsize=14)
ax1.plot(lossList, 'k-s', ms=4, label='Train')
#ax1.plot(vlosT, 'k:^', ms=4, label='Validation')

ax1.set_xlabel('Epochs')
#ax1.set_ylabel('Loss')

ax1.tick_params(axis='both', which='both', direction='in')
ax1.grid(alpha=0.5)
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_title('Accuracy', fontsize=14)
ax2.plot(accList, 'k-s', ms=4, label='Train')
#ax2.plot(timeT, vaccT, 'k:^', ms=4, label='Validation')

ax2.set_xlabel('Epochs')
#ax2.set_ylabel('Accuracy')
ax2.set_ylim(0.553, 1.003)
ax2.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1,
                                               decimals=0,
                                               symbol='%',
                                               is_latex=True))

ax2.tick_params(axis='both', which='both', direction='in')
ax2.grid(alpha=0.5)
ax2.legend(loc=4)

plt.tight_layout()
#plt.suptitle('Total', fontsize=20, va='bottom')
plt.show()

print(lossList)
print(accList)
print(' ')



# -- evaluate -- #
from sklearn.metrics import roc_curve, accuracy_score, auc

def evaluate():
    model.eval()

    acc = 0
    for idx, (data, label) in enumerate(valLoader):
        y_pred = model(data.float().to('cuda')).cpu().detach().numpy()
        acc += accuracy_score(label, np.where(y_pred > 0.5, 1, 0))
        #print("Accuracy:", accuracy_score(label,np.where(y_pred > 0.5, 1, 0)))

    acc /= (idx + 1)
    print(acc)
    fpr, tpr, thr = roc_curve(label.numpy().reshape(-1),
                              y_pred.reshape(-1,))
    roc_auc = auc(fpr, tpr)

    plt.plot(tpr, 1 - fpr, label='AUC = %03f' % roc_auc)
    plt.legend()
    plt.show()

evaluate()
