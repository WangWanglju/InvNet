import numpy as np
import pickle
import pandas as pd
import spectral
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import torch.nn as nn
import torch
import argparse

parser = argparse.ArgumentParser(description="choose Dataset")
parser.add_argument('-d', dest='Dataset', type=str, default='PA', help='choose dataset', choices=['IP','PA','SA','PU','BOT', 'KSC'])

# args = parser.parse_args()
args = parser.parse_known_args()[0]

def loadData(name):
    data_path = 'dataset/HSI/'
    if name == 'IP':
        data = loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'PA':
        data = loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        labels = loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
    elif name == 'BOT':
        data = loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
    elif name == 'KSC':
        data = loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']

    return data, labels


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    label = []
    file_names = []
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] != 0:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesLabels = y[r - margin, c - margin] - 1

                file_name = str(r) + '_' + str(c) + '.pickle'
                label.append(patchesLabels)
                file_names.append(file_name)
                index = path + file_name
                with open(index, 'wb') as f:
                    pickle.dump(patch, f)

    write_tocsv = {'file_names': file_names, 'label': label}
    data = pd.DataFrame(write_tocsv)
    data.to_csv('./all_data.csv')
    print("finish to write data!")

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


if __name__ == '__main__':
    path = "./pickle/"
    if not os.path.exists(path):
        os.makedirs(path)

    input_data, input_label = loadData(args.Dataset)
    createImageCubes(input_data, input_label, windowSize=9)
