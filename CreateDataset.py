import numpy as np
import pickle
import random
import pandas as pd
import shutil
import os
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
from Preprocess import loadData
import argparse

parser = argparse.ArgumentParser(description="choose Dataset")
parser.add_argument('-d', dest='Dataset', type=str, default='PA', help='choose dataset', choices=['IP','PA','SA','PU','BOT', 'KSC'])
parser.add_argument('-r', dest='ratio', type=float, default=0.1, help='choose ratio, IP:0.1,   KSC:0.1/0.2,    PU:0.025,   PA:0.005   SA:0.02   BOT:0.2')

# args = parser.parse_args()
args = parser.parse_known_args()[0]

random.seed(6666)

class Create_Dataset(object):
    def __init__(self, path, ratio=0.005):
        super(Create_Dataset, self).__init__()
        self.path = path
        self.ratio = ratio

    def big_dataset(self):
        num_label = list(np.unique(input_label))
        del num_label[-1]

        train_data = pd.read_csv(self.path)
        label_num = train_data.label.value_counts()

        if not os.path.exists('./trainset/'):
            os.makedirs('./trainset/')

        for i in num_label:

            if label_num[i] > 400:
                train_equ_i = train_data.file_names[train_data.label == i]

                label_i_list = random.sample(list(train_equ_i), int(len(list(train_equ_i)) * self.ratio))

                for file_name in label_i_list:
                    org_path = './pickle/' + file_name
                    shutil.move(org_path, './trainset/')
                    idx = train_data.loc[train_data['file_names'] == file_name].index.tolist()[0]
                    train_data = train_data.drop(index=idx, axis=0)

            else:
                train_equ_i = train_data.file_names[train_data.label == i]
                label_i_list = random.sample(list(train_equ_i), int(np.floor(label_num[i] * 0.2)))

                for file_name in label_i_list:
                    org_path = './pickle/' + file_name
                    shutil.move(org_path, './trainset/')
                    idx = train_data.loc[train_data['file_names'] == file_name].index.tolist()[0]
                    train_data = train_data.drop(index=idx, axis=0)
        train_data.to_csv('./test.csv')

    def small_dataset(self):
        num_label = list(np.unique(input_label))
        del num_label[-1]

        train_data = pd.read_csv(self.path)
        label_num = train_data.label.value_counts()

        if not os.path.exists('./trainset2/'):
            os.makedirs('./trainset2/')

        for i in num_label:

            if label_num[i] > 100:
                train_equ_i = train_data.file_names[train_data.label == i]

                label_i_list = random.sample(list(train_equ_i), int(len(list(train_equ_i)) * self.ratio))

                for file_name in label_i_list:
                    org_path = './pickle/' + file_name
                    shutil.move(org_path, './trainset2/')
                    idx = train_data.loc[train_data['file_names'] == file_name].index.tolist()[0]
                    train_data = train_data.drop(index=idx, axis=0)



            else:
                train_equ_i = train_data.file_names[train_data.label == i]
                label_i_list = random.sample(list(train_equ_i), 5)

                for file_name in label_i_list:
                    org_path = './pickle/' + file_name
                    shutil.move(org_path, './trainset2/')
                    idx = train_data.loc[train_data['file_names'] == file_name].index.tolist()[0]
                    train_data = train_data.drop(index=idx, axis=0)

        train_data.to_csv('./test.csv')

    def val_dataset(self):
        num_label = list(np.unique(input_label))
        del num_label[-1]

        train_data = pd.read_csv('./test.csv')
        label_num = train_data.label.value_counts()

        if not os.path.exists('./valset/'):
            os.makedirs('./valset/')

        for i in num_label:

            if label_num[i] > 100:
                train_equ_i = train_data.file_names[train_data.label == i]

                label_i_list = random.sample(list(train_equ_i), int(len(list(train_equ_i)) * self.ratio))

                for file_name in label_i_list:
                    org_path = './pickle/' + file_name
                    shutil.move(org_path, './valset/')
                    idx = train_data.loc[train_data['file_names'] == file_name].index.tolist()[0]
                    train_data = train_data.drop(index=idx, axis=0)



            else:
                train_equ_i = train_data.file_names[train_data.label == i]
                label_i_list = random.sample(list(train_equ_i), 5)

                for file_name in label_i_list:
                    org_path = './pickle/' + file_name
                    shutil.move(org_path, './valset/')
                    idx = train_data.loc[train_data['file_names'] == file_name].index.tolist()[0]
                    train_data = train_data.drop(index=idx, axis=0)

        train_data.to_csv('./test.csv')


def get_img(path):
    patch = open(path, 'rb')
    data = pickle.load(patch)
    data = data.transpose(2, 0, 1)
    data = torch.tensor(data, dtype=torch.float32)

    return data


class HSIDataset(Dataset):
    def __init__(
            self, df, data_root, transform=None
    ):
        super().__init__()
        self.df = df
        self.transforms = transform
        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        self.data_root = data_root

    def __len__(self):
        return len(os.listdir(self.data_root))

    def __getitem__(self, index: int):
        # get labels
        file_list = os.listdir(self.data_root)
        path = "{}/{}".format(self.data_root, file_list[index])

        img = get_img(path)
        img = self.transforms(img)

        loc_label = list(self.df['file_names']).index(file_list[index])
        label = self.df.loc[loc_label, 'label']

        return img, label
    
class HSITestDataset(Dataset):
    def __init__(self, df, data_root):
        super().__init__()
        self.df = df
        self.data_root = data_root

    def __len__(self):
        return len(os.listdir(self.data_root))

    def __getitem__(self, index: int):
        # get labels
        file_list = os.listdir(self.data_root)
        path = "{}/{}".format(self.data_root, file_list[index])

        img = get_img(path)

        loc_label = list(self.df['file_names']).index(file_list[index])
        label = self.df.loc[loc_label, 'label']

        return img, label

if __name__ == '__main__':
    input_data, input_label = loadData(args.Dataset)
    # IP:0.1,   KSC:0.1/0.2,    PU:0.025,   PA:0.005   SA:0.02
    create_dataset = Create_Dataset(path='./all_data.csv', ratio=args.ratio)   
    create_dataset.big_dataset()
    #create_dataset.small_dataset()
    create_dataset.val_dataset()
