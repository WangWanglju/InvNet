#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam,lr_scheduler,SGD
from transformers import AdamW, get_linear_schedule_with_warmup

from Preprocess import loadData,FocalLoss
from CreateDataset import HSIDataset
# from SSRN import SSRN
# from LiteDepthwiseNet import LiteDepthwiseNet
# from LiteDepthwiseNet import  reSSRN
from involution1 import reSSRN
# from involution_pool import reSSRN
import argparse

parser = argparse.ArgumentParser(description="regulate parameter")

parser.add_argument('-e', dest='epoch', type=int, default=500, help='# of epoch')
parser.add_argument('-s', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('-l', dest='lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('-d', dest='Dataset', type=str, default='PA', help='choose dataset', choices=['IP', 'PA','SA','PU','BOT', 'KSC'])
parser.add_argument('-c', dest='Model', type=str, default='reSSRN', help='choose Model')

# args = parser.parse_args()
args = parser.parse_known_args()[0]


CFG = {
    'file_name':args.Dataset,
    'seed': 1006,
    'model': args.Model,
    'epochs': args.epoch,
    'bs': args.batch_size,
    'lr': args.lr,
    'weight_decay':1e-6,
    'num_workers': 8,
    }

def seed_torch(seed=1006):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(CFG['seed'])

file_name = CFG['file_name']

input_data, input_label = loadData(file_name)
print(input_data.shape)


# def normalize(image):

#     mean = np.mean(image)
#     var = np.mean(np.square(image-mean))

#     image = (image - mean)/np.sqrt(var)

#     return image

# input_data = normalize(input_data)


#查看类别数量
# print(np.unique(input_label))
num_labels = len(np.unique(input_label))-1
# print(num_labels)

# num = os.listdir('./pickle/')
# print('total_num', len(num))

#检查训练集种类分布
find_label = pd.read_csv('./all_data.csv')
# label_num = find_label.label.value_counts()
# print('per', label_num)
#
per_label = []
a = os.listdir('./trainset/')
for i in a:
    y = find_label.loc[find_label['file_names'] == i]['label']
    per_label.append(y.values[0])

weight = []
for i in range(num_labels):
    weight.append(per_label.count(i))




#focal loss alpha

alpha_weight = [1 / i for i in weight]
# print(len(alpha_weight))
#
train = pd.read_csv('./all_data.csv')
train_data = HSIDataset(train, './trainset/')
valid_data = HSIDataset(train, './valset/')


train_loader = DataLoader(train_data,shuffle = True,num_workers=CFG['num_workers'],
                         batch_size=CFG['bs'])
valid_loader = DataLoader(valid_data,shuffle = False,num_workers=CFG['num_workers'],
                         batch_size=CFG['bs'])


# model =  InvNet(103,classes=9)
#model = LiteDepthwiseNet(97,16)
if file_name == 'IP':
    model = reSSRN(200,16)
elif file_name == 'PU':
    model = reSSRN(103,9)
elif file_name == 'PA':
    model = reSSRN(102,9)
elif file_name == 'SA':
    model = reSSRN(204,16)
elif file_name == 'KSC':
    model = reSSRN(176,13)
elif file_name == 'BOT':
    model = reSSRN(145,14)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# PATH = './PU_last_model.pickle'
# model.load_state_dict(torch.load(PATH))
criterion = FocalLoss(num_labels, alpha=alpha_weight)
# optimizer = SGD(model.parameters(), lr=0.001,weight_decay=0.001)
optimizer = AdamW(model.parameters(), lr=CFG['lr'], eps=1e-8, weight_decay=CFG['weight_decay'])

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*CFG['epochs']*0.2),
                                            num_training_steps=len(train_loader)*CFG['epochs'])
# optimizer = SGD(model.parameters(), 1e-3)
# optim_agc = AGC(model.parameters(), optimizer) # Needs testing

dataloaders = {
    'train':train_loader,
    'validation':valid_loader
}

RESUME = False

if RESUME:
    path_checkpoint = "./model_parameter/test/ckpt_best_50.pth"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点

    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    lr_scheduler.load_state_dict(checkpoint['lr_schedule'])

def train_model(model, criterion, optimizer, num_epochs=10):
#     model = model.to(device)
    best_acc = 0
    
    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()            
    
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
#                 inputs = inputs[:,None,:,:,:]
#                 inputs = inputs.permute(0,1,3,4,2)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
#                     print(loss)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    PATH = './weights/' + file_name + '_best_model.pickle'
                    torch.save(model.state_dict(), PATH)
                    save_path = './record/' + file_name  + '_' + CFG['model']
                    with open(save_path, 'w') as f:
                        f.write('best train acc:\t' + str(best_acc.item()) +'\n')

            print(epoch,prefix,epoch_loss.item(),epoch_acc.item())
    PATH = './weights/' + file_name + '_last_model.pickle'
    torch.save(model.state_dict(), PATH)
    
        #断点训练
#     if epoch % 20 == 0:
#         print('epoch:',epoch)
#         print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
#         checkpoint = {
#             "net": model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             "epoch": epoch,
#             'lr_schedule': lr_scheduler.state_dict()
#         }
#         if not os.path.isdir("./model_parameter/test"):
#             os.mkdir("./model_parameter/test")
#         torch.save(checkpoint, './model_parameter/test/ckpt_best_%s.pth' % (str(epoch)))
        



train_model(model, criterion, optimizer, num_epochs=CFG['epochs'])
