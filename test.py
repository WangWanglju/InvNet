import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
from CreateDataset import get_img,HSITestDataset
from CreateDataset import loadData
from tqdm import tqdm
# from LiteDepthwiseNet import LiteDepthwiseNet
# from LiteDepthwiseNet import reSSRN
from involution1 import reSSRN
# from SSRN import SSRN
import argparse

parser = argparse.ArgumentParser(description="regulate parameter")

parser.add_argument('-d', dest='Dataset', type=str, default='PA', help='choose dataset', choices=['IP', 'PA','SA','PU','BOT', 'KSC'])
parser.add_argument('-c', dest='Model', type=str, default='reSSRN', help='choose Model')

# args = parser.parse_args()
args = parser.parse_known_args()[0]

file_name = args.Dataset
input_data, input_label = loadData(file_name)
num_labels = len(np.unique(input_label))-1
# 测试
# model =  InvNet(103,9)
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

      
    
PATH = f'./weights/{file_name}_best_model.pickle'
model.load_state_dict(torch.load(PATH))

test = pd.read_csv('./test.csv')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_data = HSITestDataset(test,'./pickle/')
test_loader = DataLoader(test_data, shuffle=False,num_workers=16,
                         batch_size=128)


per_correct = [0 for i in range(num_labels)]
per_total = [0 for i in range(num_labels)]

def test():

    model.eval()
    for data in tqdm(test_loader):
        
        inputs, labels = data
#         inputs = inputs[:,None,:,:,:]
#         inputs = inputs.permute(0,1,3,4,2)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            y_pre = model(inputs)
        _, predict = torch.max(y_pre.data, dim=1)
        res = predict == labels
        for label_idx in range(len(labels)):
            label_single = labels[label_idx].item()
            per_correct[label_single] += res[label_idx].item()
            per_total[label_single] += 1
        
        
    for acc_idx in range(len(per_correct)):
        acc = per_correct[acc_idx]/per_total[acc_idx]
        print('\tclassID:%d\tacc:%f\t'%(acc_idx+1, acc))
        save_path = './record/' + file_name + '_' + args.Model 
        with open(save_path, 'a') as f:
#             f.write(PATH)
            f.write('\tclassID:%d\tacc:%f\t'%(acc_idx+1, acc))
        
    with open(save_path, 'a') as f:
        f.write('total_Accuracy: %f'%(sum(per_correct)/sum(per_total)))    
    print('total_Accuracy: %f'%(sum(per_correct)/sum(per_total)))

if __name__ == '__main__':
    test()
