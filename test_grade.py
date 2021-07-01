import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import spectral
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from CreateDataset import get_img, HSITestDataset
from CreateDataset import loadData
from tqdm import tqdm
from involution1 import reSSRN
# from LiteDepthwiseNet import reSSRN
# from SSRN import SSRN
import argparse

parser = argparse.ArgumentParser(description="regulate parameter")

parser.add_argument('-d', dest='Dataset', type=str, default='IP', help='choose dataset',
                    choices=['IP', 'PA', 'SA', 'PU', 'BOT', 'KSC'])
parser.add_argument('-c', dest='Model', type=str, default='reSSRN', help='choose Model')

# args = parser.parse_args()
args = parser.parse_known_args()[0]

file_name = args.Dataset
input_data, input_label = loadData(file_name)
num_labels = len(np.unique(input_label)) - 1
# 测试
# model =  InvNet(103,9)
# model = LiteDepthwiseNet(97,16)
if file_name == 'IP':
    model = reSSRN(200, 16)
elif file_name == 'PU':
    model = reSSRN(103, 9)
elif file_name == 'PA':
    model = reSSRN(102, 9)
elif file_name == 'SA':
    model = reSSRN(204, 16)
elif file_name == 'KSC':
    model = reSSRN(176, 13)
elif file_name == 'BOT':
    model = reSSRN(145, 14)

PATH = f'./weights/{file_name}_best_model.pickle'
model.load_state_dict(torch.load(PATH))

test = pd.read_csv('./test.csv')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_data = HSITestDataset(test, './pickle/')
test_loader = DataLoader(test_data, shuffle=False, num_workers=16,
                         batch_size=128)


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
#     print(list_diag)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
#     print(list_raw_sum)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
#     print(each_acc)
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(test_loader, name):
    count = 0
    # 模型测试
    model.eval()
    for inputs, label in tqdm(test_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        
        if count == 0:
            y_pred = outputs
            y_test = label
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_test = np.concatenate((y_test,label))

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 
                        'Grass-pasture-mowed','Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth','Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                        'Corn_senesced_green_weeds','Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                        'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk','Vinyard_untrained',
                        'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 
                        'Bitumen','Self-Blocking Bricks', 'Shadows']

    
    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)


    return classification,  oa * 100, each_acc * 100, aa * 100, kappa * 100

classification, oa, each_acc, aa, kappa = reports(test_loader, args.Dataset)
classification = str(classification)
# confusion = str(confusion)
file_name = './record/' + file_name + '_' + args.Model

with open(file_name, 'a') as x_file:
    x_file.write('\n')
    x_file.write('Kappa accuracy (%):{}'.format(kappa))
    x_file.write('\n')
    x_file.write('Overall accuracy (%):{}'.format(oa))
    x_file.write('\n')
    x_file.write('Average accuracy (%):{}'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
#     x_file.write('{}'.format(confusion))