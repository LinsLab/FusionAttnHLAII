import torch
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import pandas as pd
import argparse
import json
import re
from collections import OrderedDict
import pickle
from Loader_lihua import *
from train_test_lihua import *
import  random
from tqdm import tqdm

from model import DualEncoderModel
from feature_extraction import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.00001


def test(dataset_file,model_file):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Test！')
    
    test_data=test_data_div(dataset_file)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    threshold=0.5
    model_eval = DualEncoderModel()
    #model_eval=BiGRUOnlyModel()
    model_eval.to(device)
    model_eval.load_state_dict(torch.load(model_file, map_location='cpu'), strict = True)#加载一些训练好的模型参数,其实相当于参数初始化，比直接初始化为0效果好
    model_eval.eval()#加载模型
    
    ys,metrics=eval_step(model_eval,test_loader)
    
    return ys
   
def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in (y_prob)]) 

#将结果进行追加记录到file1
def recording_w(file1,record,w_or_a='w'):
    if isinstance(record,dict):
        #print('True')

        with open (file1,w_or_a) as f:
            for key,value in record.items():
                print('{}:{}'.format(key,value),file=f)
        f.close()
    
if __name__ == '__main__':
   
    file_independent='../dataset/train_dataset/test_data.csv'
    file_independent_comblib='../test_set1_subset/comblib_subset.txt'
    file_independent_consensus='../test_set1_subset/consensus3_subset.txt'
    file_independent_netmhciipan_ba='../test_set1_subset/netmhciipan_ba_subset.txt'
    file_independent_netmhciipan_ba_42='../test_set1_subset/netmhciipan_ba-4.2_subset.txt'
    file_independent_netmhciipan_el='../test_set1_subset/netmhciipan_el_subset.txt'
    file_independent_netmhciipan_el_42='../test_set1_subset/netmhciipan_el-4.2_subset.txt'
    file_independent_smm='../test_set1_subset/smm_align_subset.txt'
    file_independent_HLAIImaster='../test_set1_subset/HLAIImaster_subset.csv'
    file_external_comblib='../test_set2_subset/comblib_result.txt'
    file_external_concensus='../test_set2_subset/consensus3_result.txt'
    file_external_netmhciipan_ba='../test_set2_subset/netmhciipan_ba_result.txt'
    file_external_netmhciipan_ba_42='../test_set2_subset/netmhciipan_ba-4.2_result.txt'
    file_external_netmhciipan_el='../test_set2_subset/netmhciipan_el_result.txt'
    file_external_netmhciipan_el_42='../test_set2_subset/netmhciipan_el-4.2_result.txt'
    file_external_smm='../test_set2_subset/smm_align_result.txt'
    file_external_HLAIImaster='../test_set2_subset/HLAIImaster_subset.csv'
    
    
    file_independent_subset_list=[file_independent_comblib,file_independent_netmhciipan_ba,file_independent_consensus,file_independent_smm,file_independent_HLAIImaster_35]
    file_external_subset_list=[file_external_balaence,file_external_netmhciipan_ba_42,file_external_comblib,file_external_concensus,file_external_netmhciipan_ba,file_external_smm]
    for dataset_file in file_external_subset_list:
        
        ys_prob_list=[]
        model_folder='../model_pkl/best_model'
        for file in os.listdir(model_folder):
            model_file_path=os.path.join(model_folder,file)
            (ys_true,ys_preb,ys_prob)=test(dataset_file,model_file_path)
            ys_prob_list.append(np.array(ys_prob))
            
        y_prob_mean = [np.mean(scores) for scores in zip(*ys_prob_list)]     
        y_preb_list=transfer(y_prob_mean, threshold = 0.5)
        y_preb_list=[int(d) for d in y_preb_list]    
            
        
        all_data_list=pd.read_csv(dataset_file)
        all_data_list=all_data_list.values.tolist()
        
        if len(all_data_list)!=len(y_prob_mean):
            print('Error!')
        y_true_list=[]
        for i in range(len(all_data_list)):
            all_data_list[i].extend([y_prob_mean[i],all_data_list[i][2],y_preb_list[i]]) #预测分数，真实类别，预测类别
            y_true_list.append(int(all_data_list[i][2]))
            
        result_folder='../results/data_result_file/{}'.format(dataset_file.split('/')[-2])  
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        result_dataset_write_file=os.path.join(result_folder,'{}'.format(dataset_file.split('/')[-1]))
        with open(result_dataset_write_file,'w') as f:
            for sublist in all_data_list:
                line = ','.join(map(str, sublist)) + '\n'
                f.write(line)
        
        
        metrics_set = performances(y_true_list, y_preb_list, y_prob_mean, print_ = True)
        metrics_dict=dict()
        metrics_wfile1=os.path.join(result_folder,'metrics.txt')
        metrics_dict['{}_metrics'.format(dataset_file.split("/")[-1])]=metrics_set
        recording_w(metrics_wfile1, metrics_dict,'a+')
      
    

