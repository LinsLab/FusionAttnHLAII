import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
import torch
import numpy as np

import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import re
import time
import datetime
import random
import torch.nn as nn
random.seed(42)

from scipy import interp
import warnings
warnings.filterwarnings("ignore")

#from model2 import make_data,MyDataSet
from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import torch.utils.data as Data

import traceback
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import random

import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
#from utils import *

sys.path.append('/')
#from utils import hla_key_and_setTrans,hla_key_and_setTrans_2,hla_key_full_sequence
from feature_extraction import seq_feature,batch_seq_feature

#还需要引入数据处理文件中的函数(得到hla_dict等)
#还需要引入特征提取文件中提取序列特征的函数
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0001
NUM_EPOCHS = 20
hla_max_len=34
pep_max_len = 35


def P_or_N(list_entry):
    peptide_seq=[]
    p_entries=[]
    n_entries=[]
    for i in range(len(list_entry)):
        peptide_seq.append(list_entry[i][1])
        if float(list_entry[i][2])==1:
            p_entries.append(list_entry[i])
        if float(list_entry[i][2])==0:
            n_entries.append(list_entry[i])
    peptide_type=list(set(peptide_seq))
    return p_entries,n_entries,peptide_type

def train_predict_div(train_file,vaild_file):    
    
    data_train = pd.read_csv(train_file, index_col=None)
    data_val = pd.read_csv(train_file,  index_col=None)
   

    train_hla_list,train_peptide_list,train_label=data_train['HLA'],data_train['peptide'],data_train['label']
    val_hla_list,val_peptide_list,val_label=data_val['HLA'],data_val['peptide'],data_val['label']
    
    

    hla_full_sequence_dict=dict()
    full_seq_dict=json.load(open('../dataset/hla_sequnence_dict.csv'), object_pairs_hook=OrderedDict)
    for key,value in full_seq_dict.items():
        
        hla_full_sequence_dict[key]=value
    
    
    hla_feature_dict=dict()
    for key,value in hla_full_sequence_dict.items():
        hla_feature=seq_feature(value)
        hla_feature_dict[key]=hla_feature
           
    train_pep_feature=batch_seq_feature(train_peptide_list,pep_max_len,33)
    val_pep_feature=batch_seq_feature(val_peptide_list,pep_max_len,33)
    
    
    train_dataset=HPIDataset(root='data',  hla=train_hla_list, peptide=train_peptide_list,
                               y=train_label.astype(float),hla_feature_dict=hla_feature_dict,peptide_feature=train_pep_feature)
    vaild_dataset=HPIDataset(root='data', hla=val_hla_list, peptide=val_peptide_list,
                               y=val_label.astype(float),hla_feature_dict=hla_feature_dict,peptide_feature=val_pep_feature)

    return train_dataset, vaild_dataset





class HPIDataset(InMemoryDataset):
    def __init__(self,root='../data',transform=None,
                 pre_transform=None,hla=None, y=None, peptide=None,hla_feature_dict=None,peptide_feature=None):
        super(HPIDataset, self).__init__(root,transform, pre_transform)

        self.hla=hla
        self.peptide=peptide   #peptide的key就是它本身
        self.y=y
        self.hla_feature_dict=hla_feature_dict
        self.peptide_feature=peptide_feature
        self.data_hla=[]
        self.data_pep=[]
        self.process(self.hla,self.peptide,self.y,self.hla_feature_dict,self.peptide_feature)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_hla.pt', self.dataset + '_data_pep.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, hla_list, peptide_list, y_list, hla_feature_dict,peptide_feature):
        assert (len(hla_list) == len(peptide_list) and len(hla_list) == len(y_list)), 'The three lists must have the same length!'
        data_list_hla=[]
        data_list_pep=[]
        data_len = len(hla_list)
        for i in tqdm(range(data_len)):
            hla =hla_list[i]
            peptide = peptide_list[i]
            pep_feature=peptide_feature[i]
            labels = int(y_list[i])
            #contact_graph_data
            hla_feature=hla_feature_dict[hla]
    
            Data_hla={
                        'x': hla_feature,
                        'name': hla,
                        'y':labels
                    }
            
            
            Data_pep = {
                        'x': pep_feature,
                        'name': peptide,
                        'length': len(pep_feature)
                    }
            
            #print(Data_hla)
            data_list_hla.append(Data_hla)
            data_list_pep.append(Data_pep)
       
        self.data_hla= data_list_hla
        self.data_pep=data_list_pep

        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
       return self.data_hla[idx], self.data_pep[idx]
   
def test_data_div(test_file): 
    data_test = pd.read_csv(test_file, index_col=None)
    test_hla_list,test_peptide_list,test_label=data_test['HLA'],data_test['peptide'],data_test['label']
    
    hla_full_sequence_dict=dict()
    full_seq_dict=json.load(open('../dataset/hla_sequnence_dict.csv'), object_pairs_hook=OrderedDict)
    for key,value in full_seq_dict.items():
        hla_full_sequence_dict[key]=value
    
    hla_feature_dict=dict()
    for key,value in hla_full_sequence_dict.items():
        hla_feature=seq_feature(value)
        hla_feature_dict[key]=hla_feature
           
    test_pep_feature=batch_seq_feature(test_peptide_list,pep_max_len,33)
    test_dataset=HPIDataset(root='data',  hla=test_hla_list, peptide=test_peptide_list,
                               y=test_label.astype(float),hla_feature_dict=hla_feature_dict,peptide_feature=test_pep_feature)
    
    return test_dataset
    
    
def collate(data_list):
    data_hla_list=[]
    data_pep_list=[]
    for data in data_list:
        data_hla=data[0]
        data_pep=data[1]
        data_hla_list.append(data_hla)
        data_pep_list.append(data_pep)
    
    return data_hla_list,data_pep_list