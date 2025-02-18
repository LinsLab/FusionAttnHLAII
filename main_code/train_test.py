import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(42)

from scipy import interp
import warnings
warnings.filterwarnings("ignore")
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
from Transformer_performances import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

threshold=0.5
use_cuda=True
'''
def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])
'''
def transfer(y_prob, threshold):
    # 确保 y_prob 是一个 numpy 数组
    y_prob = np.array(y_prob)  # 如果 y_prob 是一个 list，将其转换为 numpy 数组
    return (y_prob > threshold).astype(int)  # 逐元素比较，大于 threshold 返回 1，否则返回 0

f_mean = lambda l: sum(l)/len(l)

def train(fold, model, device, train_loader, optimizer, epoch, epochs, loss_fn, TRAIN_BATCH_SIZE=512):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    time_train_ep = 0
    model.train()
    LOG_INTERVAL = 10
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list = []

    for batch_idx, data in tqdm(enumerate(train_loader)):
        t1 = time.time()
        data_hla = data[0]
        data_pep = data[1]

        pep_feature = torch.stack([torch.tensor(data['x'], dtype=torch.float32) for data in data_pep]).to(device)
        data_hla_feature = torch.stack([torch.tensor(hla['x'], dtype=torch.float32) for hla in data_hla]).to(device)
        data_pep_feature = torch.stack([torch.tensor(pep['x'], dtype=torch.float32) for pep in data_pep]).to(device)
        data_pep_length = torch.stack([torch.tensor(pep['length'], dtype=torch.float32) for pep in data_pep]).to(device)

        # 确保 y_true_train 的形状为 (batch_size, 1)
        y_true_train = torch.stack([torch.tensor(hla['y'], dtype=torch.float32) for hla in data_hla]).to(device).view(-1, 1)
        data_hla_lengths=torch.tensor([34]*data_pep_length.size(0))
        #output, _, _, _ = model(data_hla_feature, data_pep_feature, data_hla_lengths,data_pep_length)
        output= model(data_hla_feature, data_pep_feature, data_hla_lengths,data_pep_length)

        if torch.isnan(output).any():
            print("NaN or Inf in input")

        # 计算 loss
        #sprint('y_true_train',y_true_train[:10])
        loss = loss_fn(output, y_true_train)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

        
        y_prob_train = output.view(-1).cpu().detach().numpy()
        y_true_train = y_true_train.view(-1).cpu().numpy()  
        y_true_train_list.extend(y_true_train.tolist()) 
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(loss)

    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    print('Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(epoch, epochs, f_mean(loss_train_list), time_train_ep))

    # 直接传递 y_true_train_list 给 performances，而无需展平
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True)
    
    return ys_train, loss_train_list, metrics_train, time_train_ep




# predict
def predicting(fold,model, device, loader,epoch, epochs,loss_fn=nn.BCELoss()):
    model.eval()
    y_true_val_list, y_prob_val_list = [], []
    loss_val_list = []
    
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
       
            data_hla = data[0]
            data_pep = data[1]
            
            data_hla_feature=torch.stack([torch.tensor(hla['x'], dtype=torch.float32) for hla in data_hla]).to(device)
            data_pep_feature=torch.stack([torch.tensor(pep['x'], dtype=torch.float32) for pep in data_pep]).to(device)
            data_pep_length=torch.stack([torch.tensor(pep['length'], dtype=torch.float32) for pep in data_pep]).to(device)
            y_true_val = torch.stack([torch.tensor(hla['y'], dtype=torch.float32) for hla in data_hla]).to(device).view(-1, 1)
            
            data_hla_lengths=torch.tensor([34]*data_pep_length.size(0))
            output= model(data_hla_feature,data_pep_feature,data_hla_lengths,data_pep_length)
            loss = loss_fn(output, y_true_val)
            
           
            y_prob_val = output.view(-1).cpu().detach().numpy()
            y_true_val = y_true_val.view(-1).cpu().numpy()  
            y_true_val_list.extend(y_true_val.tolist())  
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(loss)
            
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print('Val  Epoch-{}/{}: Loss = {:.6f}'.format(epoch, epochs, f_mean(loss_val_list)))
        

        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = True)
    return ys_val, loss_val_list, metrics_val

    

def eval_step(model, test_loader,loss_fn=nn.BCELoss()):
    model.eval()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    y_true_test_list, y_prob_test_list = [], []
    y_preb_test_list=[]
    loss_test_list = []
   
   
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            data_hla = data[0]
            data_pep = data[1]
            
            data_hla_feature=torch.stack([torch.tensor(hla['x'], dtype=torch.float32) for hla in data_hla]).to(device)
            data_pep_feature=torch.stack([torch.tensor(pep['x'], dtype=torch.float32) for pep in data_pep]).to(device)
            data_pep_length=torch.stack([torch.tensor(pep['length'], dtype=torch.float32) for pep in data_pep]).to(device)
            y_true_test = torch.stack([torch.tensor(hla['y'], dtype=torch.float32) for hla in data_hla]).to(device).view(-1, 1)
            
            data_hla_lengths=torch.tensor([34]*data_pep_length.size(0)).to(device)
            output= model(data_hla_feature,data_pep_feature,data_hla_lengths,data_pep_length)
            loss = loss_fn(output, y_true_test)
           
            y_prob_test = output.view(-1).cpu().detach().numpy()
            y_true_test = y_true_test.view(-1).cpu().numpy() 
            y_true_test_list.extend(y_true_test.tolist())  
            y_prob_test_list.extend(y_prob_test)
            loss_test_list.append(loss)
            
        y_pred_test_list = transfer(y_prob_test_list, threshold)
        ys_test=(y_true_test_list, y_pred_test_list, y_prob_test_list)
        
        metrics_test = performances(y_true_test_list, y_pred_test_list, y_prob_test_list, print_ = True)
    return ys_test, metrics_test