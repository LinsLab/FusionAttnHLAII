import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
import pandas as pd
import argparse
import json
import re


from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
from Loader import train_predict_div,collate
from model import DualEncoderModel
from pytorchtools import EarlyStopping
from train_test_lihua import train,predicting


if __name__=='__main__':
    

    HPIdatasets=['hpi']
    ratio_list= [1, 3, 5]
    ratio = 1
    batch_size=512
    LR = 0.00001
    NUM_EPOCHS = 100
    print('dataset:', HPIdatasets)
    print('ratio', ratio)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    
    n_gpu = 1
    gpu_start = 0
    optional_mode = False  # Set whether you need to select the GPU ID yourself
    optional_gpus = [1]
    
    struc_hid_dim = 16
    max_pro_seq_len = 372
    if torch.cuda.is_available():
        if optional_mode:
            device_ids = optional_gpus
            n_gpu = len(device_ids)
            device = torch.device("cuda:{}".format(device_ids[0]))
        else:
            device_ids = []
            device = torch.device("cuda:{}".format(gpu_start))
            for i in range(n_gpu):
                device_ids.append(gpu_start+i)

    # Create necessary directories
    models_dir = '../models_pkl/'
    results_dir = '../results/'
    metrics_log=['epoch','type','auc','auc','mcc','f1','sensitivity','specificity','precision','recall','aupr','metrics_ep_avg','fold_metric_best','fold_ep_best','fold_best','metric_best','ep_best']
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, HPIdatasets[0])):
        os.makedirs(os.path.join(results_dir, HPIdatasets[0]))
    save_file=results_dir+'lr{}_epoch{}_model_gru_lihua_Bi_selfattentionpooling_layer3_100epoch_nhead4_dropout0.3_newmask_pos_chihua_dmodel64_256'.format(1*1e-3,512)
            
    write_f=open(save_file,'w')
    for i in range(len(metrics_log)):
        write_f.write('\t\t'+str(metrics_log[i]))
    write_f.write('\n')
    write_f.close()
    
   
    fold_best,metric_best, ep_best =0,0, -1
    scores = []
    for fold in range(1,6):
        
        train_file = '../dataset/train_dataset/train_fold_{}.csv'.format(fold)
        valid_file = '../dataset/train_dataset/val_fold_{}.csv'.format(fold)
       
        train_data, dev_data = train_predict_div(train_file, valid_file)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                    collate_fn=collate)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, shuffle=False,
                                                    collate_fn=collate)

        
        #print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label), Counter(val_data.label)))
        # Instantiate the model
        model = DualEncoderModel().to(device)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 1*1e-3)#, momentum = 0.99)
        fold_metric_best, fold_ep_best = 0, -1
        early_stopping = EarlyStopping(patience=5, verbose=True)
        
           
        #Train
        # Clear CUDA cache
        if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        for epoch in range(NUM_EPOCHS):
            ys_train, loss_train_list, metrics_train, time_train_ep = train(
                fold, model, device, train_loader, optimizer, epoch + 1, NUM_EPOCHS, loss_fn, batch_size
            )
            print('Predicting for validation data...')
            ys_val, loss_val_list, metrics_val = predicting(
                fold, model, device, dev_loader, epoch, NUM_EPOCHS, loss_fn
            )
            # Update global best metric
            save_dir = '../models_pkl/best_model'
            metrics_ep_avg = sum(metrics_val[:4])/4
            if metrics_ep_avg > fold_metric_best: 
                fold_metric_best, fold_ep_best = metrics_ep_avg, epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
             
            
            if metric_best<fold_metric_best:
                fold_best,metric_best,ep_best=fold,fold_metric_best,epoch
            metric_train_list=list(metrics_train)
            metric_val_list=list(metrics_val)
            # Write training logs
            write_f=open(save_file,'a')
            write_f.write('\t\t'+str(epoch))
            write_f.write('\t\ttrain')
            write_f.write('\t\t'+str(fold))
            for i in range(len(metric_train_list)):
                write_f.write('\t\t'+str(metric_train_list[i]))
            write_f.write('\t\t'+str(metrics_ep_avg))
            write_f.write('\t\t'+str(fold_metric_best))
            write_f.write('\t\t'+str(fold_ep_best))
            write_f.write('\t\t'+str(fold_best))
            write_f.write('\t\t'+str(metric_best))
            write_f.write('\t\t'+str(ep_best))
            write_f.write('\r\n')
            write_f.write('\t\t'+str(epoch))
            write_f.write('\t\tpredict')
            write_f.write('\t\t'+str(fold))
            for i in range(len(metric_val_list)):
                write_f.write('\t\t'+str(metric_val_list[i]))
            write_f.write('\t\t'+str(metrics_ep_avg))
            write_f.write('\t\t'+str(metric_best))
            write_f.write('\t\t'+str(ep_best))
            write_f.write('\t\t'+str(fold_best))
            write_f.write('\t\t'+str(metric_best))
            write_f.write('\t\t'+str(ep_best))
            write_f.write('\r\n')
            
           
            write_f.close()
            #early_stopping(sum(loss_val_list)/len(loss_val_list), model)
            
            #F1作为早停条件    
            if epoch > 30:
                F1=metrics_val[3]
                early_stopping(F1, model)
                if early_stopping.counter == 0:#如果早停机制计数器为0，即没有提升
                    best_test_score = fold_metric_best#最好的测试分数为当前测试分数
                if early_stopping.early_stop or epoch == NUM_EPOCHS - 1:#如果早停机制停止或者epoch等于args.epoch-1
                    scores.append(fold_metric_best)#将最好的测试分数加入到scores中
                    #path_saver = '/home/layomi/drive1/项目代码/HLA-II_code/HLAII_MODEL/models_pkl/gru_lihua_Bi_selfattentionpooling_layer3_100epoch_nhead4_dropout0.3_newmask_pos_chihua_dmodel64_256/best_model_fold{}_epoch{}'.format(fold,epoch)
                    #print('*****Path saver: ', path_saver)
                    #torch.save(model.state_dict(), path_saver)
                    break
        # Save the model    
        path_saver = '../models_pkl/best_model/best_model_fold{}_epoch{}'.format(fold,ep_best)
        print('*****Path saver: ', path_saver)
        torch.save(model.state_dict(), path_saver)
            
                        
                   
        # Release resources    
          
        del train_data
        del dev_data
        del train_loader
        del dev_loader
       
        
        
    


 



    
