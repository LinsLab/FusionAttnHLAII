import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.utils.rnn as rnn_utils
import math
from sklearn import metrics
from sklearn import preprocessing
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
from tqdm import tqdm, trange

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

import os
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import difflib
plt.rcParams['font.size'] = 12

# 设置随机种子以确保结果可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 数据参数
pep_max_len = 35  # peptide max length
hla_max_len = 34  # hla max_length
tgt_len = pep_max_len + hla_max_len


batch_size = 512


n_layers, n_heads, fold = 1, 8, 4  

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return self.dropout(x)

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.1):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, d_model, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed_input = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_input)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, src, key_padding_mask=None):
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        src = self.layer_norm1(src + attn_output)
        ff_output = self.feed_forward(src)
        src = self.layer_norm2(src + ff_output)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, src, key_padding_mask=None):
        src = self.pos_enc(src)
        src = src.transpose(0, 1)
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask)
        src = src.transpose(0, 1)
        return src

class AttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=128, dropout=0.3):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_weights = self.dropout(attn_weights)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(x)

#FusionAttnHLAII-GRU
class DualEncoderModel_GRU(nn.Module):
    def __init__(self, input_dim=33, d_model=64, num_layers=2, n_heads=4, d_ff=256, dropout=0.3):
        super(DualEncoderModel, self).__init__()
        self.pep_encoder = GRUEncoder(input_dim, d_model, num_layers, dropout)
        self.hla_encoder = GRUEncoder(input_dim, d_model, num_layers, dropout)
        self.pep_pos_enc = PositionalEncoding(d_model, dropout)
        self.hla_pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = TransformerEncoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.attention_pool = AttentionPooling(d_model, hidden_dim=128, dropout=dropout)
        self.classifier = Classifier(input_dim=tgt_len*d_model, hidden_dim=256, dropout=dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, hla_inputs, pep_inputs, hla_lengths, pep_lengths):
        """
        hla_inputs: [batch_size, hla_seq_len, input_dim]
        pep_inputs: [batch_size, pep_seq_len, input_dim]
        hla_lengths: [batch_size]
        pep_lengths: [batch_size]
        """
        device = hla_inputs.device
        hla_lengths=hla_lengths.to(device)
        pep_lengths=pep_lengths.to(device)

        hla_enc = self.hla_encoder(hla_inputs, hla_lengths)  # [batch_size, hla_seq_len, d_model*2]
        pep_enc = self.pep_encoder(pep_inputs, pep_lengths)  # [batch_size, pep_seq_len, d_model*2]
        
        hla_enc=self.hla_pos_enc(hla_enc)
        pep_enc=self.pep_pos_enc(pep_enc)

        combined_enc = torch.cat((hla_enc, pep_enc), dim=1)  # [batch_size, hla_seq_len + pep_seq_len, d_model*2]

        # Transformer
        hla_mask = torch.arange(hla_enc.size(1), device=device)[None, :] >= hla_lengths[:, None]  # [B, hla_max_len]
        pep_mask = torch.arange(pep_enc.size(1), device=device)[None, :] >= pep_lengths[:, None]  # [B, pep_max_len]

       
        combined_mask = torch.cat([hla_mask, pep_mask], dim=1)  # [B, hla_max_len+pep_max_len]
        key_padding_mask = combined_mask.to(device)
        #key_padding_mask=key_padding_mask.to(device)
        transformer_output = self.transformer(combined_enc,key_padding_mask)  # [batch_size, hla_seq_len + pep_seq_len, d_model*2]

        
        trans_output = transformer_output.contiguous().view(transformer_output.shape[0], -1)
        logits = self.classifier(trans_output)  # [batch_size, 1]
        logits=torch.sigmoid(logits)
        return logits  


