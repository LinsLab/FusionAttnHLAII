import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.1):
        super(BiGRUEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        x: [batch_size, seq_len, input_dim]
        lengths: [batch_size], 实际序列长度（不包括填充）
        返回值:
            hidden_vec: [batch_size, d_model * 2]
        """
        # 使用 pack_padded_sequence 处理变长序列
        packed_input = rnn_utils.pack_padded_sequence(
            x, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_input)
        
        # hidden_view shape: [num_layers, 2, batch_size, d_model]
        hidden_view = hidden.view(self.num_layers, 2, x.size(0), self.d_model)
        # last hidden: [2, batch_size, d_model]
        last_layer_hidden = hidden_view[-1]  
        # concat [batch_size, d_model*2]
        hidden_vec = torch.cat([last_layer_hidden[0], last_layer_hidden[1]], dim=-1)
        hidden_vec = self.dropout(hidden_vec)  # [batch_size, d_model*2]
        return hidden_vec

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
        """
        x: [batch_size, seq_len, d_model]
        """
        attn_weights = self.attention(x)        # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_weights = self.dropout(attn_weights)
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch_size, d_model]
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
            nn.Linear(128, 1)  # 输出层
        )

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        return: [batch_size, 1]
        """
        return self.classifier(x)


class BiGRUOnlyModel(nn.Module):
    def __init__(self, input_dim=33, d_model=64, num_layers=2, dropout=0.3):
        """
        input_dim: 每个氨基酸的特征维度（例如33）
        d_model: BiGRU 隐藏单元维度（单向）
        num_layers: BiGRU 层数
        dropout: dropout 率
        """
        super(BiGRUOnlyModel, self).__init__()
        
        self.hla_encoder = BiGRUEncoder(input_dim, d_model, num_layers, dropout)
        self.pep_encoder = BiGRUEncoder(input_dim, d_model, num_layers, dropout)
        self.classifier = Classifier(input_dim=4*d_model, hidden_dim=512, dropout=dropout)
        
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
        # 分别通过 BiGRU 获取每条序列的隐藏向量（[batch_size, d_model*2]）
        hla_vec = self.hla_encoder(hla_inputs, hla_lengths)
        pep_vec = self.pep_encoder(pep_inputs, pep_lengths)
        combined_vec = torch.cat((hla_vec, pep_vec), dim=1)

        logits = self.classifier(combined_vec)  # [batch_size, 1]
        logits = torch.sigmoid(logits)
        return logits
