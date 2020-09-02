#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  model.py
 * @Time    :  2020/03/02 01:07:30
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  和模型相关的函数库
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spl_sirna.sirna_util import get_seq_motif, idx_to_seq

USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

SEED = 1234
np.random.seed(1234)
torch.manual_seed(1234)
if USE_CUDA:
    torch.cuda.manual_seed(1234)


class Word2vecModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(Word2vecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size,
                                      self.embed_size,
                                      sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size,
                                     self.embed_size,
                                     sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size, (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        return: loss, [batch_size]
        '''
        batch_size = input_labels.size(0)

        # B * embed_size
        input_embedding = self.in_embed(input_labels)
        # B * (2*C) * embed_size
        pos_embedding = self.out_embed(pos_labels)
        # B * (2*C * K) * embed_size
        neg_embedding = self.out_embed(neg_labels)

        log_pos = torch.bmm(
            pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(
            neg_embedding,
            -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        # 对loss平均处理
        log_pos = F.logsigmoid(log_pos).sum(1) / log_pos.shape[1]
        log_neg = F.logsigmoid(log_neg).sum(1) / log_neg.shape[1]  # batch_size
        # log_pos = F.logsigmoid(log_pos).sum(1)
        # log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg  #[batchsize]
        return -loss

    def get_input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


class MultiMotifLSTMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers=1,
                 bidirectional=False,
                 dropout=0,
                 avg_hidden=True,
                 motif=[1, 2, 3],
                 loadvec=True,
                 device='cpu'):
        '''
        Desc：
            初始化模型，定义一些网络层级
        Args：
            vocab_size: int -- 5，即[A, G, U, C, T]
            embedding_dim: int -- 词向量的维度
            hidden_dim: int -- LSTM层hidden的维度
            output_dim: int -- 输出的维度
            n_layers: int -- LSTM的层数
            bidirectional: bool -- LSTM是否双向
            dropout: float -- drouput概率，使用在LSTM和Dropout层
            avg_hidden: bool -- 是否将hidden的平均值作为结果输出，如果是False，则使用最后一个Hidden作为LSTM的输出
        '''
        super(MultiMotifLSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.avg_hidden = avg_hidden
        self.motif = motif
        # 如果motif是整数，则赋1，如果为 list 则为长度
        self.motif_num = 1 if type(motif) == int else len(motif)
        self.pre_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.device = device

        self.input_embed_1 = nn.Embedding(vocab_size[0], embedding_dim[0])
        if loadvec:
            embed_1 = np.load('./embedding/motif-1/embedding-E100-C1-K1.npy')
            self.input_embed_1.weight.data.copy_(torch.from_numpy(embed_1))

        self.lstm_1 = nn.LSTM(embedding_dim[0],
                              hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=bidirectional,
                              dropout=(dropout if n_layers > 1 else 0))
        self.fc1_1 = nn.Linear(self.pre_output_dim, self.pre_output_dim)
        self.bn_1 = nn.BatchNorm1d(self.pre_output_dim)
        self.fc2_1 = nn.Linear(self.pre_output_dim, output_dim)

        self.input_embed_2 = nn.Embedding(vocab_size[1], embedding_dim[1])
        if loadvec:
            embed_2 = np.load('./embedding/motif-2/embedding-E25-C1-K1.npy')
            self.input_embed_2.weight.data.copy_(torch.from_numpy(embed_2))

        self.lstm_2 = nn.LSTM(embedding_dim[1],
                              hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=bidirectional,
                              dropout=(dropout if n_layers > 1 else 0))
        self.fc1_2 = nn.Linear(self.pre_output_dim, self.pre_output_dim)
        self.bn_2 = nn.BatchNorm1d(self.pre_output_dim)
        self.fc2_2 = nn.Linear(self.pre_output_dim, output_dim)

        self.input_embed_3 = nn.Embedding(vocab_size[2], embedding_dim[2])
        if loadvec:
            embed_3 = np.load('./embedding/motif-3/embedding-E200-C1-K2.npy')
            self.input_embed_2.weight.data.copy_(torch.from_numpy(embed_2))

        self.lstm_3 = nn.LSTM(embedding_dim[2],
                              hidden_dim,
                              num_layers=n_layers,
                              batch_first=True,
                              bidirectional=bidirectional,
                              dropout=(dropout if n_layers > 1 else 0))
        self.fc1_3 = nn.Linear(self.pre_output_dim, self.pre_output_dim)
        self.bn_3 = nn.BatchNorm1d(self.pre_output_dim)
        self.fc2_3 = nn.Linear(self.pre_output_dim, output_dim)

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.fc3 = nn.Linear(self.motif_num, output_dim)

        self.bn4 = nn.BatchNorm1d(self.pre_output_dim * self.motif_num)
        self.fc4 = nn.Linear(self.pre_output_dim * self.motif_num,
                             self.pre_output_dim)
        self.bn5 = nn.BatchNorm1d(self.pre_output_dim)
        self.fc5 = nn.Linear(self.pre_output_dim, output_dim)

        self.fc6 = nn.Linear(self.pre_output_dim * self.motif_num, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        '''
        Desc：
            Forward pass
        Args：
            seq: tensor(batch_size, seq_size) -- 输入的序列
        Returns：
            output: tensor(batch_size, output_dim=1) -- Predicted value
        '''
        seq = seq.long().to(self.device)
        # [batch_size, seq_size, embed_size]
        embed_seq_1 = self.dropout(self.input_embed_1(seq))
        seqs = idx_to_seq(seq, self.motif)
        seq_motif1 = get_seq_motif(seqs, self.motif[1])[0]
        seq_motif2 = get_seq_motif(seqs, self.motif[2])[0]
        seq_motif1 = torch.tensor(seq_motif1).long().to(self.device)
        seq_motif2 = torch.tensor(seq_motif2).long().to(self.device)
        # embed_seq_2 = self.dropout(self.input_embed_2(seq_motif[1]))
        embed_seq_2 = self.input_embed_2(seq_motif1)
        embed_seq_3 = self.dropout(self.input_embed_3(seq_motif2))

        # lstm_output: [batch size, seq_len, hid dim * num directions]
        #hidden, cell: [num layers * num directions, batch size, hidden_dim]
        lstm_output_1, (hidden_1, cell_1) = self.lstm_1(embed_seq_1)
        lstm_output_2, (hidden_2, cell_2) = self.lstm_2(embed_seq_2)
        lstm_output_3, (hidden_3, cell_3) = self.lstm_3(embed_seq_3)

        # hidden: [batch_size, hidden_dim * num_directions]
        hidden_1 = self.handle_hidden(lstm_output_1, hidden_1)
        hidden_2 = self.handle_hidden(lstm_output_2, hidden_2)
        hidden_3 = self.handle_hidden(lstm_output_3, hidden_3)

        # concatenate hidden, [batch_size, pre_output_dim*motif]
        hidden = torch.cat((hidden_1, hidden_2, hidden_3), dim=1)
        # hidden = self.bn4(hidden)
        # [batch_size, pre_output_dim*motif] -> [batch_size, hidden_dim]
        pre_output = self.leaky_relu(self.bn6(self.fc6(hidden)))
        pre_output = self.dropout(pre_output)
        # hidden_dim -> output_dim
        output = self.fc7(pre_output)

        # pre_output_1 = self.relu(self.bn_1(self.fc1_1(hidden_1)))
        # pre_output_1 = self.leaky_relu(self.bn_1(self.fc1_1(hidden_1)))
        # pre_output_1 = self.dropout(pre_output_1)
        # output_1 = self.fc2_1(pre_output_1)  # [batch_size, output_dim]

        # pre_output_2 = self.relu(self.bn_2(self.fc1_2(hidden_2)))
        # pre_output_2 = self.dropout(pre_output_2)
        # output_2 = self.fc2_2(pre_output_2)  # [batch_size, output_dim]

        # pre_output_3 = self.relu(self.bn_3(self.fc1_3(hidden_3)))
        # pre_output_3 = self.dropout(pre_output_3)
        # output_3 = self.fc2_3(pre_output_3)  # [batch_size, output_dim]

        # # batch_size, pre_hidden_dim -> batch_size, pre_hidden_dim*3 -> batch_size, 1
        # output_pre = torch.cat((pre_output_1, pre_output_2, pre_output_3), dim=1)
        # output_pre = self.bn4(output_pre)
        # # batch_size, pre_hidden_dim*3 -> batch_size, pre_hidden_dim
        # output_pre = self.relu(self.bn5(self.fc4(output_pre)))
        # output_pre = self.dropout(output_pre)
        # # batch_size, pre_hidden_dim -> batch_size, output_dim
        # output = self.fc5(output_pre)

        # output_pre = torch.cat((output_1, output_2, output_3), dim=1)
        # output_pre = torch.cat((output_1, output_2), dim=1)
        # output = self.fc3(output_pre)

        return output

    def handle_hidden(self, lstm_output, hidden):
        if self.avg_hidden:
            hidden = torch.sum(lstm_output, 1) / lstm_output.size(1)
        else:
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                hidden = hidden[-1, :, :]
        return hidden


class AttentionModel(nn.Module):
    def __init__(self, method, hidden_dim, bidirectional=False, device='cpu'):
        '''
        Desc：
            初始化attention层
        Args：
            method: string  --  attention的方法，有general，concat和dot
            hidden_dim: int, hidden_dim*num dirs  --  hidden的维度，应该是hidden_dim*num dirs
            device  --  是否使用cuda
        '''
        super(AttentionModel, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        if bidirectional:
            self.hidden_dim = self.hidden_dim * 2
        self.device = device

        if self.method == 'general':
            self.atten = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.method == 'concat':
            self.atten = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.tanh = nn.Tanh()
            # parameter会在反向传播时自动更新
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_dim))

    def forward(self, hidden, encoder_outputs):
        '''
        Desc：
            attention层的前向传播
        Args：
            hidden: (batch size, hidden_dim*num dir)  --  论文中decoder的上一个hidden，这里是最后一层的第20个hidden
            encoder_outputs: (seq_len, batch_size, hidden_dim*num dir)  --  论文中encoder的output，这里是最后一层前19个output
        Returns：
            energy: (seq_len, batch_size)  --  返回每个hidden对应的权重
        '''
        batch_size, seq_len = encoder_outputs.shape[1], encoder_outputs.shape[
            0]
        # shape (seq len, batch_size)
        energy = Variable(torch.zeros(seq_len, batch_size)).to(self.device)
        for i in range(seq_len):
            energy[i] = self.score(hidden, encoder_outputs[i])
        return F.softmax(energy, dim=0).transpose(0, 1).unsqueeze(
            1)  #(batch size, 1, seq len)

    def score(self, hidden, encoder_output):
        '''
        Desc：
            计算hidden和encoder outputs的energy分数
        Args：
            hidden: (batch size, hidden_dim*num dir)  --  decoder的hidden
            encoder_output: (batch size, hiddendim*numdir)  --  encoder最后一层的每个output
        Returns：
            energy: (batch size, )   --  计算得到的分数
        '''
        if self.method == 'dot':
            energy = torch.sum(hidden * encoder_output, axis=1)  # (batchsize,)
        elif self.method == 'general':
            # (batch size, hidden_dim*numdir)
            energy = self.atten(encoder_output)
            energy = torch.sum(hidden * energy, axis=1)
        elif self.method == 'concat':
            # (batch size, hidden dim)
            energy = self.tanh(
                self.atten(torch.cat((hidden, encoder_output), axis=1)))
            energy = torch.sum(energy * self.other, axis=1)
        return energy  # (batch size, )


class AttenLSTMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers=1,
                 bidirectional=False,
                 dropout=0,
                 avg_hidden=True,
                 vector_path=None,
                 attention_method=None,
                 save_energy=False,
                 device='cpu'):
        '''
        Desc：
            初始化单一输入LSTM模型，定义一些网络层级
        Args：
            vocab_size: int -- 5，即[A, G, U, C, T]
            embedding_dim: int -- 词向量的维度
            hidden_dim: int -- LSTM层hidden的维度
            output_dim: int -- 输出的维度
            n_layers: int -- LSTM的层数
            bidirectional: bool -- LSTM是否双向
            dropout: float -- drouput概率，使用在LSTM和Dropout层
            avg_hidden: bool -- 是否将hidden的平均值作为结果输出，如果是False，则使用最后一个Hidden作为LSTM的输出
        '''
        super(AttenLSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.avg_hidden = avg_hidden
        self.pre_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.device = device

        self.input_embed = nn.Embedding(vocab_size, embedding_dim)
        if vector_path is not None:
            vec_embed = np.load(vector_path)
            self.input_embed.weight.data.copy_(torch.from_numpy(vec_embed))

        self.attention_method = attention_method
        self.attention = AttentionModel(attention_method, hidden_dim,
                                        self.bidirectional, self.device)
        # self.energies = list()
        self.save_energy = save_energy

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=(dropout if n_layers > 1 else 0))
        self.fc1 = nn.Linear(self.pre_output_dim, self.pre_output_dim)
        self.bn = nn.BatchNorm1d(self.pre_output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.pre_output_dim, output_dim)

    def forward(self, seq):
        '''
        Desc：
            Forward pass
        Args：
            seq: tensor(batch_size, seq_size) -- 输入的序列
        Returns：
            output: tensor(batch_size, output_dim=1) -- Predicted value
        '''
        # embed_seq: [batch_size, seq_size, embed_size]
        seq = seq.long().to(self.device)
        embed_seq = self.dropout(self.input_embed(seq))
        #lstm_output: [batch size, seq_len, hid dim * num directions]
        #hidden, cell: [num layers * num directions, batch size, hidden_dim]
        lstm_output, (hidden, cell) = self.lstm(embed_seq)

        if self.avg_hidden:
            hidden = torch.sum(lstm_output, 1) / lstm_output.size(1)
        else:
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                # hidden = self.dropout(hidden[-1, :, :])
                hidden = hidden[-1, :, :]

        # hidden: [batch_size, hidden_dim * num_directions]
        if self.attention_method is not None:
            # [seq_len, batch_size, hidden_dim*num_dirs]
            encoder_outputs = lstm_output.transpose(0, 1)
            # [batch_size, 1, seq_len]
            energy = self.attention(hidden, encoder_outputs)
            # self.energies.append(energy)
            # context: #[batch_size, hidden_dim*numdirs]=[batch_size, 1, seq_len] · [batch_size, seq_len, hidden_dim*numdirs]
            context = energy.bmm(encoder_outputs.transpose(0, 1)).squeeze()
        else:
            context = hidden

        # print('energy:\n{}'.format(energy[0]))
        pre_output = self.leaky_relu(self.bn(self.fc1(context)))
        pre_output = self.dropout(pre_output)
        output = self.fc2(pre_output)
        # return: [batch_size, output_dim]
        if self.save_energy:
            output = (output, energy)
        else:
            output = (output, output)
        return output


class AttenPadLSTMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers=1,
                 bidirectional=False,
                 dropout=0,
                 avg_hidden=True,
                 pad_idx=None,
                 vector_path=None,
                 attention_method=None,
                 save_energy=False,
                 device='cpu'):
        '''
        Desc：
            初始化单一输入LSTM模型，定义一些网络层级
        Args：
            vocab_size: int -- 5，即[A, G, U, C, T]
            embedding_dim: int -- 词向量的维度
            hidden_dim: int -- LSTM层hidden的维度
            output_dim: int -- 输出的维度
            n_layers: int -- LSTM的层数
            bidirectional: bool -- LSTM是否双向
            dropout: float -- drouput概率，使用在LSTM和Dropout层
            avg_hidden: bool -- 是否将hidden的平均值作为结果输出，如果是False，则使用最后一个Hidden作为LSTM的输出
        '''
        super(AttenLSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.avg_hidden = avg_hidden
        self.pre_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.device = device

        self.input_embed = nn.Embedding(vocab_size,
                                        embedding_dim,
                                        padding_idx=pad_idx)
        if vector_path is not None:
            vec_embed = np.load(vector_path)
            self.input_embed.weight.data.copy_(torch.from_numpy(vec_embed))

        self.attention_method = attention_method
        # (seq len, batch size)
        self.attention = AttentionModel(attention_method, hidden_dim,
                                        self.bidirectional, self.device)
        # self.energies = list()
        self.save_energy = save_energy

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=(dropout if n_layers > 1 else 0))
        self.fc1 = nn.Linear(self.pre_output_dim, self.pre_output_dim)
        self.bn = nn.BatchNorm1d(self.pre_output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.pre_output_dim, output_dim)

    def forward(self, seq):
        '''
        Desc：
            Forward pass
        Args：
            seq: tensor(batch_size, seq_size) -- 输入的序列
        Returns：
            output: tensor(batch_size, output_dim=1) -- Predicted value
        '''
        # get sequence length and total length
        mask = torch.gt(seq, 0)
        seq_lens = torch.sum(mask, axis=1)
        total_length = seq.shape[1]

        # embed_seq: [batch_size, seq_size, embed_size]
        seq = seq.long().to(self.device)
        embed_seq = self.dropout(self.input_embed(seq))

        embed_packed = nn_utils.rnn.pack_padded_sequence(embed_seq,
                                                         seq_lens,
                                                         batch_first=True,
                                                         enforce_sorted=False)
        output_packed, (hidden, cell_state) = self.lstm(embed_packed)

        #lstm_output: [batch size, seq_len, hid dim * num directions]
        #hidden, cell: [num layers * num directions, batch size, hidden_dim]
        lstm_output, length = nn_utils.rnn.pad_packed_sequence(
            output_packed, batch_first=True, total_length=total_length)

        if self.avg_hidden:
            # get seq_lens with shape of [batch_size, hidden_dim] to torch.div
            div_val = seq_lens.reshape(2, 1).repeat(1,
                                                    hidden.shape[-1]).float()
            hidden = torch.div(torch.sum(lstm_output, 1), div_val)
        else:
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                # hidden = self.dropout(hidden[-1, :, :])
                hidden = hidden[-1, :, :]

        # hidden: [batch_size, hidden_dim * num_directions]
        if self.attention_method is not None:
            # [seq_len, batch_size, hidden_dim*num_dirs]
            encoder_outputs = lstm_output.transpose(0, 1)
            # [batch_size, 1, seq_len]
            energy = self.attention(hidden, encoder_outputs)
            # self.energies.append(energy)
            # context: #[batch_size, hidden_dim*numdirs]=[batch_size, 1, seq_len] · [batch_size, seq_len, hidden_dim*numdirs]
            context = energy.bmm(encoder_outputs.transpose(0, 1)).squeeze()
        else:
            context = hidden

        # print('energy:\n{}'.format(energy[0]))
        pre_output = self.leaky_relu(self.bn(self.fc1(context)))
        pre_output = self.dropout(pre_output)
        output = self.fc2(pre_output)
        # return: [batch_size, output_dim]
        if self.save_energy:
            output = (output, energy)
        else:
            output = (output, output)
        return output


class EmbeddingLSTMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers=1,
                 bidirectional=False,
                 dropout=0,
                 avg_hidden=True,
                 vector_path=None,
                 device='cpu'):
        '''
        Desc：
            初始化单一输入LSTM模型，定义一些网络层级
        Args：
            vocab_size: int -- 5，即[A, G, U, C, T]
            embedding_dim: int -- 词向量的维度
            hidden_dim: int -- LSTM层hidden的维度
            output_dim: int -- 输出的维度
            n_layers: int -- LSTM的层数
            bidirectional: bool -- LSTM是否双向
            dropout: float -- drouput概率，使用在LSTM和Dropout层
            avg_hidden: bool -- 是否将hidden的平均值作为结果输出，如果是False，则使用最后一个Hidden作为LSTM的输出
        '''
        super(EmbeddingLSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.avg_hidden = avg_hidden
        self.pre_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.device = device

        self.input_embed = nn.Embedding(vocab_size, embedding_dim)
        if vector_path is not None:
            vec_embed = np.load(vector_path)
            self.input_embed.weight.data.copy_(torch.from_numpy(vec_embed))

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=(dropout if n_layers > 1 else 0))
        self.fc1 = nn.Linear(self.pre_output_dim, self.pre_output_dim)
        self.bn = nn.BatchNorm1d(self.pre_output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.pre_output_dim, output_dim)

    def forward(self, seq):
        '''
        Desc：
            Forward pass
        Args：
            seq: tensor(batch_size, seq_size) -- 输入的序列
        Returns：
            output: tensor(batch_size, output_dim=1) -- Predicted value
        '''
        # embed_seq: [batch_size, seq_size, embed_size]
        seq = seq.long().to(self.device)
        embed_seq = self.dropout(self.input_embed(seq))
        #lstm_output: [batch size, seq_len, hid dim * num directions]
        #hidden, cell: [num layers * num directions, batch size, hidden_dim]
        lstm_output, (hidden, cell) = self.lstm(embed_seq)

        if self.avg_hidden:
            hidden = torch.sum(lstm_output, 1) / lstm_output.size(1)
        else:
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                # hidden = self.dropout(hidden[-1, :, :])
                hidden = hidden[-1, :, :]

        # hidden: [batch_size, hidden_dim * num_directions]
        # hidden = self.dropout(hidden)
        pre_output = self.leaky_relu(self.bn(self.fc1(hidden)))
        pre_output = self.dropout(pre_output)
        output = self.fc2(pre_output)
        # return: [batch_size, output_dim]
        return output


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim,
                             int((embedding_dim + output_dim) / 2))
        self.fc2 = nn.Linear(int((embedding_dim + output_dim) / 2), output_dim)
        self.dropout = nn.Dropout(dropout)

    ''' seq: [batch_size, seq_size]'''

    def forward(self, text):
        # [batch_size,seq_len,emb_dim]
        embedded = self.embedding(text.long().to('cuda'))
        # batch_size, embed_size
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze()
        output = self.dropout(self.fc1(pooled))
        return self.fc2(output)


def count_parameters(model=None):
    '''
    Desc：
        计算模型中参数个数
    Args：
        model  --  待参数计数的模型
    Returns：
        res  --  model中参数的个数
    '''
    if model is None:
        raise ValueError("model不可为空")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
