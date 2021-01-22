#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  data.py
 * @Time    :  2020/04/03 14:45:27
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  和数据预处理相关的函数库
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.utils.data as tud
import sirna_util
import torch

SEED = 1234
np.random.seed(1234)


def char_remove(data, chr_list=None):
    '''
    Desc：
        将data中所有的chr_list中的字符去除
    Args：
        data: pd.Series/ndarray/list  --  待处理数据
        chr_list: ndarray/list -- 待移除的字符，若为None，则原样返回data
    Returns：
       data -- 去除所有chr_list中字符后的data
    '''
    if chr_list is None:
        return data
    if type(data) == pd.Series:
        for ch in chr_list:
            data = data.str.replace(ch, '')
    elif type(data) == np.ndarray or type(data) == list:
        for ch in chr_list:
            for i, w in enumerate(data):
                data[i] = w.replace(ch, '')
    print(data)
    return data


def cal_time(start_time, end_time):
    '''
    Desc：
        计算时间差，返回小时，分钟和秒
    Args：
        start_time  --  开始时间
        end_time  --  结束时间
    Returns：
        elapsed_hours, elapsed_mins, elapsed_secs  --  经过的小时，分钟，秒
    '''
    if end_time < start_time:
        raise ValueError("结束时间不可小于开始时间")
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_mins = int((elapsed_time - elapsed_hours * 3600) // 60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60 - elapsed_hours * 3600)
    return elapsed_hours, elapsed_mins, elapsed_secs


def copy_part_of_data(xdata, ydata, yrange=[], copytimes=1):
    '''
    Desc：
        按照y值的范围，将数据集的部分数据进行拷贝扩增
    Args：
        xdata: ndarray  --  数据集中的x
        ydata: ndarray  --  数据集中的y
        yrange: tuple/list(tuple)  --  需扩增的数据的y值范围，若为空，则扩增全部样本
        copytimes: int  --  扩增的次数
    Returns：
        x, y: ndarray  --  扩增后的数据，y是一维数据
    '''
    # 异常处理
    if type(xdata) not in [np.ndarray, list
                           ] or type(ydata) not in [np.ndarray, list]:
        raise Exception("xdata和ydata类型需要为list或numpy.ndarray")
    if len(xdata) == 0:
        raise Exception("xdata不能为空")
    if len(yrange) > 0 and type(yrange) != list and type(yrange) != tuple:
        raise Exception("yrange的类型必须为list或tuple，如[(1, 2)], (1,2)")
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # label标签不超过1维
    ydata = ydata.flatten()
    if len(ydata.shape) > 1:
        raise Exception("ydata的维度不可以超过1维")
    # yrange不指定则拷贝全部数据
    if type(yrange) is tuple:
        yrange = [yrange]
    if len(yrange) == 0:
        ymin, ymax = np.min(ydata), np.max(ydata)
        ydata = [(ymin, ymax)]

    # 对要拷贝的数据进行拼接
    xdata_shape = list(xdata.shape)
    xdata_shape[0] = 0
    ydata_shape = list(ydata.shape)
    ydata_shape[0] = 0
    res_copy_x, res_copy_y = np.empty(xdata_shape), np.empty(ydata_shape)
    for i in range(copytimes):
        for (miny, maxy) in yrange:
            idx = (ydata >= miny) & (ydata <= maxy)
            copy_data_x, copy_data_y = xdata[idx], ydata[idx]
            res_copy_x = np.concatenate((res_copy_x, copy_data_x))
            res_copy_y = np.concatenate((res_copy_y, copy_data_y))

    xdata = np.concatenate((xdata, res_copy_x[1:]))
    ydata = np.concatenate((ydata, res_copy_y[1:]))
    return xdata, ydata


def truncate_part_of_data(xdata, ydata, yrange=[]):
    '''
    Desc：
        按照y值的范围，将数据集的部分数据进行截断
    Args：
        xdata: ndarray  --  数据集中的x
        ydata: ndarray  --  数据集中的y
        yrange: list(tuple)  --  需截断的数据的y值范围，为空则不截断
    Returns：
        x, y: ndarray  --  截断后的数据，y是一维数据
    '''
    # 异常处理
    if type(xdata) not in [np.ndarray, list
                           ] or type(ydata) not in [np.ndarray, list]:
        raise Exception("xdata和ydata类型需要为numpy.ndarray")
    if len(xdata) == 0:
        raise Exception("xdata不能为空")
    if len(yrange) > 0 and type(yrange) != list and type(yrange) != tuple:
        raise Exception("yrange的类型必须为list或tuple，如[(1, 2)], (1,2)")

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    # label标签不超过1维
    ydata = ydata.flatten()
    if len(ydata.shape) > 1:
        raise Exception("ydata的维度不可以超过1维")
    if type(yrange) is tuple:
        yrange = [yrange]
    if len(yrange) == 0:
        return xdata, ydata

    # 提取需要的数据
    for (miny, maxy) in yrange:
        idx = (ydata < miny) | (ydata > maxy)
        xdata, ydata = xdata[idx], ydata[idx]
    return xdata, ydata


def get_sample_freq(data=None):
    '''
    Desc：
        统计样本出现次数
    Args：
        data:ndarray/pd.Series/list -- 待统计数据
    Returns：
        res:ndarray  --  一列是样本，一列是对应的频次
    '''
    if data is None:
        raise ValueError("data不能为空")
    if type(data) not in [pd.DataFrame, pd.Series, list, np.ndarray]:
        raise ValueError("data类型只支持DataFrame, Series, ndarray和list")
    if type(data) in [list, np.ndarray]:
        data = pd.Series(data)
    elif type(data) is pd.DataFrame:
        data = data[0]
    freq = data.value_counts()
    samples = freq.index
    res = pd.DataFrame(list(zip(samples, freq)))
    return res.values


def standardize_data(data, axis=None, std=0, mean=0):
    '''
    Desc：
        对数据进行标准化处理，返回标准化后的数据
    Args：
        data: ndarray/list  --  待标准化的数据
        axis: int  --  标准化的维度
        std: list/int  --  可选的标准差
        mean: list/int  --  可选的均值
    Returns：
        data: ndarray  --  标准化后的data
    '''
    # 异常处理
    listFlag = False
    if type(data) is list:
        data = np.array(data)
        listFlag = True
    elif type(data) != list and type(data) != np.ndarray:
        raise TypeError("data类型应该为list或np.ndarray")
    if type(axis) != int:
        raise TypeError("axis类型应该为整形")

    # 判断要标准化的维度和是否输入指定的均值和标准差
    if axis is None and std == 0:
        std = np.std(data)
    elif axis is not None and std == 0:
        std = np.std(data, axis=axis)
    if axis is None and mean == 0:
        mean = np.mean(data)
    elif axis is not None and mean == 0:
        mean = np.mean(data, axis=axis)

    # 标准化数据
    data = (data - mean) / std
    return data.tolist() if listFlag else data


def normalize_data(data, axis=None):
    '''
    Desc：
        对数据进行归一化，即变成0-1范围内小数
    Args：
        data: ndarray/list  --  待归一化的数据
        axis: int  --  归一化的维度
    Returns：
        data: ndarray  --  归一化后的data
    '''
    # 异常处理
    listFlag = False
    if type(data) is list:
        data = np.array(data)
        listFlag = True
    elif type(data) != list and type(data) != np.ndarray:
        raise TypeError("data类型应该为list或np.ndarray")
    if type(axis) != int and axis is not None:
        raise TypeError("axis类型应该为整形")

    # 判断要归一化的维度，计算最大值和最小值
    data_max = data_min = 0
    if axis is None:
        data_max = np.max(data)
        data_min = np.min(data)
    else:
        data_max = np.max(data, axis=axis)
        data_min = np.min(data, axis=axis)

    # 归一化数据
    data = (data - data_min) / (data_max - data_min)
    return data.tolist() if listFlag else data


def split_dataset(xdata,
                  ydata,
                  valid_size=0.2,
                  test_size=0.2,
                  shuffle=True,
                  random_state=None):
    '''
    Desc：
        对数据集进行划分，分为训练集、验证集、测试集
    Args：
        xdata: ndarray  --  所有的特征集
        ydata: ndarray  --  所有的label
        valid_size: float/int  --  验证集占所有数据的比例，如果为int则是样本数
        test_size: float/int  --  测试集占所有数据的比例，如果为int则是样本数
        shuffle: boolean  --  是否需要将数据集打乱后划分
        random_state:  --  如果是整形，则作为随机数种子，如果是随机状态实例，则作为随机数生成器使用，默认None，调用`np.random`
    Returns：
        x_train, x_val, x_test, y_train, y_val, y_test: ndarray  --  划分好的训练集、验证集、测试集
    '''
    # 异常处理
    if type(valid_size) is float:
        if valid_size > 1 or valid_size < 0:
            raise ValueError("float类型valid_size必须在0到1之间")
    elif type(valid_size) is int:
        if valid_size < 0 or valid_size > len(xdata):
            raise ValueError("int类型valid_size必须在0到样本总数之间")
    else:
        raise TypeError("valid_size类型错误")

    if type(test_size) is float:
        if test_size > 1 or test_size < 0:
            raise ValueError("float类型test_size必须在0到1之间")
    elif type(test_size) is int:
        if test_size < 0 or test_size > len(xdata):
            raise ValueError("int类型test_size必须在0到样本总数之间")
    else:
        raise TypeError("test_size类型错误")

    # 划分出测试集
    x_train = x_val = x_test = y_train = y_test = y_val = []
    if test_size == 0:
        x_train, y_train = xdata, ydata
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            xdata,
            ydata,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state)

    # 划分出训练集和验证集
    if valid_size != 0:
        valid_size = valid_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=valid_size,
            shuffle=shuffle,
            random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test


class EmbeddedTextDataset(tud.Dataset):
    def __init__(self,
                 xdata,
                 ydata,
                 word_to_idx=None,
                 idx_to_word=None,
                 max_vocab_size=None,
                 encode_label=False):
        '''
        Desc：
            对文本数据进行情感分类或者其他预测任务时，进行通用的Word Embedding编码，并将数据封装成Dataset，不同长度的使用pad补充为0
        Args：
            xdata: ndarray(N,  ) -- 所有的数据，字符串
            ydata: ndarray -- 可能存在的对应目标值，如positive和negative的情感标签
            word_to_idx: dict  --  可选的word到idx的对应关系字典，若使用padding，则生成word_to_idx的时候，0需要是padding，不可以是其他的，不然会把其他字符转化为padding
            idx_to_word: list/ndarray  --  可选的idx到word的对应关系列表
                                          若不存在word_to_idx和idx_to_word，则会自动生成
            max_vocab_size: int  --  最大的字典长度
            encoded_data: ndarray -- 编码后的文本数据
        '''
        super(EmbeddedTextDataset, self).__init__()
        # 异常处理：判断xdata和ydata长度是否一致
        if len(xdata) != len(ydata):
            raise ValueError("xdata长度和ydata长度必须一致")
        if type(xdata) != np.ndarray:
            xdata = np.array(xdata)
        if type(ydata) != np.ndarray:
            ydata = np.array(ydata)
        if len(xdata.flatten().shape) > 1:
            raise ValueError("xdata只能为待编码文本序列的一维数据")
        ydata = ydata.flatten()

        # 获取词汇和下标的互相对应关系
        if word_to_idx is None and idx_to_word is None:
            self.idx_to_word, self.word_to_idx = self.get_vocab(
                xdata, vocab_size=max_vocab_size, use_unk=True, use_pad=True)
            # <pad>和<unk>标记的下标
            self.pad_idx = self.word_to_idx['<pad>']
            self.unk_idx = self.word_to_idx['<unk>']
        elif word_to_idx is None and idx_to_word is not None:
            self.idx_to_word = idx_to_word
            self.word_to_idx = dict()
            for i, word in enumerate(idx_to_word):
                self.word_to_idx[word] = i
        elif word_to_idx is not None and idx_to_word is None:
            itm = list(word_to_idx.items())
            itm = sorted(itm, key=lambda w: w[1])
            self.idx_to_word = [w for (w, i) in itm]
            self.word_to_idx = word_to_idx
        else:
            self.idx_to_word = idx_to_word
            self.word_to_idx = word_to_idx

        # label的词典
        if encode_label:
            self.label_to_idx, self.idx_to_label = self.get_vocab(
                ydata, use_unk=False, use_pad=False)
        else:
            self.label_to_idx, self.idx_to_label = self.get_vocab(
                ydata, use_unk=False, use_pad=False)

        # 对较短的句子进行补全
        self.max_length = max([len(x) for x in xdata])
        self.encoded_data = np.zeros((xdata.shape[0], self.max_length),
                                     dtype=int)

        # 对label编码
        if encode_label:
            self.encoded_label = np.zeros(len(ydata))
            for i in range(ydata.shape[0]):
                self.encoded_label[i] = self.label_to_idx[ydata[i]]
        else:
            self.encoded_label = ydata

        # 对数据编码
        for i in range(xdata.shape[0]):
            xlen = len(xdata[i])
            for j in range(self.max_length):
                self.encoded_data[i][j] = self.word_to_idx[
                    "<pad>"] if j >= xlen else self.word_to_idx[xdata[i][j]]

    def __len__(self):
        '''
        Desc：
            返回数据集样本总数
        '''
        return self.encoded_label.shape[0]

    def __getitem__(self, idx):
        '''
        Desc：
            返回idx对应的序列编码结果
        Args:
            idx -- 检索的序列下标
        Returns：
            encoded_data[idx], encoded_label[idx] -- 对应编码数据和label
        '''
        return self.encoded_data[idx], self.encoded_label[idx]

    def get_vocab(self, data, vocab_size=None, use_unk=True, use_pad=False):
        '''
        Desc：
            对data进行编码，返回word_to_idx和idx_to_word
        Args：
            data: ndarray  --  所有的文本数据，可以是任意维度
            vocab_size: int  --  最大的字典大小，按照词的出现频率排序，剩下的低频词表示为<unk>
            use_unk: bool  --  是否使用unk标签表示其余的词
            use_pad: bool  --  是否使用pad标签
        Returns：
            word_to_idx: dict  --  词到下标的对应词典
            idx_to_word: list  --  下标到词的对应列表
        '''
        data = np.array(data).flatten()  # 变为1维向量
        # 异常处理：如果没有设置最大字典大小，则设为全部
        if vocab_size is None:
            vocab_size = len(set(data))
        self.vocab = dict(Counter(data).most_common(vocab_size))

        # 添加<unk>和<pad>标签
        if use_unk:
            self.vocab["<unk>"] = len(data) - np.sum(list(self.vocab.values()))
        if use_pad:
            self.vocab["<pad>"] = 0

        # 获得词和下标的对应关系并返回
        idx_to_word = [word for word in self.vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        return idx_to_word, word_to_idx


class Word2VecDataset(tud.Dataset):
    def __init__(self,
                 text,
                 word_to_idx,
                 idx_to_word,
                 word_freqs,
                 C=1,
                 K=1,
                 MOTIF=1):
        '''
        Desc：
            word2vec模型数据集构建函数
        Args：
            text   --  待编码文本
            word_to_idx  --  词汇到下标的转换词典
            idx_to_word  --  下标到词汇的转换列表
            word_freqs  --  词频
            C  --  窗口大小，前后C个碱基，共2C个
            K  --  每个正样本采样K个负样本，一共2C*K个
            MOTIF  --  使用1-MOTIF长度
        '''
        super(Word2VecDataset, self).__init__()
        self.text_encoded = sirna_util.get_seq_motif(text, motif=MOTIF)[0]
        self.text_encoded = torch.Tensor(self.text_encoded).long()

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.C = C
        self.K = K

        self.word_freqs = torch.Tensor(word_freqs)
        self.center_sample, self.pos_sample, self.neg_sample = self.get_sample(
            self.text_encoded)

    def __len__(self):
        '''
        Desc：
            这里返回正样本个数，每个序列有78=17*4+2*5
        '''
        return len(self.pos_sample)

    def __getitem__(self, idx):
        '''
        Desc：
            返回中心词、相邻词、不相邻词q
        Args：
            idx  --  查询的词汇索引
        Returns：
            self.center_sample[idx], self.pos_sample[idx], self.neg_sample
        '''
        return self.center_sample[idx], self.pos_sample[idx], self.neg_sample[
            idx]

    def encode_split_text(self, text):
        '''
        Desc：
            编码并分割一整句文本
        '''
        res = []
        for seq in text:
            res.append([word_to_idx.get(word) for word in seq])
        return res

    def get_sample(self, seqs_encoded):
        '''
        Desc：
            返回seq序列的所有正样本，及对应选出的负样本
        '''
        center_sample = []
        pos_sample = []
        neg_sample = []
        seq_len = len(seqs_encoded[0])

        for seq in seqs_encoded:  # seq 句子
            for i, s in enumerate(seq):  # s 词
                pos_idx = list(range(max(0, i - self.C), i)) + list(
                    range(i + 1, min(seq_len, i + self.C + 1)))
                pos_word = seq[pos_idx]  #tensor
                while len(pos_word) < 2 * self.C:  # 对缺少的数据填充
                    pos_word = torch.cat((pos_word, pos_word))[:2 * self.C]

                neg_word = torch.multinomial(self.word_freqs,
                                             self.K * pos_word.shape[0],
                                             True)  # 负采样
                while len(neg_word) < 2 * self.C * self.K:  # 对缺少的数据填充
                    neg_word = torch.cat(
                        (neg_word, neg_word))[:2 * self.C * self.K]

                pos_sample.append(pos_word)
                neg_sample.append(neg_word)
                center_sample.append(s)
        return center_sample, pos_sample, neg_sample


class SeqDataset(tud.Dataset):
    def __init__(self, seqs, seq_freq, base_to_idx, idx_to_base):
        '''
        Desc：
            将序列数据转换为dataset保存，只保存1模motif数据
        Args：
            seqs: ndarray(N,  ) -- 所有的序列数据，字符串
            seq_freq: ndarray -- 所有序列对应的沉默效率
            base_to_idx: dict -- base to idx
            idx_to_base: list -- idx to base
            motif: int -- 最大motif大小
            seq_encoded: list(torch.LongTensor)) -- 编码后的seq，list里面分别是各个motif的编码
            seq_freq: torch.DoubleTensor
        '''
        super(SeqDataset, self).__init__()
        self.seq_encoded = torch.empty((seqs.shape[0], len(seqs[0])),
                                       dtype=torch.int).long()
        for i in range(seqs.shape[0]):
            for j, bp in enumerate(seqs[i]):
                self.seq_encoded[i][j] = base_to_idx[bp]
        self.base_to_idx = base_to_idx
        self.idx_to_base = idx_to_base
        self.motif = motif
        self.seq_freq = torch.FloatTensor(seq_freq)

    def __len__(self):
        '''
        Desc：
            返回数据集样本总数
        '''
        return self.seq_freq.shape[0]

    def __getitem__(self, idx):
        '''
        Desc：
            返回idx对应的序列信息，包括所有motif的对应的下标
        Args:
            idx -- 检索的序列下标
        Returns：
            seq_encoded[idx], seq_freq[idx] -- 对应字符串和频率
        '''
        return self.seq_encoded[idx], self.seq_freq[idx]
