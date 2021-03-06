#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  sirna.py
 * @Time    :  2020/04/03 15:34:26
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  包含一些siRNA处理相关的函数库
'''

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_util

from tqdm import tqdm
from Bio import Entrez
from Bio import SeqIO

def get_idx_base(motif=1, padding=False):
    '''
    Desc：
        获取siRNA的motif和对应编码的下标，用于词向量编码，每次调用参数相同，则结果相同
    Args：
        motif(int/list/ndarray) -- 1，2...，可以是单个的motif，也可以是一个列表一起编码
        padding(bool) -- 是否使用padding，设置为P，index为0
    Returns：
        idx_to_base(list), base_to_idx(dict) -- 词和下标的相互转换
    '''
    # 异常处理
    if type(motif) not in [int, list, np.ndarray]:
        raise TypeError("motif类型必须为整形、list或ndarray")
    if type(motif) in [list, np.ndarray]:
        if min(motif) < 1:
            raise ValueError("motif数值必须为正整数")
        max_motif = max(motif)
    else:
        max_motif = motif

    # 碱基词典
    vocab = ['A', 'G', 'U', 'C', 'T']
    res = []
    tmp_res = vocab.copy()
    single_motif = [vocab.copy()]

    # 对多维motif进行处理
    while (max_motif > 1):
        max_motif -= 1
        tmp = []
        for w in vocab:
            for x in tmp_res:
                tmp.append(w + x)
        tmp_res = tmp.copy()
        single_motif.append(tmp_res)

    # 计算motif和对应下标的关系并返回
    if type(motif) in [list, np.ndarray]:
        for m in motif:
            res.extend(single_motif[m - 1])
        if padding:
            res.append('<pad>')
        idx_to_base = [base for base in res]
        base_to_idx = {base: i for i, base in enumerate(idx_to_base)}
    else:
        res = single_motif[motif - 1].copy()
        if padding:
            res.append('<pad>')
        idx_to_base = [base for base in res]
        base_to_idx = {base: i for i, base in enumerate(idx_to_base)}
    return idx_to_base, base_to_idx


def idx_to_seq(seqs, motif=1):
    '''
    Desc：
        将idx形式的序列转化为字符串
    Args：
        seqs: tensor(batch_size, seq_size) -- 序列的idx编码表示
        motif: int/list/ndarray -- 最大的motif大小，默认为1
    Returns：
        res: ndarray(batch_size, ) -- 列表形式的字符串序列
    '''
    import torch

    # 异常处理
    if type(motif) not in [int, list, np.ndarray]:
        raise TypeError("motif类型必须为整形、list或ndarray")
    if type(motif) in [list, np.ndarray]:
        if min(motif) < 1:
            raise ValueError("motif数值必须为正整数")
    if type(seqs) == list:
        seqs = np.array(seqs)
    if type(seqs) not in [list, np.ndarray, torch.tensor, torch.Tensor]:
        print(type(seqs))
        raise TypeError("seqs 类型只支持list, ndarray和tensor")

    # 把维度扩充为2维
    if len(seqs.shape) == 1:
        if type(seqs) == np.ndarray:
            seqs = np.expand_dims(seqs, axis=0)
        elif type(seqs) == torch.tensor:
            seqs = seqs.unsqueeze(0)
    elif len(seqs.shape) != 2:
        raise ValueError("seqs 只能为1维数据或二维数据")

    # 获取motif和下标对应关系
    res = np.empty((seqs.shape[0], ), dtype=object)
    idx_to_base, _ = get_idx_base(motif)

    # 对序列字符串进行拼接
    for i, seq in enumerate(seqs):
        res[i] = ''.join([idx_to_base[i] for i in seq])
    return res


def get_seq_motif(seqs, motif=1):
    '''
    Desc：
        获取各个序列所有的motif，并返回各个motif对应的idx
    Args：
        seqs: list/ndarray_object(batch_size, ) -- 输入的字符串序列，可以是batch_size个也可以是单个
        motif: int/list/ndarray -- 单个的motif大小或motif的列表
    Returns：
        res: list(ndarray) -- seq各个motif的idx
    '''
    # 异常处理
    if type(motif) not in [int, list, np.ndarray]:
        raise TypeError("motif类型必须为整形、list或ndarray")
    if type(motif) in [list, np.ndarray]:
        if min(motif) < 1:
            raise ValueError("motif数值必须为正整数")
    if type(seqs) == str:
        seqs = [seqs]
    if type(seqs) == list:
        seqs = np.array(seqs)

    # 保证全部大写
    for i, s in enumerate(seqs):
        seqs[i] = seqs[i].upper()

    res = []
    _, base_to_idx = get_idx_base(motif=motif)

    # 从小到大循环得到所有的motif
    if type(motif) in [list, np.ndarray]:
        nlist = np.array(motif) - 1
    elif type(motif) == int:
        nlist = [motif - 1]
    for n in nlist:
        seq_motif = np.empty((seqs.shape[0], len(seqs[0]) - n), dtype=int)
        for i in range(seqs.shape[0]):
            for j in range(len(seqs[0]) - n):
                if seqs[i][j:j + n + 1] == '':
                    raise ValueError("各序列长度需要一致")
                seq_motif[i][j] = base_to_idx[seqs[i][j:j + n + 1]]
        res.append(seq_motif)
    return res


def filter_sirna(data=None):
    '''
    Desc：
        从data中获取只包含A/a, G/g, U/u, C/c, T/t的21bp的siRNA序列数据，
    Args：
        data:DataFrame/ndarray/list  --  待处理数据
    Returns：
        sirna:ndarray  --  提取出的siRNA数据
    '''
    # 异常处理
    if data is None:
        raise ValueError("data不可为空")
    if type(data) not in [pd.DataFrame, np.ndarray, list]:
        raise ValueError("data只允许DataFrame，ndarray和list三种类型")

    # 统一成DataFrame
    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)
    # 从DataFrame变成Series
    data = data.iloc[:, 0]
    # 所有序列大写
    data = data.str.upper()

    # 去掉所有空格,5'和3',r-和d-等无用字符
    chr_list = [
        ' ', '5', '3', "'", "’", "′", "`", "[", "]", "r", "D", "d",
        chr(65313),
        chr(65319),
        chr(65333), "(", ")", "-", "–", '"', 'N', 'n', 'v', 'V'
    ]
    data = data_util.char_remove(data, chr_list)

    seq_len = data.str.len()
    sirna = data[seq_len == 21]
    return sirna.values


def rna_pair_and_reverse(seqs, reverse=True):
    '''pair rna sequences and reverse sequences

    Args：
    seq: [list, ndarray] -- input sequences
    Returns：
    res: [list, ndarray] -- output sequence
    '''
    map = {'A': 'U', 'G': 'C', 'C': 'G', 'U': 'A'}
    res = seqs.copy()
    for idx, seq in enumerate(seqs):
        seq = seq.upper()
        tmp = "".join([map[s] for s in seq])
        if reverse:
            res[idx] = tmp[::-1]
        else:
            res[idx] = tmp
    return res


def antisense_to_sense_cdna(seqs):
    '''transform antisense sequences to cdna sequences

    Args：
      seqs: [list, ndarray] -- antisense sequences
    Returns：
      res: [list, ndarray] -- cdna sequences
    '''
    map = {'A': 'T', 'G': 'C', 'C': 'G', 'U': 'A'}
    res = seqs.copy()
    for idx, seq in enumerate(seqs):
        seq = seq.upper()
        tmp = "".join([map[s] for s in seq])
        res[idx] = tmp[::-1]
    return res


def cdna_to_antisense(seqs):
    '''transform cdna to antisense strand sequences

    Args：
      seqs: [list, ndarray] -- cdna sequences
    Returns：
      res: [list, ndarray] -- antisense strand sequences
    '''
    map = {'T': 'A', 'G': 'C', 'C': 'G', 'A': 'U'}
    res = seqs.copy()
    for idx, seq in enumerate(seqs):
        seq = seq.upper()
        tmp = "".join([map[s] for s in seq])
        res[idx] = tmp[::-1]
    return res


def get_target_pos(sirna, cdnas):
    '''get sirna target index in cdna for each sequences

    Args：
      seqs: [list, ndarray] -- antisense sequences
      cdna: [list, ndarray] -- cdna sequences
    Returns：
      res: [list, ndarray] -- position idx list
    '''
    if len(sirna) != len(cdnas):
        raise ValueError("The shape of sirna and cdna should be the same")
    res = []
    target = antisense_to_sense_cdna(sirna)
    for idx, tar in enumerate(target):
        cur_cdna = cdnas[idx]
        res.append(cur_cdna.find(tar))
    return res


def entrez_fetch_seq(id, batch_size=10, temporary_save=False, save_format='fasta-2line'):
    '''Fetch Entrez results in batch and saved in Bio.records format. If temporary_save is True, results will be saved in ./temp_sequences.fasta for temporary use.
    '''
    res = []
    for idx in tqdm(range(0, len(id), batch_size)):
        seq_records = []
        record_ids = id[idx:idx + batch_size]

        result_handle = Entrez.efetch(db="nucleotide", rettype="gb", id=record_ids)
        seqRecord = SeqIO.parse(result_handle, format='gb')

        # get records from GeneBankIterator
        for idx, record in enumerate(seqRecord):
            seq_records.append(record)

        if temporary_save:
            with open('temp_sequences.fasta', 'a') as fout:
                SeqIO.write(seq_records, fout, save_format)
        res.extend(seq_records)
    return res

if __name__ == "__main__":
    seq = ["CAAAAUUAUCCACUGUUUUUG"]
    cdna = ["CTTCCTTGTTTGGTCTGCTGTGGATCTGCCTTATTGCATATGCCATGCATCAGATAATGGATGCATCAGATAATGGTGTTAGACAAAGCTTCATTGTGAACAACCTAATGCATTTTAGAGAAACAATCTCATCACATTTTTTCTAGCCTTTCCTACATTTAAACTTGCTGTTGCCCAAATTATAATTTTTTAAATGTCTTTGGTGGGCTTCTGTTAATTCACATGACTTGAGCTTATAGCTATGTCTACTGCACAGATTGGGTAATGGAACACTAAACTTTTATACTTGAAAATGACAGCCTTAAATGCTCATATCAGTCACAAATCTAGGATGTACTGTCTTGTTGTATGTGAGCTTTGTAGAGATTTTTAAAAATATAAGCATCACCTTCCCATTGAAGAGTGGAGAGAGTCTACTGGATGACTGGCCAGGAACTTTCTCTCTGAATCGGACATTTGGATGTCTTCTTTCTTCCAAGAAATGGTGGTTCACATTAAAGTATCATGGCCTTATGTATGCTCAAATGGAATCTTATGTAACTTTCTTATTTAATTTTGGTCTGCTTATTTTTAGATAAAATTGAAAGGAATTGTATAAATCAATTAACATATTAGCTGAGTTG"]
    # mrna = "".join(['U' if x == 'T' else x for x in cdna])
    # target_rna = rna_pair_and_reverse(seq)
    # print(cdna_to_antisense(seq))
    print(antisense_to_sense_cdna(seq))
    # print(get_target_pos(seq, cdna)[0])
    # print(cdna[0][596:596+19] == antisense_to_sense_cdna(seq)[0])