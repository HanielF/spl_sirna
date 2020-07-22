#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  train_evaluate.py
 * @Time    :  2020/03/02 00:27:45
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  包含train和evaluate函数
'''
import scipy
import torch
import numpy as np
import torch.utils.data as tud

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

def train(model, iterator, optimizer, criterion, device='cpu'):
    '''
    Desc：
        用于训练模型的函数，返回训练的平均epoch误差
    Args：
        model: torch.nn.model  --  待训练的模型
        iterator  --  包含x和y的dataloader
        optimizer  --  优化器
        criterion  --  损失函数
        device  --  指定的设备，可以是'cpu'或'cuda'
    Returns：
        loss: float  --  每个epoch的平均损失
    '''
    # 设置model的状态为train
    model.train()
    epoch_loss = 0

    # x: tensor[batch_size, seq_size]，每个样本的词的idx，y: 对应label
    for i, (x, y) in enumerate(iterator):
        x, y = x.to(DEVICE), y.to(DEVICE)
        # 在BP之前需要zero_grad，将之前的梯度置零
        optimizer.zero_grad()
        predictions = model(x).view(-1, 1)
        # 计算loss并且backward
        loss = criterion(predictions.flatten().to(DEVICE), y.to(DEVICE))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device='cpu', pcc=False, acc=False):
    '''
    Desc：
        测试模型的函数，返回验证集的平均epoch误差
    Args：
        model: torch.nn.model  --  带验证的模型
        iterator: dataloader  --  用于验证模型的数据集
        criterion  --  损失函数
        device  --  运行的设备，可以是'cpu'或'cuda'
    Returns：
        res  --  保存loss和可选的pcc，acc，调用时，使用loss、pcc和acc分别赋值
    '''
    # 设置model状态为evaluate
    model.eval()
    epoch_loss = 0.
    epoch_pcc = 0.
    epoch_acc = 0.
    # 验证时不用计算梯度，也不用BP
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x, y = x.to(DEVICE), y.to(DEVICE)
            predictions = model(x)

            # 计算Loss
            loss = criterion(predictions.flatten().to(DEVICE), y.to(DEVICE))
            epoch_loss += loss.item()

            # 计算相关系数PCC
            tmp_r, _ = scipy.stats.pearsonr(y.view(-1).cpu().numpy(), predictions.view(-1).cpu().numpy())
            epoch_pcc += tmp_r

            # 计算准确度ACC
            tmp_acc = np.sum(predictions.view(-1).cpu().numpy() == y.view(-1).cpu().numpy())*1.0/predictions.shape[0]
            epoch_acc += tmp_acc

    # 返回对应数据
    res = [epoch_loss / len(iterator)]
    if pcc:
        res.append(epoch_pcc / len(iterator))
    if acc:
        res.append(epoch_acc / len(iterator))
    return res


def predict_samples(model, data=None):
    '''
    Desc：
        使用model预测某几个样本的结果
    Args：
        model -- 训练好的model
        data: list or tud.dataloader -- 保存待预测数据，这些序列需要已经被分词好
                list: 保存的是对应下标的encoded序列，只返回预测值
                tud.dataloader： 保存测试数据和对应的label，__getitem__方法中应该返回一个样本和对应的label，返回预测值和真实值
        base_to_idx: dict  --  词到下标的对应关系字典
    Returns：
        pre: ndarray(n, ) -- 预测出的结果
        freq: ndarray(n, ) -- 真实的label，只有在data类型为dataloader时才返回
    '''
    # 异常处理
    if data is None:
        raise ValueError("data为分割好的待预测的文本序列，不可为None")

    # 设置模型为evaluate
    model.eval()
    # 对list和ndarray的处理
    lst = [list, np.ndarray]
    if type(data) in lst:
        res_x = torch.tensor(np.array(data)).long().to(DEVICE)
        res_freq = np.empty(0)

        # batch_size设置为64，分batch预测
        for i in range(0, res_x.shape[0], 64):
            batch_x = res_x[i:i + 64, :]
            batch_pre = model(batch_x)
            if batch_pre.is_cuda:
                batch_pre = batch_pre.cpu()
            batch_pre = batch_pre.data.numpy().flatten()
            res_freq = np.concatenate((res_freq, batch_pre))
        return res_freq
    # 对DataLoader的处理
    elif type(data) == tud.DataLoader:
        pre = freq = np.empty(0)
        for i, (x, y) in enumerate(data):
            prediction = model(x)
            if prediction.is_cuda:
                prediction = prediction.cpu()
            prediction = prediction.data.numpy().flatten()
            y = y.numpy().flatten()
            pre = np.concatenate((pre, prediction))
            freq = np.concatenate((freq, y))
        return pre, freq