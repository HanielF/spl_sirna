#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  plot_util.py
 * @Time    :  2020/04/03 17:01:42
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  包含绘图方面的构件
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def make_scatter(xdata,
                 ydata,
                 xlabel=None,
                 ylabel=None,
                 xtick=None,
                 ytick=None,
                 title=None,
                 error=False,
                 pcc=False,
                 yxline=False,
                 filename=None,
                 show=False):
    '''
    Desc：
        在回归任务或分类任务中，绘制预测结果和真实值的散点图，并且保存在本地
    Args：
        xdata: ndarray  --  真实值
        ydata: ndarray  --  预测值
        xlabel: str  --  x轴的标签
        ylabel: str  --  y轴的标签
        xtick: list/ndarray  --  x轴的下标
        ytick: list/ndarray  --  y轴的下标
        title: str  --  title内容
        error: Bool  --  是否计算xdata和ydata的误差，包含平均误差，最大和最小误差
        pcc: Bool  --  是否计算xdata和ydata的相关系数
        yxline: Bool  --  是否绘制y=x参考线
        filename: str  --  保存在本地的路径
    '''
    # 异常处理
    if type(xdata) is not list and type(xdata) is not np.ndarray:
        raise TypeError("xdata格式需要为list或np.ndarray")
    if type(ydata) is not list and type(ydata) is not np.ndarray:
        raise TypeError("ydata格式需要为list或np.ndarray")
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if xlabel is not None and type(xlabel) is not str:
        xlabel = str(xlabel)
    if ylabel is not None and type(ylabel) is not str:
        ylabel = str(ylabel)

    if title is not None and type(title) is not str:
        title = str(title)

    if xtick is not None and type(xtick) is not list and type(
            xtick) is not np.ndarray:
        raise TypeError("xtick格式应该为list或numpy.ndarray")
    if ytick is not None and type(ytick) is not list and type(
            ytick) is not np.ndarray:
        raise TypeError("ytick格式应该为list或numpy.ndarray")

    # 设置显示文字样式
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10,
    }
    # 绘制主要部分
    plt.figure(figsize=[6, 6])
    plt.scatter(x=xdata, y=ydata, marker='.', c='black', s=16)

    # 绘制xticks和yticks
    xmax, xmin = np.max(xdata), np.min(xdata)
    ymax, ymin = np.max(ydata), np.min(ydata)
    if xtick is None:
        plt.xticks(np.linspace(start=xmin, stop=xmax, num=11))
    else:
        plt.xticks(xtick)
    if ytick is None:
        plt.yticks(np.linspace(start=ymin, stop=ymax, num=11))
    else:
        plt.yticks(ytick)

    # 绘制y=x参考线
    if yxline:
        line_x, line_y = max(xmin, ymin), min(xmax, ymax)
        plt.plot([line_x, line_y], [line_x, line_y], 'g--', lw=1.5)
    # 计算误差
    if error:
        abs_diff = np.abs(ydata - xdata)
        avg_diff = np.sum(abs_diff) / xdata.size
        max_diff = np.max(abs_diff)
        min_diff = np.min(abs_diff)
        pos_x = xmin + (xmax - xmin) * 0.7
        plt.text(pos_x, ymin + (ymax - ymin) * 0.2,
                 'Average error: {:.3f}'.format(avg_diff), font2)
        plt.text(pos_x, ymin + (ymax - ymin) * 0.15,
                 'Max error: {:.3f}'.format(max_diff), font2)
        plt.text(pos_x, ymin + (ymax - ymin) * 0.1,
                 'Min error: {:.3f}'.format(min_diff), font2)
    # 计算相关系数
    if pcc:
        r, _ = pearsonr(xdata, ydata)
        plt.text(pos_x, ymin + (ymax - ymin) * 0.05, 'PCC: {:.3f}'.format(r),
                 font2)

    # 绘制label和title
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)
    plt.title(title, font1)

    # 保存和显示图
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def make_plot(data,
              labels=None,
              titles=None,
              filename=None,
              show=False,
              xtick_space=1):
    '''
    Desc：
        绘制折线图，可以在一个画布中绘制多张图，并可以保存在本地
    Args：
        data: list/ndarray  --  包含所有待绘制的数据
        labels: list/ndarray  --  包含所有子图中的label
        titles: list/ndarray  --  包含所有子图的标题
    Sample:
        data: [[valid_loss]], valid_loss=[1,2,3]
        labels: [['Valid_Loss']]
        titles: [['Loss of Valid Data']]

        data: [[train_epoch_loss, valid_epoch_loss], [valid_r]]
        labels: [['Train_Loss', 'Val_Loss'], ['Valid_R']]
        titles: ['Loss of Train and Valid Data', 'PCC of Valid Data']
    '''
    # 创建画布
    fig = plt.figure(figsize=[10, 6])
    fig.subplots_adjust(hspace=0.6)
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
    }
    # 对可能缺少的label和title进行补全
    labels = [''] * len(data) if labels is None else labels
    titles = [''] * len(data) if titles is None else titles
    if len(labels) < len(data):
        labels.extend([''] * (len(data) - len(labels)))
    if len(titles) < len(data):
        titles.extend([''] * (len(data) - len(titles)))

    # 迭代每个子图数据
    for i, fig_data in enumerate(data):
        label = labels[i]
        title = titles[i]

        ymax, ymin = np.max(fig_data), np.min(fig_data)
        xmax, xmin = len(fig_data[0]), 0

        if xtick_space == 1:
            tickNum = len(fig_data[0]) + 1
        else:
            tickNum = (len(fig_data[0]) -
                       len(fig_data[0]) % xtick_space) // xtick_space + 1
        xtick = np.linspace(xmin, xmax, tickNum, dtype=int).tolist()
        ytick = np.linspace(ymin, ymax, 11).tolist()

        # 创建子图
        ax = fig.add_subplot(len(data),
                             1,
                             i + 1,
                             xlim=(xmin, xmax),
                             ylim=(ymin, ymax),
                             xticks=xtick,
                             yticks=ytick)

        # 绘制每个折线图
        for j, plt_data in enumerate(fig_data):
            plt_label = label[j]
            x = np.arange(0, xmax, step=1)
            ax.plot(x, plt_data, label=plt_label, marker='.')
        ax.legend(loc='best', prop=font2)
        plt.title(title, font1)

    # 保存和显示图
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
