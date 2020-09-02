#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  file_util.py
 * @Time    :  2020/04/03 22:03:47
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  文件相关构件库
'''
import pandas as pd
import numpy as np
import sys
import os


def get_data_from_file(fpath,
                       xname=None,
                       yname=None,
                       upper=False,
                       dropna=True,
                       encode='utf-8'):
    '''
    Desc：
        使用于大部分情况的Excel文件数据提取，从csv/excel(xls,xlsx,xlsm..)/txt格式的文件中提取训练数据和可能存在的label
    Args：
        fpath: str  --  文件路径
        xname: list or str  --  数据列名，如果为None，则为全部
        yname: list or str  --  标签列名，如果为None，则视为无，如果xname为None，yname不为None，yname依然生效
        upper: bool  --  是否对所有的数据进行大写转换
        dropna: bool  --  是否丢弃缺失值所在行
        encode: str  --  读取后的编码方式，默认utf-8
    Returns：
        x, y: ndarray(-1,1)  --  训练数据和对应标签
    '''
    path = sys.path[0]
    fpath = os.path.join(path, fpath)
    print("read fpath:", fpath)
    file_type = fpath.split('.')[-1]
    raw_data = pd.DataFrame()

    # 判断文件类型
    try:
        if file_type == 'csv':
            raw_data = pd.read_csv(fpath, encoding=encode)
        elif file_type == 'txt':
            raw_data = pd.read_csv(fpath, sep=' ', encoding=encode)
        elif file_type in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf']:
            raw_data = pd.read_excel(fpath)
    except Exception as e:
        raise e

    # 对要读取的数据内容进行分析
    data = pd.DataFrame()
    if xname is None:
        xname = raw_data.columns.values.tolist()
    else:
        xname = [xname] if type(xname) != list else xname
    if yname is None:
        yname = []
    else:
        yname = [yname] if type(yname) != list else yname

    # 获取要读取的数据
    data = raw_data[xname + yname]  # DataFrame
    data = data.dropna(axis=0) if dropna else data

    # 区分X和Y并返回
    # x, y = data[xname].values.squeeze(), data[yname].values.squeeze()  # DataFrame
    x, y = data[xname], data[yname]  # DataFrame
    if upper:
        for i in x.columns:
            if type(x.loc[0, i]) is str:
                x.loc[:, i] = x.loc[:, i].str.upper()
                # x[i] = x[i].str.upper()
    x, y = x.values.squeeze(), y.values.squeeze()
    return x, y


def write_csv_excel(data,
                    fpath,
                    columns=None,
                    header=False,
                    sheet_name=None,
                    nan_rep='NULL',
                    encoding=None):
    '''
    Desc：
        将序列数据写入csv文件，默认不写入DataFrame的index和header
    Args：
        data:DataFrame/ndarray/list -- ndarray格式数据
        fpath -- 写入文件路径或文件流，文件类型可以是csv，xlsx，txt
        columns -- 可选的列
        header -- 是否要写入列名
        sheet_name -- 在写入excel时可选，指定sheet名
        nan_rep -- 是否要将Nan替换成其他字符串
    Returns：
        None -- None
    '''
    path = sys.path[0]
    fpath = os.path.join(path, fpath)
    print("write fpath:", fpath)

    if type(data) not in [pd.DataFrame, np.ndarray, pd.Series, list]:
        raise ValueError("data数据类型只支持DataFrame, Series, ndarray和list")

    data = pd.DataFrame(data)
    file_type = fpath.split('.')[-1]

    if file_type == 'csv':
        data.to_csv(fpath,
                    columns=columns,
                    index=False,
                    header=header,
                    na_rep=nan_rep,
                    encoding=encoding)
    elif file_type in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf']:
        writer = pd.ExcelWriter(fpath)
        data.to_excel(writer,
                      sheet_name=sheet_name,
                      na_rep=nan_rep,
                      columns=columns,
                      header=header,
                      index=False,
                      encoding=encoding)
    elif file_type == 'txt':
        data.to_csv(fpath,
                    sep=' ',
                    columns=columns,
                    index=False,
                    header=header,
                    na_rep=nan_rep,
                    encoding=encoding)
    else:
        raise ValueError("写入文件只支持csv, txt, xls, xlsx, xlsm, xlsb, odf")
