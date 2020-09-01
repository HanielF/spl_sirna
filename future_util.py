def extend_sirna(data, origin, dis=10):
    '''
    Desc：
        将data中的siRNA分别向前后扩充dis个碱基，若siRNA在origin的开头或者结尾，则循环
    Args：
        data: ndarray/list  --  保存siRNA的数据
        origin: ndarray/list  --  保存mRNA数据，用于扩充siRNA
        dis: int  --  向左右分别扩充的数量
    Returns：
        res: ndarray  --  扩充后的siRNA数据
    '''
    pass