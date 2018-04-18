import numpy as np


class ODGrowthFactor:
    '''
    创建一个增长率预测模型

    Parameters:
        g_func: 增长率函数
        error: 收敛检验误差
        t_round: OD矩阵有效位数
        od_round: O&D向量有效位数
    Returns:
        一个增长率预测模型
    '''

    AVG = 'f_Average'
    DET = 'f_Detroit'
    FRA = 'f_Frater'
    FUR = 'f_Furness'

    def __init__(self, g_func=AVG, error=0.01, t_round=3, od_round=4):
        self.error = error
        self.g_func = self.__getattribute__(g_func)
        self.od_round = od_round
        self.t_round = t_round

    def fit(self, Tr):
        '''
        设置现状年OD表

        Parameters:
            Tr: 现状年的OD矩阵，不含sum项
        '''
        self.Tr = Tr
        self.Or = Tr.sum(axis=1)
        self.Dr = Tr.sum(axis=0)

    def predict(self, Of, Df):
        '''
        预测规划年出行分布量

        Parameters:
            Of: 规划年产生量向量
            Df: 规划吸引量向量

        Returns:
            array: 规划年OD表，含sum项
        '''
        self.iter_count = 0
        # 迭代计算
        Tret = self.iter_main(self.Tr, Of, Df)
        return Tret

    def iter_main(self, Tb, Of, Df):
        '''
        迭代计算

        Parameters:
            Tb: 迭代OD矩阵
            Of: 规划年产生量向量
            Df: 规划年出行量向量
        Returns:
            array: 规划年OD矩阵
        '''
        self.iter_count += 1

        # 计算出行量增长率
        f = np.round(self.g_func(Tb, Of, Df), self.od_round)

        # 计算规划年OD
        Tp = np.round(Tb * f, self.t_round)

        # 是否需要迭代
        if self.should_iter(Tp,  Of, Df):
            return self.iter_main(Tp, Of, Df)
        else:
            return Tp

    def should_iter(self, Tp, Of, Df):
        '''
        判断是否需要继续迭代

        Parameters:
            Tp: 预测OD矩阵
            Of: 规划年产生量向量
            Df: 规划年吸引量向量
            error: 收敛误差,default=0.01
        Returns:
            Boolean: 如果需要迭代返回True，反之Flase
        '''
        # 计算A&G增长率
        Fop = Of / Tp.sum(axis=1)
        Fdp = Df / Tp.sum(axis=0)

        return (np.abs((1-Fop)) >
                self.error).any() or (np.abs((1-Fdp)) > self.error).any()

    def f_Average(self, Tb, Of, Df):
        '''
        平均增长率法-增长函数

        Parameters:
            Of: 规划年产生量向量
            Df: 规划年吸引量向量
        Returns:
            array: 增量率矩阵
        '''
        # 计算A&G增长率
        Fd, Fo = np.meshgrid(Df / Tb.sum(axis=0), Of / Tb.sum(axis=1))
        ret = (Fd + Fo) / 2
        return ret

    def f_Detroit(self, Tb, Of, Df):
        '''
        底特律法-增长函数

        Parameters:
            Tb: 迭代OD矩阵
            Of: 规划年产生量向量
            Df: 规划年吸引量向量
        Returns:
            array: 增长率矩阵
        '''
        Fo = Of / Tb.sum(axis=1)
        Fd = Df / Tb.sum(axis=0)
        FD = Fo.reshape(-1, 1) * Fd / (Df.sum() / Tb.sum(axis=0).sum())
        return FD

    def f_Frater(self, Tb, Of, Df):
        '''
        佛莱特法-增长函数

        Parameters:
            Tb: 迭代OD矩阵
            Of: 规划年产生量向量
            Df: 规划年吸引量向量
        Returns:
            array: 增长率矩阵
        '''
        Ob = Tb.sum(axis=1)
        Db = Tb.sum(axis=0)
        Fo = Of / Ob
        Fd = Df / Db
        Li = Ob / (Tb * Fd).sum(axis=1)
        Lj = Db / (Tb * Fo).sum(axis=0)
        return Fo.reshape(-1, 1) * Fd * (Li.reshape(-1, 1) + Lj) / 2

    def f_Furness(self, Tb, Of, Df):
        '''
        弗尼斯-增长函数

        Parameters:
            Tb: 迭代OD矩阵
            Of: 规划年产生量向量
            Df: 规划年吸引量向量
        Returns:
            array: 增长率矩阵
        '''
        if self.iter_count % 2 == 1:
            return (Of / Tb.sum(axis=1)).reshape(-1, 1)
        else:
            return Df / Tb.sum(axis=0)

    @staticmethod
    def sum(T):
        '''
        对OD矩阵列与行求和

        Parameters:
            OD: OD矩阵
        Returns:
            求和后的矩阵
        '''
        T = np.hstack((T, T.sum(axis=1).reshape(-1, 1)))
        return np.vstack((T, T.sum(axis=0)))


#######################
#  渣渣课本 例题数据
#######################
ZZ = {
    'Tn': np.array([
        [150, 100, 50],
        [400, 100, 200]
    ]),
    'Rn': np.array([
        [3, 2, 5],
        [3, 5, 4]
    ])
}

########################
#  邵春福 PPT 例题数据
########################
CF = {
    'Tn': np.array([
        [17, 7, 4],
        [7, 38, 6],
        [4, 5, 17]
    ]),
    'Rn': np.array([
        [7, 17, 22],
        [17, 15, 23],
        [22, 23, 7]
    ]),
    'Rf': np.array([
        [4, 9, 11],
        [9, 8, 12],
        [11, 12, 4]
    ]),
    'Of': np.array([38.6, 91.9, 36.0]),
    'Df': np.array([39.3, 90.3, 36.9])
}


if __name__ == '__main__':
    model = ODGrowthFactor(g_func=ODGrowthFactor.FUR, error=0.03)
    model.fit(CF['Tn'])
    ret = model.predict(CF['Of'], CF['Df'])
    print('迭代结果为：\n', model.sum(ret))
    print('共迭代 {} 次'.format(model.iter_count))
