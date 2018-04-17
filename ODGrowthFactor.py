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

    def fit(self, od_now):
        '''
        设置现状年OD表

        Parameters:
            od_now: 现状年的OD矩阵，不含sum项
        '''
        self.od_now = od_now
        self.ori_now = od_now.sum(axis=1)
        self.dest_now = od_now.sum(axis=0)

    def predict(self, ori_fut, dest_fut):
        '''
        预测规划年出行分布量

        Parameters:
            ori_fut: 规划年产生量向量
            dest_fut: 规划吸引量向量

        Returns:
            array: 规划年OD表，含sum项
        '''
        # 初始化迭代变量
        self.iter_count = 0
        self.od_base = self.od_now
        self.ori_base = self.ori_now
        self.dest_base = self.dest_now

        # 迭代计算
        od = self.iter_main(ori_fut, dest_fut)

        # 计算A&G
        o_sum = np.atleast_2d(od.sum(axis=1)).T
        od_o = np.hstack((od, o_sum))
        d_sum = np.atleast_2d(od_o.sum(axis=0))
        od_od = np.vstack((od_o, d_sum))
        return od_od

    def iter_main(self, ori_fut, dest_fut):
        '''
        迭代计算

        Parameters:
            ori_fut: 规划年产生量向量
            dest_fut: 规划年出行量向量

        Returns:
            array: 规划年OD表，不含sum项
        '''
        self.iter_count += 1

        # 计算出行量增长率
        f = np.round(self.g_func(ori_fut, dest_fut), self.od_round)

        # 计算规划年OD
        od_predict = np.round(self.od_base * f, self.t_round)

        # 计算规划年A&G
        ori_predict = od_predict.sum(axis=1)
        dest_predict = od_predict.sum(axis=0)

        # 是否需要迭代
        if self.should_iter(ori_predict, dest_predict,  ori_fut, dest_fut):
            self.dest_base = dest_predict
            self.ori_base = ori_predict
            self.od_base = od_predict
            return self.iter_main(ori_fut, dest_fut)
        else:
            return od_predict

    def should_iter(self, ori_predict, dest_predict, ori_fut, dest_fut):
        '''
        判断是否需要继续迭代

        Parameters:
            ori_predict: 规划年产生量预测值向量
            dest_predict: 规划年吸引两预测值向量
            ori_fut: 规划年产生量向量
            dest_fut: 规划年吸引量向量
            error: 收敛误差,default=0.01
        Returns:
            Boolean: 如果需要迭代返回True，反之Flase
        '''
        # 计算A&G增长率
        F_ori_predict = ori_fut / ori_predict
        F_dest_predict = dest_fut / dest_predict

        return (np.abs((1-F_ori_predict)) >
                self.error).any() or (np.abs((1-F_dest_predict)) > self.error).any()

    def f_Average(self, ori_fut, dest_fut):
        '''
        平均增长率法-增长函数

        Parameters:
            ori_fut: 规划年产生量向量
            dest_fut: 规划年吸引量向量
        Returns:
            array: 增量率矩阵
        '''
        # 计算A&G增长率
        F_ori = ori_fut / self.ori_base
        F_dest = dest_fut / self.dest_base
        F_dest, F_ori = np.meshgrid(F_dest, F_ori)
        ret = (F_dest + F_ori) / 2
        return ret

    def f_Detroit(self, ori_fut, dest_fut):
        '''
        底特律法-增长函数

        Parameters:
            ori_fut: 规划年产生量向量
            dest_fut: 规划年吸引量向量
        Returns:
            array: 增长率矩阵
        '''
        F_ori = ori_fut / self.ori_base
        F_dest = dest_fut / self.dest_base
        F_dest_all = dest_fut.sum() / self.dest_base.sum()
        F_dest_ratio, F_ori = np.meshgrid(F_dest / F_dest_all, F_ori)
        ret = F_ori * F_dest_ratio
        return ret

    def f_Frater(self, ori_fut, dest_fut):
        '''
        佛莱特法-增长函数

        Parameters:
            ori_fut: 规划年产生量向量
            dest_fut: 规划年吸引量向量
        Returns:
            array: 增长率矩阵
        '''
        F_ori = ori_fut / self.ori_base
        F_dest = dest_fut / self.dest_base

        Li = self.ori_base / (self.od_base * F_dest).sum(axis=1)
        Lj = self.dest_base / (self.od_base * F_ori).sum(axis=0)

        Lj, Lj = np.meshgrid(Lj, Li)
        F_dest, F_ori = np.meshgrid(F_dest, F_ori)
        return F_ori * F_dest * ((Li + Lj)/2)

    def f_Furness(self, ori_fut, dest_fut):
        '''
        弗尼斯-增长函数

        Parameters:
            ori_fut: 规划年产生量向量
            dest_fut: 规划年吸引量向量
        Returns:
            array: 增长率矩阵
        '''
        if self.iter_count % 2 == 1:
            F_ori = np.tile(ori_fut / self.od_base.sum(axis=1),
                            (dest_fut.size, 1)).T
            return F_ori
        else:
            F_dest = np.tile(
                dest_fut / self.od_base.sum(axis=0), (ori_fut.size, 1))
            return F_dest

    def __repr__(self):
        return '<ODPredict object g_func:{} error:{} t_round:{}, od_round:{}>'.format(self.g_func.__name__, self.error, self.t_round, self.od_round)


####################################
#  渣渣课本 例题数据
####################################
ZZ_OD_NOW = np.array([
    [200, 100, 100],
    [150, 250, 200],
    [100, 150, 150],
])
ZZ_O_F = np.array([1000, 1000, 1250])
ZZ_D_F = np.array([1250, 900, 1100])

####################################
#  邵春福 PPT 例题数据
####################################
CF_OD_NOW = np.array([
    [17, 7, 4],
    [7, 38, 6],
    [4, 5, 17]
])
CF_O_F = np.array([38.6, 91.9, 36.0])
CF_D_F = np.array([39.3, 90.3, 36.9])


if __name__ == '__main__':
    model = ODGrowthFactor(g_func=ODGrowthFactor.AVG, error=0.01)
    model.fit(ZZ_OD_NOW)
    ret = model.predict(ZZ_O_F, ZZ_D_F)
    print('迭代结果为：\n', ret)
    print('共迭代 {} 次'.format(model.iter_count))
