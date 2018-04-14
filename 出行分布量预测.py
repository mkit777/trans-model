import numpy as np


class ODPredict:
    '''
    出行量分布预测
    '''
    AVG = 'f_Average'
    DET = 'f_Detroit'
    FRA = 'f_Frater'

    def __init__(self, error=0.01, g_func=AVG):
        self.error = error
        self.g_func = self.__getattribute__(g_func)

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
        return np.round(od_od)

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
        # f = self.f_Average(ori_fut, dest_fut)
        # f = self.f_Detroit(ori_fut, dest_fut)
        # f = self.f_Frater(ori_fut, dest_fut)
        #f = self.f_Furness(ori_fut, dest_fut)
        f = self.g_func(ori_fut, dest_fut)
        # 计算规划年OD
        od_predict = self.od_base * f
        print(od_predict)
        print('第', self.iter_count, '次迭代结果：\n', od_predict)
        print('-'*20)

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
        print('第', self.iter_count, '次增长函数值：\n', ret)
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
        print('第', self.iter_count, '次增长函数值：\n', ret)
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

        print('第', self.iter_count, '次产生位置系数：\n', Li)
        print('第', self.iter_count, '次吸引位置系数：\n', Lj)

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
        u = np.mat(np.ones(ori_fut.size)).T
        d = np.mat(dest_fut).T
        o = np.mat(ori_fut).T
        t = np.mat(self.od_base)
        f = self.furness_iter(u, None, o, d, t)
        return f

    def furness_iter(self, u, v, o, d, t):
        '''
        待实现
        '''
        pass
        # v = np.mat(np.array(u) * np.array(t)).I * o
        # u_test = np.mat(np.array(v) * np.array(t).T).I * d
        # v_test = np.mat(np.array(u_test) * np.array(t)).I * o
        # u2 = np.hstack((u_test, u))
        # v2 = np.hstack((v_test, v))
        # u_s = np.sum(np.abs(u2 - u2.mean(axis=1))) / u.size / u2.mean(axis=1)
        # v_s = np.sum(np.abs(v2 - v2.mean(axis=1))) / v.size / v2.mean(axis=1)
        # if (u_s > 0.03).any() and (v_s > 0.03).any():
        #     return self.furness_iter(u, v_test, o, d, t)
        # else:
        #     u = np.array(u_test).T
        #     v = np.array(v_test).T
        #     u, v = np.meshgrid(v, u)
        #     return u*v


# 现状年origin-destination分布矩阵
ORI_DEST_NOW = np.array([
    [200, 100, 100],
    [150, 250, 200],
    [100, 150, 150],
])

# 规划年产生量向量
ORI_FUT = np.array([1000, 1000, 1250])

# 规划年吸引量向量
DEST_FUT = np.array([1250, 900, 1100])

if __name__ == '__main__':
    model = ODPredict()
    model.fit(ORI_DEST_NOW)
    ret = model.predict(ORI_FUT, DEST_FUT)
    print('\n迭代结果为：\n', ret)
    print('共迭代 {} 次'.format(model.iter_count))
