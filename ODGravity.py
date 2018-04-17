import numpy as np


class ODGravity:
    '''
    创建一个重力预测模型

    Parameters:
        model: 重力模型
        func_r: 阻抗函数
        error: 迭代误差
    Returns:
        一个重力预测模型
    '''
    UN = 'unrestraint'
    VH = 'voorhees'
    BPR = 'bpr'
    BC = 'bin_constraint'

    POWER = 'power'
    EXP = 'exp'

    def __init__(self, model=BC, func_r=POWER):
        # 设置重力模型
        self.model = self.__getattribute__(model)

        # 设置阻抗函数
        self.func_r = self.__getattribute__(func_r)

    def fit(self, OD, R):
        '''
        提供训练数据
        Parameters:
            OD: 现状年OD矩阵
            R: 现状年阻抗矩阵
        Returns:
            None
        '''
        # 求现状年 O & D
        self.Or = OD.sum(axis=1)
        self.Dr = OD.sum(axis=0)

        self.Tr = OD
        self.Rr = R
        # 训练模型
        self.model()

    def unrestraint(self):
        '''
        无约束重力模型法
        '''
        # 延长O&D
        Or = np.repeat(self.Or, self.Dr.size)
        Dr = np.tile(self.Dr, self.Or.size)       

        # 计算系数
        E = np.ones(self.Tr.size)
        lnOD = np.log(Or * Dr)
        if self.func_r.__name__ == 'power':
            Fr = np.log(self.Rr.ravel())
        else:
            Fr = self.Rr.ravel()

        # 构建系数矩阵
        A = np.vstack((E, lnOD, Fr)).T

        # 构建目标列向量
        Y = np.log(self.Tr.ravel())

        # least-square 拟合结果
        x1, x2, x3 = np.linalg.lstsq(A, Y, rcond=-1)[0]

        # 系数反算
        x1 = np.exp(x1)

        # 生成模型
        self.predict_main = lambda O, D, R: x1 * \
            (O.reshape(-1, 1) * D)**x2 * self.func_r(R, -x3)

    def voorhees(self, r=1):
        '''
        乌尔希斯中立模型
        Parameters:
            r: 迭代系数
        '''
        # 计算阻抗函数值矩阵
        Fr = self.func_r(self.Rr, r)

        # 现状年理论OD矩阵
        T = self.Or.reshape(-1, 1) * self.Dr * Fr / (self.Dr * Fr).sum(axis=1)

        # 计算实际&理论平均阻抗
        Rr = (self.Tr*self.Rr).sum() / self.Tr.sum()
        Rp = (T*self.Rr).sum() / T.sum()

        # 计算误差
        error = np.abs(Rp-Rr) / Rr

        # 判断是否迭代
        if error > 0.03:
            if Rr > Rp:
                return self.voorhees(r/2)
            else:
                return self.voorhees(r/2*3)
        else:
            # 生成模型
            self.predict_main = lambda O, D, R: O.reshape(-1, 1) * D * self.func_r(
                R, r) / (D * self.func_r(R, r)).sum(axis=1)
            return r, T

    def bpr(self):
        '''
        美国公路局重力模型
        '''

        # 使用Voorhees计算出r与理论T
        r, T = self.voorhees()

        # 标定系数
        M = self.Tr / T
        Y = self.Tr / self.Or.reshape(-1, 1)
        K = (1 - Y)*M / (1 - Y*M)
        
        # 生成模型
        self.predict_main = lambda O, D, R: O.reshape(-1, 1) * D * self.func_r(
            R, r) * K / (D * self.func_r(R, r)*K).sum(axis=1)

    def bin_constraint(self):
        '''
        双约束重力模型
        '''
        self.r, Ki, Kj = self.bin_r_iter()
        self.K = Ki.reshape(-1, 1) * Kj
        self.predict_main = lambda O, D, R: self.K * \
            O.reshape(-1, 1) * D * self.power(R, self.r)

    def bin_r_iter(self, r=1):
        '''
        双约束r迭代函数
        '''
        Fr = self.power(self.Rr, r)
        Kj = np.ones(self.Dr.size)
        Ki = Ki = (Kj * self.Dr * Fr).sum(axis=1)**-1
        Ki, Kj = self.bin_k_iter(Fr, Ki, Kj)
        T = Ki.reshape(-1, 1) * Kj * self.Or.reshape(-1, 1) * self.Dr * Fr
        Rr = (self.Tr * self.Rr).sum() / self.Tr.sum()
        Rp = (T * self.Rr).sum() / T.sum()
        error = np.abs(Rp-Rr) / Rr
        if error > 0.03:
            if Rr > Rp:
                return self.bin_r_iter(r/2)
            else:
                return self.bin_r_iter(r/2*3)
        else:
            return r, Ki, Kj

    def bin_k_iter(self, Fr, Ki, Kj):
        '''
        双约束K迭代函数
        '''
        Kj_new = (Ki * self.Or * Fr.T).sum(axis=1)**-1
        Ki_new = (Kj_new * self.Dr * Fr).sum(axis=1)**-1
        if (np.abs(1 - Ki_new / Ki) > 0.03).any() or (np.abs(1 - Kj_new / Kj) > 0.03).any():
            return self.bin_k_iter(Fr, Ki_new, Kj_new)
        else:
            return Ki_new, Kj_new

    def predict(self, O, D, R):
        '''
        预测规划年出行分布量
        '''
        return self.predict_main(O, D, R)

    @staticmethod
    def power(C, r):
        '''
        幂函数-阻抗函数
        '''
        return 1 / C ** r

    @staticmethod
    def exp(C, r):
        '''
        指数函数阻抗函数
        '''
        return 1 / np.exp(C * r)


#######################
#  渣渣课本 例题数据
#######################
ODnow = np.array([
    [150, 100, 50],
    [400, 100, 200]
])

Rnow = np.array([
    [3, 2, 5],
    [3, 5, 4]
])

########################
#  邵春福 PPT 例题数据
########################
CF_OD_NOW = np.array([
    [17, 7, 4],
    [7, 38, 6],
    [4, 5, 17]
])
CF_C_NOW = np.array([
    [7, 17, 22],
    [17, 15, 23],
    [22, 23, 7]
])

CF_C_FUT = np.array([
    [4, 9, 11],
    [9, 8, 12],
    [11, 12, 4]
])
CF_O_F = np.array([38.6, 91.9, 36.0])
CF_D_F = np.array([39.3, 90.3, 36.9])
if __name__ == '__main__':
    gravity = ODGravity(model=ODGravity.BPR)
    gravity.fit(CF_OD_NOW, CF_C_NOW)
    ret = gravity.predict(CF_O_F, CF_D_F, CF_C_FUT)
    print(ret)
