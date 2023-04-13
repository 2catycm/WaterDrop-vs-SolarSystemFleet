# %%
import typing

import numpy as np
import torch
from pathlib import Path

import src.utils as utils
import geatpy as ea
from spoc_delivery_scheduling_evaluate_code import trappist_schedule

udp = trappist_schedule()

data = utils.get_data()
opportunity_list = utils.get_opportunity_list(data)
stations, asteroids = utils.get_stations_and_asteroids(opportunity_list)
opportunities = len(opportunity_list)


def opt_decision():
    # 返回给EA.py
    problem = SpocDeliveryScheduling()
    NIND = 100
    # MAXGEN = budget // (NIND * dim)
    MAXGEN = 400
    # 构建算法
    algorithm = ea.soea_SEGA_templet(problem,
                                     ea.Population(Encoding='RI', NIND=NIND),
                                     #  MAXGEN=80,  # 最大进化代数。
                                     MAXGEN=MAXGEN,  # 最大进化代数。
                                     logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                     #  trappedValue=1e-8,  # 单目标优化陷入停滞的判断阈值。
                                     trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                     maxTrappedCount=100)  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm,
                      seed=128, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
                      dirName=f'result_{Path(__file__).name}')
    print(res)
    return res["Vars"]


class SpocDeliveryScheduling(ea.Problem):  # Inherited from Problem class.
    def __init__(self):
        M = 1  # M is the number of objects.
        name = 'SpocDeliveryScheduling'  # Problem's name.
        maxormins = [0] * M  # 0 表示最大化
        self.time_window_dim = 2 * stations
        self.asteroid_dim = 2 * asteroids

        Dim = self.time_window_dim + self.asteroid_dim  # 决策维度。 开始结束时间，选择了哪个station的哪个opportunity
        varTypes = [0] * self.time_window_dim + [
            1] * self.asteroid_dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * self.time_window_dim + [0, 1] * asteroids
        ub = [80] * self.time_window_dim + [12, 8] * asteroids  # 时间窗口可以是0-80， station选择可以是0-12(0表示没有收集)。
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 其他变量
        self.indexes = np.arange(asteroids) + 1

    def aimFunc(self, pop):  # Write the aim function here, pop is an object of Population class.
        Vars = pop.Phen  # Get the decision variables, 每一行是一个。
        udp_vars = np.zeros((Vars.shape[0], Vars.shape[1] + asteroids))
        indexes = np.tile(self.indexes, (Vars.shape[0], 1))
        udp_vars[:, self.time_window_dim] = Vars[:, self.time_window_dim]
        udp_vars[:, self.time_window_dim::3] = indexes  # 固定的id
        udp_vars[:, self.time_window_dim + 1::3] = Vars[:, self.time_window_dim::2]
        udp_vars[:, self.time_window_dim + 2::3] = Vars[:, self.time_window_dim + 1::2]
        fitness = np.array([udp.fitness(var) for var in udp_vars])
        pop.ObjV = -fitness[:, 0].reshape(-1, 1)
        pop.CV = np.sum(fitness[:, 1:], axis=1).reshape(-1, 1)

    def calReferObjV(self):  # Calculate the theoretic global optimal solution here.
        uniformPoint, ans = ea.crtup(self.M, 10000)  # create 10000 uniform points.
        realBestObjV = uniformPoint / 2
        return realBestObjV
