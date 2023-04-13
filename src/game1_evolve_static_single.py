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
    problem = WaterDropMarch()
    NIND = 100
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


class WaterDropMarch(ea.Problem):  # Inherited from Problem class.
    def __init__(self):
        M = 1  # M is the number of objects.
        name = 'WaterDropMarch'  # Problem's name.
        maxormins = [0] * M  # 0 表示最大化

        self.station_groups = opportunity_list.groupby('station')
        Dim = stations + opportunities  # 决策维度。 12个切换时机+13920个让星决策(实际上不应该那么多)
        varTypes = [1] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * Dim  # The lower bound of each decision variable.
        ub = list(map(int, self.station_groups.count().time)) + [1] * opportunities  # 切换时机是1
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.

        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 其他变量
        self.indexes = np.arange(asteroids) + 1
        self.time_window_dim = 2 * stations

    def calReferObjV(self):  # Calculate the theoretic global optimal solution here.
        return 9.594

    def aimFunc(self, pop):  # Write the aim function here, pop is an object of Population class.
        Vars = pop.Phen  # Get the decision variables, 每一行是一个。
        raw_vars = self.vars2raw_vars(Vars)
        udp_vars = self.raw_vars2udp_vars(raw_vars)
        fitness = np.array([udp.fitness(var) for var in udp_vars])
        pop.ObjV = -fitness[:, 0].reshape(-1, 1)
        pop.CV = np.sum(fitness[:, 1:], axis=1).reshape(-1, 1)

    def vars2raw_vars(self, Vars):

        pass

    def assignments2time_window(self, assignments):
        # assignments 是 P* ()
        pass

    def raw_vars2udp_vars(self, raw_vars):
        udp_vars = np.zeros((raw_vars.shape[0], raw_vars.shape[1] + asteroids))
        indexes = np.tile(self.indexes, (raw_vars.shape[0], 1))
        udp_vars[:, self.time_window_dim] = raw_vars[:, self.time_window_dim]
        udp_vars[:, self.time_window_dim::3] = indexes  # 固定的id
        udp_vars[:, self.time_window_dim + 1::3] = raw_vars[:, self.time_window_dim::2]
        udp_vars[:, self.time_window_dim + 2::3] = raw_vars[:, self.time_window_dim + 1::2]
        return udp_vars
