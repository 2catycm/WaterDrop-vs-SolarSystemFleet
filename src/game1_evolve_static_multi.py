# %%
import typing

import numpy as np
import torch
from pathlib import Path

import src.utils as utils
import geatpy as ea
from spoc_delivery_scheduling_evaluate_code import trappist_schedule
import joblib

udp = trappist_schedule()

data = utils.get_data()
opportunity_list = utils.get_opportunity_list(data)
stations, asteroids = utils.get_stations_and_asteroids(opportunity_list)
opportunities = len(opportunity_list)


def opt_decision():
    # 返回给EA.py
    problem = WaterDropMarch()
    # NIND = 1000
    NIND = 20
    MAXGEN = 10
    # MAXGEN = 500
    # 构建算法
    algorithm = ea.moea_NSGA3_templet(problem,
                                      ea.Population(Encoding='RI', NIND=NIND),  # Set 100 individuals.
                                      MAXGEN=MAXGEN,  # Set the max iteration number.
                                      logTras=1,  # Set the frequency of logging. If it is zero, it would not log.
                                      trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                      maxTrappedCount=100  # 进化停滞计数器最大上限值。
                                      )
    # 求解
    # res = ea.optimize(algorithm,
    #                   seed=128, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
    #                   dirName=f'result_{Path(__file__).name}')

    res = ea.optimize(algorithm,
                      seed=128, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
                      dirName=f'result_{Path(__file__).name}',
                      prophet=np.array([
                          # [*i/stations for i in range(stations)],
                          np.round(problem.ub * np.arange(stations)/stations).astype(int),
                          np.round(problem.ub * (stations-1-np.arange(stations))/stations).astype(int),
                           ]
                      ))
    # print(res)
    objVs = np.min(res['ObjV'], axis=1)
    print(objVs)
    print(f"最强的是{max(objVs)}")

    raise "还没写Vars转换为官方vector"


class WaterDropMarch(ea.Problem):  # Inherited from Problem class.
    def __init__(self):
        M = stations  # M is the number of objects.
        name = 'WaterDropMarch'  # Problem's name.
        maxormins = [-1] * M  # -1 表示最大化

        self.station_groups = opportunity_list.groupby('station')
        # Dim = stations + opportunities  # 决策维度。 12个切换时机+13920个让星决策(实际上不应该那么多)
        # varTypes = [1] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        # lb = [0] * Dim  # The lower bound of each decision variable.
        # ub = list(map(int, self.station_groups.count().time)) + [1] * opportunities  # 切换时机是1
        # lbin = [1] * Dim  # Whether the lower boundary is included.
        # ubin = [1] * Dim  # Whether the upper boundary is included.

        Dim = stations  # 决策维度。 12个切换时机+先到先得
        varTypes = [1] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * Dim  # The lower bound of each decision variable.
        ub = list(map(int, self.station_groups.count().time))  # 切换时机是1
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [0] * Dim  # Whether the upper boundary is included.

        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.indices = self.station_groups.indices

    def calReferObjV(self):  # Calculate the theoretic global optimal solution here.
        # uniformPoint, ans = ea.crtup(self.M, 10000)  # create 10000 uniform points.
        # realBestObjV = uniformPoint / 2
        # return realBestObjV
        return np.ones(self.M) * 9.594

    def evalVars(self, Vars):
        # shift_positions = Vars[:, :stations]
        # station_order = np.argsort(shift_positions, axis=1)
        # for station in station_order.T:
        CV = np.zeros((Vars.shape[0], 1))
        # obj_v = np.zeros(Vars.shape[0], self.M)
        obj_v = np.array([self.evalSingleVars(svar) for svar in Vars])
        return obj_v, CV

    # @ea.Problem.single
    def evalSingleVars(self, svar):
        shift_positions = svar[:stations]  # 这个只是station内部的位置
        shift_positions = [self.indices[station_minus_one + 1][position] for station_minus_one, position in
                           enumerate(shift_positions)]

        station_order = np.argsort(shift_positions)
        obj_v = np.zeros(self.M)
        asteroids_used = np.zeros(asteroids + 1, dtype=bool)
        for i, station_minus_one in enumerate(station_order):
            station = station_minus_one + 1
            left = shift_positions[station_minus_one]
            right = shift_positions[station_order[i + 1]] if i < stations - 1 else opportunities
            all_ops = opportunity_list.iloc[left:right]
            all_ops = all_ops[all_ops.station == station]  # 只有这条船有关的才有用
            all_ops = all_ops

            all_ops = all_ops[asteroids_used[all_ops.asteroid] == 0]  # 没有量子通信逃逸的飞船
            obj_v[station_minus_one] = all_ops[['A', 'B', 'C']].sum(axis=0).min()  # 增加奖赏
            asteroids_used[all_ops.asteroid] = 1  # 设置为使用过了
        return obj_v
