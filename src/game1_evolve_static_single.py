# %%
import typing
from concurrent.futures import ThreadPoolExecutor
from random import random

import numpy as np
import torch
from pathlib import Path

import src.utils as utils
import geatpy as ea
from spoc_delivery_scheduling_evaluate_code import trappist_schedule
import joblib
import src.game1_dynamic_giveup as game1_dynamic_giveup

udp = trappist_schedule()

data = utils.get_data()
opportunity_list = utils.get_opportunity_list(data)
stations, asteroids = utils.get_stations_and_asteroids(opportunity_list)
opportunities = len(opportunity_list)


def opt_decision(dynamic=False, log=False, equal_shift=False):
    # 返回给EA.py
    problem = WaterDropMarch()
    # 软件测试用的超小参数
    # NIND = 20
    # MAXGEN = 10
    # 算法对比使用的参数
    NIND = 100
    MAXGEN = 20
    # 基本参数
    # NIND = 1000
    # MAXGEN = 200
    # MAXGEN = 500

    # 构建算法
    # algorithm = ea.soea_SGA_templet(problem,
    algorithm = ea.soea_DE_targetToBest_1_L_templet(problem,
                                                    ea.Population(Encoding='RI', NIND=NIND),  # Set 100 individuals.
                                                    MAXGEN=MAXGEN,  # Set the max iteration number.
                                                    logTras=1,
                                                    # Set the frequency of logging. If it is zero, it would not log.
                                                    trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                                    maxTrappedCount=100  # 进化停滞计数器最大上限值。
                                                    )
    prophet = np.array([
        np.round(problem.ub * np.arange(stations) / stations).astype(int),
        np.round(problem.ub * (stations - 1 - np.arange(stations)) / stations).astype(int),
    ]
    ) if equal_shift else None

    res = ea.optimize(algorithm,
                      seed=None, verbose=log, drawing=log, outputMsg=log, drawLog=log, saveFlag=log,
                      dirName=f'result_{Path(__file__).name}',
                      prophet=prophet
                      )
    # print(res)
    objVs = np.min(res['ObjV'], axis=1)
    print(objVs)
    print(f"最强的内部fitness是{max(objVs)}")

    return vars2udp_var(res['Vars'], problem, dynamic)


def vars2udp_var(vars, problem, dynamic=False):
    import src.representations as resp
    env = game1_dynamic_giveup.WaterDropMarch(opportunity_list, vars[0])
    if dynamic:
        rewards, best_do_give_up_stars = game1_dynamic_giveup.best_sample_rewards(env, samples=20)
        # rewards, best_do_give_up_stars = game1_dynamic_giveup.best_sample_rewards(env, samples=200, log=True)
        # rewards, best_do_give_up_stars = game1_dynamic_giveup.best_sample_rewards(env, samples=1000)
        vars = resp.make_vars(vars, best_do_give_up_stars)
    else:
        vars = resp.make_vars(vars)
    raw_vars = resp.vars2raw_vars(vars, problem.indices, opportunity_list)
    udp_vars = resp.raw_vars2udp_vars(raw_vars, )
    fit = np.array([udp.fitness(udp_var) for udp_var in udp_vars])
    print(fit)
    return udp_vars[int(fit[:, 0].argmin())], -fit[:, 0].max() if not dynamic else rewards


class WaterDropMarch(ea.Problem):  # Inherited from Problem class.
    def __init__(self):
        M = 1  # M is the number of objects.
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
        # self.K = 10
        self.K = 1

    def calReferObjV(self):  # Calculate the theoretic global optimal solution here.
        # uniformPoint, ans = ea.crtup(self.M, 10000)  # create 10000 uniform points.
        # realBestObjV = uniformPoint / 2
        # return realBestObjV
        return np.ones((100, self.M)) * 9.594

    def evalVars(self, Vars):
        # shift_positions = Vars[:, :stations]
        # station_order = np.argsort(shift_positions, axis=1)
        # for station in station_order.T:
        CV = np.zeros((Vars.shape[0], 1))
        # obj_v = np.zeros(Vars.shape[0], self.M)
        obj_v = np.array([self.evalSingleVars(svar) for svar in Vars]).reshape(Vars.shape[0], self.M)

        # envs = [game1_dynamic_giveup.WaterDropMarch(opportunity_list, svar) for svar in Vars]
        # rewards = [game1_dynamic_giveup.best_sample_rewards(env, samples=10)[0] for env in envs]
        # with ThreadPoolExecutor(max_workers=16) as executor:
        #     env_futures = [executor.submit(game1_dynamic_giveup.WaterDropMarch, opportunity_list, svar) for svar in
        #                    Vars]
        #     futures = [executor.submit(game1_dynamic_giveup.best_sample_rewards, env.result(), samples=10, log=False)
        #                for env in env_futures]
        # rewards = [future.result()[0] for future in futures]
        # obj_v += rewards
        # obj_v = np.array(rewards).reshape(Vars.shape[0], self.M)

        return obj_v, CV

    # @ea.Problem.single
    def evalSingleVars(self, svar):
        shift_positions = svar[:stations]  # 这个只是station内部的位置
        shift_positions = [self.indices[station_minus_one + 1][position] for station_minus_one, position in
                           enumerate(shift_positions)]

        station_order = np.argsort(shift_positions)
        obj_v = np.zeros(stations)
        asteroids_used = np.zeros(asteroids + 1, dtype=bool)
        for i, station_minus_one in enumerate(station_order):
            station = station_minus_one + 1
            left = shift_positions[station_minus_one]
            right = shift_positions[station_order[i + 1]] if i < stations - 1 else opportunities
            all_ops = opportunity_list.iloc[left:right]
            right_time = opportunity_list.time.iloc[right] if i < stations - 1 else 81.1
            all_ops = all_ops[all_ops.station == station]  # 只有这条船有关的才有用
            all_ops = all_ops[all_ops.time < right_time - 1]  # 不能僵硬切换

            # all_ops = all_ops[asteroids_used[all_ops.asteroid] == 0]  # 没有量子通信逃逸的飞船

            # obj_v[station_minus_one] = all_ops[['A', 'B', 'C']].sum(axis=0).min()  # 增加奖赏
            # asteroids_used[all_ops.asteroid] = 1  # 设置为使用过了
            ABC = np.zeros(3)
            for op in all_ops.itertuples():
                # 这里不能批量操作，因为量子逃逸可能随时发生，同一个station也不能用两次。
                if asteroids_used[op.asteroid] == 1:
                    continue
                ABC[0] += op.A
                ABC[1] += op.B
                ABC[2] += op.C
                # 为下一次筛选增加条件
                asteroids_used[op.asteroid] = 1

            obj_v[station_minus_one] = ABC.min()
        return -np.log(np.sum(np.exp(-self.K * obj_v))) / self.K

        # env = game1_dynamic_giveup.WaterDropMarch(opportunity_list, svar)
        # rewards, best_do_give_up_stars = game1_dynamic_giveup.best_sample_rewards(env, samples=10)
        # return rewards
