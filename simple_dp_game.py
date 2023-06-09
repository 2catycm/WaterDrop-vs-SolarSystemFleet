from pathlib import Path

this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory
import sys

sys.path.append(project_directory.as_posix())

import typing
from collections import OrderedDict
from typing import Tuple

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from src.utils import zeros_space, dict_into_single_array  # , space_dict_into_single_array
import numpy as np

class WaterDropMarchEasyMode(gym.Env):
    metadata = {
        "render_modes": ["human",  # render to the current display or terminal and return Nothing
                         "ansi",  # 返回一个 str
                         "rgb_array"],  # Return a numpy.ndarray with shape (x, y, 3)
        "render_fps": 4}
    very_big = 10

    def __init__(self, opportunity_list: pd.DataFrame, shift_positions: np.ndarray, render_mode=None, rows=12,
                 channels=340,
                 ):
        """动态多阶段决策是否让星，返回一个决策列表。

        Args:
            opportunity_list (np.ndarray): 一万多条记录
            shift_positions (np.ndarray): 12个迁移index，index是记录中的位置
            render_mode (str, optional): 没啥用的动态游戏展示. Defaults to None.
        """
        self.first_time = False  # 是不是第一段，还没有合适的min
        self.opportunity_list = opportunity_list
        self.rows = rows or len(opportunity_list.station.unique())
        self.channels = channels or len(opportunity_list.asteroid.unique())
        self.opportunities = len(opportunity_list)
        self.action_space = spaces.Discrete(2)  # 决定是否放弃。1表示放弃。
        # 计算一下问题的空间。
        self.group = opportunity_list.groupby('station')
        self.group_indices = self.group.indices
        self.star_table = self.get_star_table(shift_positions)
        self.stars = len(self.star_table)

        self.observation_space = spaces.Dict({"position": spaces.Discrete(self.stars),
                                              "banned_channels": spaces.MultiBinary(self.channels + 1),
                                              "current_ABC": spaces.Box(low=0, high=np.inf, shape=(3,),
                                                                        dtype=np.float32),
                                              "previous_min": spaces.Box(low=0, high=np.inf, shape=(1,),
                                                                         dtype=np.float32),
                                              "current_row": spaces.Discrete(self.rows + 1),
                                              })
        self.state: OrderedDict = self.reset()

        # human mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # 返回值
        self.do_give_up_stars = np.zeros((1, self.opportunities))

    def normalized(self, state):
        state['position'] = int(state['position'])
        state['banned_channels'] = np.array(state['banned_channels'], dtype=int).reshape((self.channels + 1,))
        state['current_ABC'] = np.array(state['current_ABC'], dtype=np.float32).reshape((3,))
        state['previous_min'] = np.array(state['previous_min'], dtype=np.float32).reshape((1,))
        state['previous_min'] = np.array(state['previous_min'], dtype=np.float32)
        state['current_row'] = int(state['current_row'])
        return state

    def get_star_table(self, shift_positions) -> pd.DataFrame:
        # 根据12个切换局部位置，得到星序表。
        # 星序表是 opportunity list 的子集。
        # 按照时间顺序，但是 分为12个时间段，每个时间段只有一个station。
        shift_positions = np.array(shift_positions, dtype=int)
        shift_positions = [self.group_indices[station_minus_one + 1][position] for station_minus_one, position in
                           enumerate(shift_positions)]
        station_order = np.argsort(shift_positions)
        res = []
        for i, station_minus_one in enumerate(station_order):
            station = station_minus_one + 1
            left = shift_positions[station_minus_one]
            right = shift_positions[station_order[i + 1]
            ] if i < self.rows - 1 else self.opportunities
            right_time = self.opportunity_list.time.iloc[right] if i < self.rows - 1 else 81.1
            all_ops = self.opportunity_list.iloc[left:right]
            res.append(all_ops[(all_ops.station == station)  # 只有这条船有关的才有用
                               & (all_ops.time < right_time - 1)  # 不能僵硬切换
                               ]
                       )
        res = pd.concat(res)
        return res

    def reset(self) -> typing.OrderedDict:
        state = zeros_space(self.observation_space)
        # state['previous_min'] = np.inf  # 一开始没有上界限制
        state['previous_min'] = self.very_big  # 一开始没有上界限制
        state['current_row'] = self.star_table.iloc[0].station
        self.first_time = True
        self.state = self.normalized(state)
        self.do_give_up_stars = np.zeros((1, self.opportunities))
        return self.state

    def step(self, action) -> Tuple[typing.OrderedDict, float, bool, dict]:
        # 使用设计的reward来进行学习。
        reward, done, info = 0, False, dict()
        position = self.state['position']
        star = self.star_table.iloc[position]
        is_banned = self.state['banned_channels'][int(star.asteroid)] == 1
        if action == 0 and not is_banned:
            # 不让星
            # ABC的有效增加量。
            new_ABC = self.state['current_ABC'] + star[['A', 'B', 'C']].to_numpy()
            reward += new_ABC.min() - self.state['current_ABC'].min()
            self.state['current_ABC'] = new_ABC
            self.state['banned_channels'][int(star.asteroid)] = 1
        elif action == 1 or is_banned:
            self.do_give_up_stars[0, self.star_table.index[position]] = True
            # 让星
            reward += 0  # 短期没有收益
        else:
            raise Exception("Illegal Action!")
        # 如果是本row的最后一个，就要进行结算
        if (position == self.stars - 1) or self.star_table.iloc[position + 1].station != self.state['current_row']:
            # 首先换行
            if position != self.stars - 1:  # 如果是最后的话，换行可能报错。
                self.state['current_row'] = self.star_table.iloc[position + 1].station
            # 分情况讨论
            if self.state['current_ABC'].min() >= self.state['previous_min']:
                # 则没有灾难发生，previous_min也不变，但是本阶段的reward收回。
                reward += -self.state['current_ABC'].min()
            elif not self.first_time:
                # 则本阶段的reward有效，而之前的min无效，min更新。
                reward += -self.state['previous_min']
                self.state['previous_min'] = self.state['current_ABC'].min()
            else:
                # 注意特殊判断，如果是第一次做切换，就都不需要扣除。
                reward += 0
                self.state['previous_min'] = self.state['current_ABC'].min()
                self.first_time = False
            self.state['current_ABC'] = np.zeros(3)  # 清零
        # position的变化。前进一格
        self.state['position'] = position + 1
        if position == self.stars - 1:
            done = True  # 如果是全局最后一个，结束游戏。
        self.state = self.normalized(self.state)
        return self.state, reward, done, info

    def real_reward_step(self, action) -> Tuple[typing.OrderedDict, float, bool, dict]:
        # 返回 observation（?比state的信息可以更多）, reward, done, info
        reward, done, info = 0, False, dict()
        position = self.state['position']
        star = self.star_table.iloc[position]
        is_banned = self.state['banned_channels'][int(star.asteroid)] == 1
        if action == 0 and not is_banned:
            # 不让星
            # ABC的有效增加量。
            new_ABC = self.state['current_ABC'] + star[['A', 'B', 'C']].to_numpy()
            reward += new_ABC.min() - self.state['current_ABC'].min()
            self.state['current_ABC'] = new_ABC
            self.state['banned_channels'][int(star.asteroid)] = 1
        elif action == 1 or is_banned:
            # 让星
            reward += 0  # 短期没有收益
        else:
            raise Exception("Illegal Action!")
        # 如果是本row的最后一个，就要进行结算
        if (position == self.stars - 1) or self.star_table.iloc[position + 1].station != self.state['current_row']:
            # 首先换行
            if position != self.stars - 1:  # 如果是最后的话，换行可能报错。
                self.state['current_row'] = self.star_table.iloc[position + 1].station
            # 分情况讨论
            if self.state['current_ABC'].min() >= self.state['previous_min']:
                # 则没有灾难发生，previous_min也不变，但是本阶段的reward收回。
                reward += -self.state['current_ABC'].min()
            elif not self.first_time:
                # 则本阶段的reward有效，而之前的min无效，min更新。
                reward += -self.state['previous_min']
                self.state['previous_min'] = self.state['current_ABC'].min()
            else:
                # 注意特殊判断，如果是第一次做切换，就都不需要扣除。
                reward += 0
                self.state['previous_min'] = self.state['current_ABC'].min()
                self.first_time = False
            self.state['current_ABC'] = np.zeros(3)  # 清零
        # position的变化。前进一格
        self.state['position'] = position + 1
        if position == self.stars - 1:
            done = True  # 如果是全局最后一个，结束游戏。
        self.state = self.normalized(self.state)
        return self.state, reward, done, info

    def render(self, mode='human'):
        return None

    def close(self):
        return None