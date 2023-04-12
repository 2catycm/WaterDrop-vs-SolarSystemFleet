# %%
import typing
from collections import OrderedDict
from typing import Tuple

import gym
from gym import spaces
import numpy as np
import torch


# class GameState:
#     def __init__(self):
#         self.channel:int = 0
#         self.opportunities = torch.zeros((0, 6))

def zeros_space(space: spaces.Space):
    if isinstance(space, spaces.Dict):
        res = OrderedDict()
        for k in space.keys():
            res[k] = zeros_space(space[k])
        return res
    elif isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    elif isinstance(space, spaces.Discrete):
        return np.zeros(1, dtype=np.int64)
    elif isinstance(space, spaces.MultiBinary):
        return np.zeros(space.n, dtype=bool)
    elif isinstance(space, spaces.MultiDiscrete):
        return np.zeros(space.nvec, dtype=np.int64)


class WaterDropMarch(gym.Env):
    metadata = {
        "render_modes": ["human",  # render to the current display or terminal and return Nothing
                         "ansi",  # 返回一个 str
                         "rgb_array"],  # Return a numpy.ndarray with shape (x, y, 3)
        "render_fps": 4}

    def __init__(self, opportunity_list: np.ndarray, rows=12, channels=340, render_mode=None
                 ):
        """
        将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作。gym提供了spaces方法。
        rows 是 station的数量。在这个游戏形式化中，station是水滴能够走的列。
        channels 是 asteroids的数量。在这个游戏形式化中，asteroid是太阳系舰队的量子通信频道。
        """
        self.opportunity_list = opportunity_list
        self.rows = rows or len(np.unique(opportunity_list[:, 1]))
        self.channels = channels or len(np.unique(opportunity_list[:, 2]))
        self.opportunities = len(opportunity_list)
        self.action_space = spaces.Discrete(2)  # 决定做不做切换
        self.observation_space = spaces.Dict({"position": spaces.Discrete(self.opportunities),
                                              "current_row": spaces.Discrete(self.rows + 1),
                                              # "disabled": spaces.MultiDiscrete([2 for _ in range(self.opportunities)]),
                                              "banned_rows": spaces.MultiBinary(self.rows + 1),
                                              "banned_channels": spaces.MultiBinary(self.channels),
                                              "current_ABC": spaces.Box(low=0, high=np.inf, shape=(3,),
                                                                        dtype=np.float32),
                                              "previous_min": spaces.Box(low=0, high=np.inf, shape=(0,),
                                                                         dtype=np.float32),
                                              "row_locked": spaces.Discrete(2)  # 1天约束
                                              })
        # print(self.observation_space)
        self.state = self.reset()
        # print(self.state)

        # 记录上一步上锁时间。
        self.lock_time = None

        # human mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def step(self, action) -> Tuple[typing.OrderedDict, float, bool, dict]:
        # 返回 observation（?比state的信息可以更多）, reward, done, info
        reward, done, info = 0, False, None

        star = self.opportunity_list[self.state['position']]
        time, station, asteroid, ABC = star
        if action == 0 or self.state['row_locked'] == 1:
            # 1. 如果action是不做切换， 那就保持轨道，看看这一个星星能不能获得
            # 不能获得情况，被ban的channel
            if self.state['banned_channels'][asteroid] == 0:
                # ABC会积累。
                self.state['currentABC'] += ABC

            # 不用改 row, row不会ban，但是channel可能ban，
            # 这里我们默认切换了必定抢星。而不会让星于后人。
            self.state['banned_channels'][asteroid] = 1
            # 不会做胜利结算
            # row locked 可能解锁
            if self.state['position'] - self.lock_time > 1:
                self.state['row_locked'] = 0

        elif action == 1:
            # 2. 如果action是切换
            self.lock_time = time
            self.state['current_row'] = station
        else:
            raise Exception('invalid action!')

        self.state['position'] += 1  # 前进一格
        return self.state, reward, done, info

    def reset(self) -> typing.OrderedDict:
        state = zeros_space(self.observation_space)
        state['previous_min'] = np.inf  # 一开始没有上界限制
        # state = self.observation_space.sample()
        # state['position'] = 0 # 第一个机会
        # state['current_row'] = 0 # 一开始在空行上， 前面有些机会可以忽略，否则就会丧失机会。
        # state['previous_min'] = np.inf  # 一开始没有上界
        # state['row_locked'] = 0  # 一开始没有锁定，可以切换行
        # state['disabled'] *= 0  # 广播。一开始没有被激光炸毁、量子逃逸的机会
        # state['currentABC'] *= 0  # 广播。一开始没有收集任何物质。注意，这里的ABC是当前考虑的行（station）中最小的。
        self.state = state
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None
