import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from pathlib import Path

from gym import spaces

this_file = Path(__file__).resolve()
this_directory = this_file.parent


def get_data(path=this_directory.parent / 'data/spoc/scheduling/candidates.txt'):
    with open(path) as f:
        data = json.load(f)
    return data


def get_opportunity_list(data):
    # 规约为一个简单点的表示
    # List of Opportunity,
    # Opportunity = [time, station, asteroid, A, B, C] .
    opportunities = []
    for asteroid in data.keys():
        for station in data[asteroid].keys():
            for oppo_id_minus_one, op in enumerate(data[asteroid][station]):
                # print(op)
                opportunities.append([op[0], int(station), int(asteroid), *op[1:], oppo_id_minus_one+1])
    opportunities.sort(key=lambda x: x[0])
    df = pd.DataFrame(opportunities, columns=["time", "station", "asteroid", "A", "B", "C", "oppo_id"])
    return df


def get_stations_and_asteroids(opportunity_list: pd.DataFrame):
    return len(opportunity_list.station.unique()) , len(opportunity_list.asteroid.unique())


def zeros_space(space: spaces.Space):
    if isinstance(space, spaces.Dict):
        res = OrderedDict()
        for k in space.keys():
            res[k] = zeros_space(space[k])
        return res
    elif isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    elif isinstance(space, spaces.Discrete):
        # return np.zeros(1, dtype=np.int64)
        return 0
    elif isinstance(space, spaces.MultiBinary):
        return np.zeros(space.n, dtype=bool)
    elif isinstance(space, spaces.MultiDiscrete):
        return np.zeros(space.nvec, dtype=np.int64)

def dict_into_single_array(d:OrderedDict):
    res = []
    for k, v in d.items():
        if isinstance(v, OrderedDict):
            res.append(dict_into_single_array(v))
        else:
            res.append(v)
    return np.hstack(res)

# def space_dict_into_single_array(s:spaces.Space):
#     if isinstance(space, spaces.Dict):
#         res = OrderedDict()
#         for k in space.keys():
#             res[k] = zeros_space(space[k])
#         return res
#     elif isinstance(space, spaces.Box):
#         return np.zeros(space.shape, dtype=space.dtype)
#     elif isinstance(space, spaces.Discrete):
#         # return np.zeros(1, dtype=np.int64)
#         return 0
#     elif isinstance(space, spaces.MultiBinary):
#         return np.zeros(space.n, dtype=bool)
#     elif isinstance(space, spaces.MultiDiscrete):
#         return np.zeros(space.nvec, dtype=np.int64)