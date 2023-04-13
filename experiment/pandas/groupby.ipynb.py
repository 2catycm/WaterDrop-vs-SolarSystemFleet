#%%
from pathlib import Path

import numpy as np
import pandas as pd
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent.parent
import sys
sys.path.append(project_directory.as_posix())
#%%
import geatpy as ea
import joblib
import src.utils as utils
data = utils.get_data()
opportunity_list = utils.get_opportunity_list(data)
stations, asteroids = utils.get_stations_and_asteroids(opportunity_list)
opportunities = len(opportunity_list)
# %%
group = opportunity_list.groupby('station')
group
# %%
s1 = list(group)[0][1]
s1
#%%
shift_positions = [0.000000000000000000e+00,9.800000000000000000e+01,
                   1.930000000000000000e+02,2.890000000000000000e+02,
                   3.860000000000000000e+02,4.820000000000000000e+02,
                   6.020000000000000000e+02,6.770000000000000000e+02,
                   7.670000000000000000e+02,8.620000000000000000e+02,
                   9.690000000000000000e+02,1.073000000000000000e+03]

# %%
def star_table(shift_positions):
        # 根据12个切换局部位置，得到星序表。
        # 星序表是 opportunity list 的子集。
        # 按照时间顺序，但是 分为12个时间段，每个时间段只有一个station。
        shift_positions = np.array(shift_positions, dtype=int)
        shift_positions = [group.indices[station_minus_one + 1][position] for station_minus_one, position in
                           enumerate(shift_positions)]
        station_order = np.argsort(shift_positions)
        res = []
        for i, station_minus_one in enumerate(station_order):
            station = station_minus_one + 1
            left = shift_positions[station_minus_one]
            right = shift_positions[station_order[i + 1]] if i < stations - 1 else opportunities
            right_time =opportunity_list.time.iloc[right] if i < stations - 1 else 81.1
            all_ops = opportunity_list.iloc[left:right]
            res.append(all_ops[(all_ops.station == station)  # 只有这条船有关的才有用
                              & (all_ops.time < right_time - 1)  # 不能僵硬切换
                ]
            )
        res = pd.concat(res)
        return res
s = star_table(shift_positions)
s.to_csv('star_table.csv', index=True)
# %%
s.head()
#%%
q = np.zeros(3)
q += s.iloc[0][['A', 'B', 'C']]
q+=q
q
# %%
