#%%
from pathlib import Path

import numpy as np
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
sys.path.append(project_directory.as_posix())

import src.utils as utils
data = utils.get_data()
opportunity_list = utils.get_opportunity_list(data)
stations, asteroids = utils.get_stations_and_asteroids(opportunity_list)
opportunities = len(opportunity_list)
#%%
station_groups = opportunity_list.groupby('station')
group_indices = station_groups.indices
#%%
import pandas as pd
df = pd.read_csv(project_directory/'result_game1_evolve_static_single.py'/'optPop'/'Phen.csv', header=None, names=[f'station{i}'for i in range(1, 12+1)])
df = df.astype(int)
df.head()
#%%
Vars = df.to_numpy()
Vars
#%%
Vars = np.vstack((Vars, Vars))
Vars
# %%
shift_positions = Vars[:, :stations]
shift_list_positions = [group_indices[station_minus_one + 1][position] 
                   for station_minus_one, position in
                           enumerate(shift_positions.T)]
shift_list_positions = np.array(shift_list_positions)
shift_list_positions
# %%
start_time = opportunity_list.time.to_numpy()[shift_list_positions]
start_time 

#%%
augment_start_time = np.vstack((start_time, np.ones(start_time.shape[1])*80))
augment_start_time
# %%
station_order = np.argsort(augment_start_time, axis=0)
# station_order
station_rank = np.argsort(station_order, axis=0)
station_rank
#%%
end_time = np.zeros_like(start_time)
for col in range(end_time.shape[1]):
    # 每一列是一个population
    end_time[:, col] = augment_start_time[station_order[station_rank[:-1,col]+1, col], col]
start_time, end_time
# %%
# col_index = np.tile(np.arange(Vars.shape[0]), (12, 1))
# end_time = augment_start_time[station_order[station_rank[:-1, :]+1], col_index]
# end_time
# # %%
# station_order[station_rank[:-1, :]+1]
#%%
opportunity_list.time.iloc[2]
# %%
do_give_up_stars = np.zeros((Vars.shape[0], opportunities))
Vars = np.hstack((Vars, do_give_up_stars))
Vars
#%%
import src.representations as repre
repre.vars2raw_vars(Vars, group_indices, opportunity_list)

# %%
# for op in opportunity_list:
for op in opportunity_list.itertuples(index=True):
    print(op)
    print(op.Index)
    print(op.index)
    print(op.time)
    break

# %%
# opportunity_list.Index
# %%
A = np.arange(9).reshape(3,3)
B = np.arange(3).reshape(3,1)
# np.vstack((A,B))
np.hstack((A,B))

# %%
