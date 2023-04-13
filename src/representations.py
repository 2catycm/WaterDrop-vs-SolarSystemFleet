# %%
import numpy as np


def make_vars(shift_positions, do_give_up_stars=None, opportunities=13920):
    if do_give_up_stars is None:
        do_give_up_stars = np.zeros((shift_positions.shape[0], opportunities))
    return  np.hstack((shift_positions, do_give_up_stars))
    
def vars2raw_vars(Vars, group_indices, opportunity_list, stations=12, asteroids=340, opportunities=13920):
    """切换时机+抢星表示转换为原始表示。
    Args:
        Vars (array(p x (stations+opportunities))): p个个体
    """
    Vars = np.array(Vars).astype(int)
    shift_positions = Vars[:, :stations]
    do_give_up_stars = Vars[:, stations:]

    # 首先计算 窗口期
    shift_list_positions = [group_indices[station_minus_one + 1][position]
                            for station_minus_one, position in
                            enumerate(shift_positions.T)]  # 现在打横的是不同的p
    shift_list_positions = np.array(shift_list_positions)

    start_time = opportunity_list.time.to_numpy()[shift_list_positions]
    augment_start_time = np.vstack((start_time, np.ones(start_time.shape[1]) * 81.1))
    station_order = np.argsort(augment_start_time, axis=0)
    station_rank = np.argsort(station_order, axis=0)
    end_time = np.zeros_like(start_time)
    for col in range(end_time.shape[1]):
        # 每一列是一个population
        end_time[:, col] = augment_start_time[station_order[station_rank[:-1, col] + 1, col], col]
    end_time = end_time - 1  # 满足必须间隔1的标准
    time_windows = make_time_windows(start_time.T, end_time.T)
    # 然后计算 raw_assignments
    station_ids = np.zeros((Vars.shape[0], asteroids))
    opportunity_ids = np.zeros((Vars.shape[0], asteroids))
    for individual in range(Vars.shape[0]):
        asteroids_used = np.zeros(asteroids + 1, dtype=bool)
        for i, station_minus_one in enumerate(station_order[:-1, individual]):
            station = station_minus_one + 1
            left = shift_list_positions[station_minus_one, individual]
            right = shift_list_positions[station_order[i + 1, individual], individual] if i < stations - 1 else opportunities
            all_ops = opportunity_list.iloc[left:right]
            right_time = opportunity_list.time.iloc[right] if i < stations - 1 else 81.1
            all_ops = all_ops[(all_ops.station == station)  # 只有这条船有关的才有用
                              & (all_ops.time < right_time - 1)  # 不能僵硬切换
                              & (do_give_up_stars[individual, all_ops.index] == 0)  # 没有让星的地方才留下来
                              & (asteroids_used[all_ops.asteroid] == 0)  # 没有量子通信逃逸的飞船
                              ]
            # 筛选完毕
            # station_ids[individual, all_ops.asteroid]
            for op in all_ops.itertuples():
                # 这里不能批量操作，因为量子逃逸可能发生。
                if asteroids_used[op.asteroid] == 1:
                    continue
                station_ids[individual, op.asteroid-1] = station
                opportunity_ids[individual, op.asteroid-1] = op.oppo_id
                # 为下一次筛选增加条件
                asteroids_used[op.asteroid] = 1

    raw_assignments = make_raw_assignments(station_ids, opportunity_ids)
    return make_raw_vars(time_windows, raw_assignments)



# %%


# def assignments2time_window(assignments):
#     """通过分配表示自动计算时间窗表示

#     Args:
#         assignments (array(p, 340*2)): 每一行是 [station, opportunity, ...]
#     """
#     pass

def make_raw_assignments(station_ids, opportunity_ids, asteroids=340):
    raw_assignments = np.zeros((station_ids.shape[0], asteroids * 2))
    raw_assignments[:, 0::2] = station_ids
    raw_assignments[:, 1::2] = opportunity_ids
    return raw_assignments


def make_time_windows(start_time, end_time, stations=12):
    time_windows = np.zeros((start_time.shape[0], 2 * stations))
    time_windows[:, 0::2] = start_time
    time_windows[:, 1::2] = end_time
    return time_windows


def make_raw_vars(time_windows, raw_assignments):
    """结合

    Args:
        time_windows (px24): _description_
        raw_assignments (px(340*2)): _description_

    Returns:
        array: raw_vars
    """
    return np.hstack((time_windows, raw_assignments))


def raw_vars2udp_vars(raw_vars, asteroid_indices=np.arange(340) + 1, asteroids=340, time_window_dim=24):
    """原始表示转换为udp表示。

    Args:
        raw_vars (_type_): _description_
        asteroid_indices (_type_, optional): _description_. Defaults to np.arange(340)+1.
        asteroids (int, optional): _description_. Defaults to 340.
        time_window_dim (int, optional): _description_. Defaults to 24.

    Returns:
        _type_: _description_
    """
    udp_vars = np.zeros((raw_vars.shape[0], raw_vars.shape[1] + asteroids))
    indexes = np.tile(asteroid_indices, (raw_vars.shape[0], 1))
    udp_vars[:, :time_window_dim] = raw_vars[:, :time_window_dim]
    udp_vars[:, time_window_dim::3] = indexes  # 固定的id
    udp_vars[:, time_window_dim + 1::3] = raw_vars[:, time_window_dim::2]
    udp_vars[:, time_window_dim + 2::3] = raw_vars[:, time_window_dim + 1::2]
    return udp_vars
