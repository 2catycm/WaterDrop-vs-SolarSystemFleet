import numpy as np


def vars2raw_vars(Vars):
    pass


def assignments2time_window(assignments):
    # assignments 是 P* ()
    pass


def raw_vars2udp_vars(raw_vars, asteroid_indices=np.arange(340) + 1, asteroids=340, time_window_dim=24):
    udp_vars = np.zeros((raw_vars.shape[0], raw_vars.shape[1] + asteroids))
    indexes = np.tile(asteroid_indices, (raw_vars.shape[0], 1))
    udp_vars[:, time_window_dim] = raw_vars[:, time_window_dim]
    udp_vars[:, time_window_dim::3] = indexes  # 固定的id
    udp_vars[:, time_window_dim + 1::3] = raw_vars[:, time_window_dim::2]
    udp_vars[:, time_window_dim + 2::3] = raw_vars[:, time_window_dim + 1::2]
    return udp_vars
