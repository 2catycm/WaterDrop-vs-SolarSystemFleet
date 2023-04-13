import json
import numpy as np
import pandas as pd

from pathlib import Path

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
