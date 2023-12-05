import pdb
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd

ids = [
    'TrackID',
    'DriverID',
    'TeamID',
]

if __name__ == "__main__":
    data = pd.read_hdf("data/f1_dataset.h5")
    prev_lap_time = []
    for idx in tqdm(range(len(data))):
        id = data.iloc[idx].loc[ids+["Year","LapNumber"]]
        if idx ==0:
            prev_lap_time.append(pd.Timedelta(0.0))
            continue

        if id["LapNumber"] == 1:
            prev_lap_time.append(pd.Timedelta(0.0))
            continue

        if (data.iloc[idx-1].loc[ids]==id[ids]).to_numpy().all():
            prev_lap_time.append(data.iloc[idx-1].loc["LapTime"])
        else:
            prev_lap_time.append(pd.NaT)

    data["PrevLapTime"] = prev_lap_time
    data.to_hdf("data/f1_dataset.h5", key="data", format='fixed', mode='w')
