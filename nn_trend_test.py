import os
import pdb

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from estimator.RaceNet import RaceNetBranched, RaceNetBranchedPretrainer, F1Dataset

args = {
      'num_layers': 3,
      'hidden_dim': 640,
      'out_dim': 128,
      'emb_dim': 10,
      'dropout': 0.4,
  }
# Make sure these lists match in estimator/RaceNet.py
# They're mostly here for reference.
cols = [
    'Year',
    'TrackID',
    'DriverID',
    'TeamID',
    'LapNumber',
    'TyreLife',
    'CompoundID',
    'Stint',
    'PitStop',
    'YellowFlag',
    'RedFlag',
    'SC',
    'VSC',
    "Rainfall",
    ]

cons = [
    'TrackID',
    'DriverID',
    'TeamID',    
    'Year',
]

lap_cols = [
    "LapNumber",
    "TyreLife",
    "CompoundID",
    ]

events = [
    "PitStop",
    "YellowFlag",
    "RedFlag",
    "SC",
    "VSC",
    'Rainfall',
    ]

time_cols = [
    ]

ids = [
    'TrackID',
    'DriverID',
    'TeamID',    
    'Year',
    ]

weather_cols = [
    'AirTemp',
    'Humidity',
    'Pressure',
    'TrackTemp',
    'WindDirection',
    'WindSpeed'
    ]

if __name__ == "__main__":
    filename = "outputs/racenet_branched_12_08_11_14_3l_640.pt"
    model = RaceNetBranched(args, num_drivers=26, num_tracks=27, num_teams=11)
    model_state_dict = torch.load(filename)
    model.load_state_dict(model_state_dict)
    model.eval()

    data = pd.read_hdf("data/f1_dataset.h5")
    dataset = F1Dataset(data)

    num_laps = 50

    rng = np.random.default_rng(228)
    lap_times = np.zeros((num_laps,2))
    idx = rng.integers(len(dataset))

    cons = torch.tensor([dataset[idx][1]])
    weath = torch.tensor([dataset[idx][2]])
    lap = torch.tensor([[2,2,2]])
    events = torch.tensor([dataset[idx][3]])    
    breakpoint()
    lap_time = model(lap,cons,weath,events).item()

    for i in tqdm(range(num_laps)):
        lap[0,0:2] += 1
        #lap[0,-1] = lap_time
        delt = 1
        next_lap_time = model(lap,cons,weath,events).item()
        r = lap_time - next_lap_time
        lap_times[i,0] = lap_time
        lap_times[i,1] = r        
        lap_time = next_lap_time

    print(lap_times)
    plt.plot([i+1 for i in range(num_laps)],lap_times[:,0],'r')
    plt.plot([i+1 for i in range(num_laps)],lap_times[:,1],'b')
    plt.show()