import os
import pdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from estimator.RaceNet import RaceNetBranched, F1Dataset
from learning.RaceMDP import *

args = {
    'num_layers': 3,
    'hidden_dim': 640,
    'out_dim': 128,
    'emb_dim': 10,
    'dropout': 0.4}
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
    "Rainfall"]
lap_cols = [
    "LapNumber",
    "TyreLife",
    "CompoundID",
    "Stint",
    "PitStop",
    "YellowFlag",
    "RedFlag",
    "SC",
    "VSC",
    'Rainfall']
time_cols = [
    "PrevLapTime"]
ids = [
    'TrackID',
    'DriverID',
    'TeamID']
weather_cols = [
    'AirTemp',
    'Humidity',
    'Pressure',
    'TrackTemp',
    'WindDirection',
    'WindSpeed']

def main():
    filename = "outputs/racenet_branch_12_05_22_26_3l_640_pred_scale_w_stint.pt"
    model = RaceNetBranched(args, num_drivers=26, num_tracks=27, num_teams=11)
    model_state_dict = torch.load(filename)
    model.load_state_dict(model_state_dict)
    model.eval()

    data = pd.read_hdf("data/f1_dataset.h5")
    dataset = F1Dataset(data)

    num_laps = 50

    mdp = RaceMDP(model, gamma=0.9)

    # Set up initial state
    events = RaceEvents(pit_stop=False,
                        yellow_flag=False,
                        red_flag=False,
                        safety_car=False,
                        virtual_safety_car=False,
                        rainfall=False)
    weather = RaceWeather(air_temp=25.0,
                          humidity=55.0,
                          pressure=980.0,
                          track_temp=35.0,
                          wind_direction=0.0,
                          wind_speed=1.0)
    consts = RaceConstants(year=23,
                           stint=1,
                           track_id=13,
                           driver_id=3,
                           team_id=3)
    state = RaceState(t_im1=0,
                      tire_age=1,
                      lap_number=1,
                      tire_id=0,
                      events=events,
                      weather=weather,
                      constants=consts)
    
    mdp.set_state(state)

    # Hooke-Jeeves
    policy = AgeBasedRandomTirePolicy(
        [10, 10, 10, 10, 10])
    
    U = mdp.rollout(policy, depth=10)

    breakpoint()

if __name__ == "__main__":
    main()