import os
import pdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from estimator.RaceNet import RaceNetBranched, F1Dataset
from learning.state import *
from learning.strategy import *
from learning.RaceMDP import *
from learning.optimizer import HookeJeeves

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
    filename = "outputs/racenet_branched_12_08_11_14_3l_640.pt"
    model = RaceNetBranched(args, num_drivers=26, num_tracks=27, num_teams=11)
    model_state_dict = torch.load(filename)
    model.load_state_dict(model_state_dict)
    model.eval()

    # data = pd.read_hdf("data/f1_dataset.h5")
    # dataset = F1Dataset(data)
    
    num_laps = 75

    mdp = RaceMDP(model, gamma=0.9, t_p_estimate=30.0, num_laps=num_laps)

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
    consts = RaceConstants(year=22,
                           stint=1,
                           track_id=13,
                           driver_id=3,
                           team_id=3)
    state = RaceState(t_im1=500,
                      tire_age=1,
                      lap_number=1,
                      tire_id=0,
                      tire_ids_used=[],
                      events=events,
                      weather=weather,
                      constants=consts)
    
    mdp.set_init_state(state)
    mdp.set_state(state)

    # policy = AgeBasedRandomTirePolicy(
    #     np.array([[30, 30, 30, 10, 10]]))
    policy = AgeSequencePolicy(
        np.array([[25, 25],
                  [1, 0]]))
    
    U_init = mdp.rollout(policy, depth=num_laps, reset=True)
    print("Initial U:", U_init.item())

    # opt = HookeJeeves([16], [[0, num_laps]], 100, 1, [2])
    opt = HookeJeeves([1, 1], [[0, num_laps], [0, 4]], 100, 1, [1, 1])

    policy = opt.eval(policy, mdp.rollout, [num_laps, True])
    print("Final policy:") 
    print(policy.get_parameters())

    U_final = mdp.rollout(policy, depth=num_laps, reset=True)
    print("Final U:", U_final.item())


if __name__ == "__main__":
    main()