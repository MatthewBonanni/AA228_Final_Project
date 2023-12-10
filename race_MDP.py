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
    "CompoundID",]

events = [
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
    track_id = 13
    filename = "learning/policy/q_pols/q_learn_track_"+str(track_id)+".npz"
    if os.path.exists(filename):
        in_data = np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Track hasn't been trained for Q Learn.")
    
    # Load events trajectory
    filename = "data/"+str(track_id)+"_2022_events.npz"
    events_traj = np.load(filename, allow_pickle=True)['events_traj'][:,1:]

    num_laps = events_traj.shape[-1]
    start_tire = in_data["start_tire"].item()

    mdp = RaceMDP(model,
                  gamma=1.0,
                  t_p_estimate=30.0,
                  num_laps=num_laps)

    # Set up initial state
    events = RaceEvents(pit_stop=False,
                        yellow_flag=False,
                        red_flag=False,
                        safety_car=False,
                        virtual_safety_car=False,
                        rainfall=False)
    weather = RaceWeather(air_temp=28.2,
                          humidity=31.28,
                          pressure=993.05,
                          track_temp=43.32,
                          wind_direction=186.72,
                          wind_speed=1.59)
    consts = RaceConstants(year=22,
                           stint=1,
                           track_id=track_id,
                           driver_id=3,
                           team_id=3)
    state = RaceState(t_im1=500,
                      tire_age=1,
                      lap_number=1,
                      tire_id=start_tire,
                      tire_ids_used=[],
                      events=events,
                      weather=weather,
                      constants=consts)
    
    mdp.set_init_state(state)
    mdp.set_state(state)

    q_policy = QLearnPolicy(track_id)
    U_q = mdp.mc_rollout(q_policy, num_laps, 5, reset=True)
    U_q_event, traj_q = mdp.traj_rollout(q_policy, depth=num_laps,
                                  in_events=events_traj[-5:,:],
                                  reset=True)
    print("U, Q-Learn:", U_q)
    print("U, Q-Learn with Real Events:", U_q_event)

    state.tire_id = -1
    state.stint = 0
    mdp.set_init_state(state)
    mdp.set_state(state)

    policy = AgeSequencePolicy(
        np.array([[1, int(num_laps)//3, int(num_laps)//3],
                  [0, 0, 0]]))
    
    U_init = mdp.mc_rollout(policy, depth=num_laps, num_rollouts=10, reset=True)
    print("Initial U:", U_init)

    opt = HookeJeeves([32, 1], [[0, num_laps], [0, 2]], 5, 1, [2, 1])

    policy = opt.eval(policy, mdp.mc_rollout, [num_laps, 5, True])
    print("Final policy:") 
    print(policy.get_parameters())

    U_final = mdp.mc_rollout(policy, depth=num_laps, num_rollouts=10, reset=True)
    print("Final U:", U_final)
    U_opt_event, traj_opt = mdp.traj_rollout(policy, depth=num_laps,
                                 in_events = events_traj[-5:,:], 
                                 reset=True)
    print("U, Final Policy with Real Events:", U_opt_event)
    breakpoint()
    np.savez('data/rollout_traj', traj_q = traj_q, 
             U_q = U_q_event, 
             traj_opt = traj_opt, 
             U_opt = U_opt_event)



if __name__ == "__main__":
    main()