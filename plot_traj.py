import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

data = np.load('data/rollout_traj.npz')
traj_q = data['traj_q']
traj_opt = data['traj_opt']

# Get Real Traj
track_id = 13
driver_id = 3
race_data = pd.read_hdf('data/f1_dataset.h5')
race_data = race_data[race_data["TrackID"]==track_id]
race_data = race_data[race_data["Year"]==2022]
race_data = race_data[race_data["DriverID"]==driver_id]

race_data["Reward"] = -(race_data.loc[:,"LapTime"] - race_data.loc[:,"PrevLapTime"])
race_data.loc[:,"Reward"] = [i.total_seconds() for i in race_data.loc[:,"Reward"]]
race_data.loc[:,"Reward"] = race_data.loc[:,"Reward"] - 30*race_data.loc[:,"PitStop"] 

race_data.loc[:,"LapTime"] = [i.total_seconds() for i in race_data.loc[:,"LapTime"]]

traj_real = race_data.loc[:,["LapNumber",
                             "LapTime",
                             "Reward",
                             "CompoundID",
                             "PitStop",
                             "YellowFlag",
                             "RedFlag",
                             "SC",
                             "VSC",
                             "Rainfall"]].to_numpy(dtype=float)[1:,:].T


events = traj_opt[-5]

pl.figure("Total_Time")
pl.plot(traj_opt[0,:], np.cumsum(traj_opt[1,:]), 'r', label="Hooke-Jeeves Optimization")
pl.plot(traj_q[0,:], np.cumsum(traj_q[1,:]), 'b',label="Q-Learning")
pl.plot(traj_real[0,:], np.cumsum(traj_real[1,:]), 'g',label="Actual Race Winner")

pl.plot(traj_opt[0,:][traj_opt[-6,:].nonzero()], np.cumsum(traj_opt[1,:])[traj_opt[-6,:].nonzero()], 'ro', label="Pit Stop")
pl.plot(traj_q[0,:][traj_q[-6,:].nonzero()], np.cumsum(traj_q[1,:])[traj_q[-6,:].nonzero()], 'bo')
pl.plot(traj_real[0,:][traj_real[-6,:].nonzero()], np.cumsum(traj_real[1,:])[traj_real[-6,:].nonzero()], 'bo')

pl.plot(traj_q[0,:][traj_q[-5,:].nonzero()], np.cumsum(traj_q[1,:])[traj_q[-5,:].nonzero()], 'y*', label="Yellow Flag")
pl.plot(traj_q[0,:][traj_q[-3,:].nonzero()], np.cumsum(traj_q[1,:])[traj_q[-3,:].nonzero()], 'ys', label="Safety Car")
pl.plot(traj_q[0,:][traj_q[-2,:].nonzero()], np.cumsum(traj_q[1,:])[traj_q[-2,:].nonzero()], 'ys')

pl.ylabel("Cumulative Time (s)")
pl.xlabel("Lap Number")
pl.title("Cumulative Lap Time")
pl.legend()


pl.figure("Lap_Time")
pl.plot(traj_opt[0,:], traj_opt[1,:], 'r', label="Hooke-Jeeves Optimization")
pl.plot(traj_q[0,:], traj_q[1,:], 'b', label="Q-Learning")
pl.plot(traj_real[0,:], traj_real[1,:], 'g', label="Real Driver")

pl.plot(traj_opt[0,:][traj_opt[-6,:].nonzero()], traj_opt[1,:][traj_opt[-6,:].nonzero()], 'ro', label="Pit Stop")
pl.plot(traj_q[0,:][traj_q[-6,:].nonzero()], traj_q[1,:][traj_q[-6,:].nonzero()], 'bo')
pl.plot(traj_real[0,:][traj_real[-6,:].nonzero()], traj_real[1,:][traj_real[-6,:].nonzero()], 'go',)

pl.plot(traj_q[0,:][traj_q[-5,:].nonzero()], traj_q[1,:][traj_q[-5,:].nonzero()], 'y*', label="Yellow Flag")
pl.plot(traj_q[0,:][traj_q[-3,:].nonzero()], traj_q[1,:][traj_q[-3,:].nonzero()], 'ys', label="Safety Car")
pl.plot(traj_q[0,:][traj_q[-2,:].nonzero()], traj_q[1,:][traj_q[-2,:].nonzero()], 'ys')

pl.ylabel("Lap Time (s)")
pl.xlabel("Lap Number")
pl.title("Lap Times")
pl.legend()

pl.figure("Reward")
pl.plot(traj_opt[0,:], traj_opt[2,:], 'r', label="Hooke-Jeeves Optimization")
pl.plot(traj_q[0,:], traj_q[2,:], 'b', label="Q-Learning")
pl.plot(traj_real[0,:], traj_real[2,:], 'g', label="Real Driver")

pl.plot(traj_opt[0,:][traj_opt[-6,:].nonzero()], traj_opt[2,:][traj_opt[-6,:].nonzero()], 'ro', label="Pit Stop")
pl.plot(traj_q[0,:][traj_q[-6,:].nonzero()], traj_q[2,:][traj_q[-6,:].nonzero()], 'bo', )
pl.plot(traj_real[0,:][traj_real[-6,:].nonzero()], traj_real[2,:][traj_real[-6,:].nonzero()], 'go', )

pl.plot(traj_q[0,:][traj_q[-5,:].nonzero()], traj_q[2,:][traj_q[-5,:].nonzero()], 'y*', label="Yellow Flag")
pl.plot(traj_q[0,:][traj_q[-3,:].nonzero()], traj_q[2,:][traj_q[-3,:].nonzero()], 'ys', label="Safety Car")
pl.plot(traj_q[0,:][traj_q[-2,:].nonzero()], traj_q[2,:][traj_q[-2,:].nonzero()], 'ys')

pl.ylabel("Reward")
pl.xlabel("Lap Number")
pl.title("Reward Along Trajectory")
pl.legend()

pl.figure("Tires")

pl.plot(traj_opt[0,:], traj_opt[3,:], 'r', label="Hooke-Jeeves Optimization")
pl.plot(traj_q[0,:], traj_q[3,:], 'b', label="Q-Learning")
pl.plot(traj_real[0,:], traj_real[3,:], 'g', label="Real Driver")

pl.ylabel("Tire ID")
pl.xlabel("Lap Number")
pl.yticks([0,1,2],["Soft","Med","Hard"])
pl.title("Tire Choice")
pl.legend()



pl.show()