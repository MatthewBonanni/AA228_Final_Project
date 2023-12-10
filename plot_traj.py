import numpy as np
import matplotlib.pyplot as pl

data = np.load('data/rollout_traj.npz')
traj_q = data['traj_q']
traj_opt = data['traj_opt']
events = traj_opt[-5]

pl.figure("Total Time")
pl.plot(traj_opt[0,:], np.cumsum(traj_opt[1,:]), 'r', label="Hooke-Jeeves Optimization")
pl.plot(traj_q[0,:], np.cumsum(traj_q[1,:]), 'b',label="Q-Learning")

pl.plot(traj_opt[0,:][traj_opt[3,:].nonzero()], np.cumsum(traj_opt[1,:])[traj_opt[3,:].nonzero()], 'ro')
pl.plot(traj_q[0,:][traj_q[3,:].nonzero()], np.cumsum(traj_q[1,:])[traj_q[3,:].nonzero()], 'bo')
pl.ylabel("Cumulative Time (s)")
pl.xlabel("Lap Number")
pl.legend()


pl.figure("Lap Time")
pl.plot(traj_opt[0,:], traj_opt[1,:], 'r', label="Hooke-Jeeves Optimization")
pl.plot(traj_q[0,:], traj_q[1,:], 'b', label="Q-Learning")

pl.plot(traj_opt[0,:][traj_opt[3,:].nonzero()], traj_opt[1,:][traj_opt[3,:].nonzero()], 'ro')
pl.plot(traj_q[0,:][traj_q[3,:].nonzero()], traj_q[1,:][traj_q[3,:].nonzero()], 'bo')

pl.ylabel("Lap Time (s)")
pl.xlabel("Lap Number")
pl.legend()

pl.show()