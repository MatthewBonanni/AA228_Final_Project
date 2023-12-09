import pandas as pd
import time
import numpy as np
import scipy.sparse as sparse

from learning.q_learning import QLearn
from learning.model_gen import PolicyGen

def main():
    data = pd.read_hdf("data/f1_dataset.h5")
    track = 16
    track_names = np.loadtxt('data/track_ids.txt',dtype=str,delimiter=',')
    track_name = track_names[track,1][:-11].replace(' ','_')
    polgen = PolicyGen(data,track)
    T,R = polgen.gen_model()
    qlearn = QLearn(data,track=track)
    policy = qlearn.solve(100)
    filename = "learning/policy/q_learn_track_"+str(track)
    np.savez(filename, policy)
    breakpoint()

if __name__ == "__main__":
    main()