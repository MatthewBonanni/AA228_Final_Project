import pandas as pd
import time
import numpy as np
import scipy.sparse as sparse

from learning.q_learning import QLearn, QLambda
from learning.model_gen import PolicyGen

def main():
    data = pd.read_hdf("data/f1_dataset.h5")
    tracks = [4,8,10,11,13,16]
    for track in tracks:
        track_names = np.loadtxt('data/track_ids.txt',dtype=str,delimiter=',')
        track_name = track_names[track,1][:-11].replace(' ','_')
        print("Gen Transition model for:", track_name)
        polgen = PolicyGen(data,track)
        T,R = polgen.gen_model()
        #print("Q_learning for:", track_name)
        #qlearn = QLambda(data,track=track)
        #policy = qlearn.solve(100)
        #out_data = {"policy":policy, 
        #            "ravel":qlearn.ravel_shape, 
        #            "start_tire":qlearn.start_tire}
        #filename = "learning/policy/q_learn_track_"+str(track)
        #np.savez(filename, out_data)

if __name__ == "__main__":
    main()