import numpy as np
import pandas as pd
import scipy.sparse as sparse

from tqdm import tqdm

import time

from multiprocessing import Process, Queue

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
class PolicyGen():
    def __init__(self, data, track, num_states=None, disc=0.95):
        race = data[data["TrackID"] == track]
        race = race[race["PrevLapTime"] > pd.Timedelta(0)]
        track_names = np.loadtxt('data/track_ids.txt',dtype=str,delimiter=',')
        self.track = str(track)
        cols = lap_cols + events
        num_laps = race["LapNumber"].max()+1
        num_tires = 5
        num_events = len(events)
        self.disc = disc

        lt = np.array([i.total_seconds() for i in race.loc[:,"LapTime"]])
        prev_lt = np.array([i.total_seconds() for i in race.loc[:,"PrevLapTime"]])
        self.reward = lt-prev_lt

        state = race.loc[:,cols].to_numpy(dtype=int)
        next_state = np.zeros(state.shape,dtype=int)
        for i in range(state.shape[0]-1):
            ns = state[i+1]
            if ns[0] == state[i,0]+1:
                next_state[i] = ns

        mask = [(next_state[i,:]==0).all() for i in range(next_state.shape[0])] 
        mask = np.array(mask,dtype=int)-1
        state = state[mask.nonzero()]
        next_state = next_state[mask.nonzero()]
        self.reward = self.reward[mask.nonzero()]

        self.action = np.zeros((state.shape[0],))
        for i in range(next_state.shape[0]):
            s = next_state[i,:]
            if s[cols.index("PitStop")] == 1:
                self.action[i] = s[cols.index("CompoundID")] + 1
                self.reward[i] = self.reward[i] - 30
            else:
                self.action[i] = 0

        state=state[:,len(lap_cols):]
        next_state=next_state[:,len(lap_cols):]

        self.ravel_shape = np.array([2 for i in range(num_events)],dtype=int)
        
        self.state = np.ravel_multi_index(state.T, self.ravel_shape)
        self.state_next = np.ravel_multi_index(next_state.T, self.ravel_shape)
        self.num_states = int((2**num_events))
        self.num_action = num_tires + 1       

        self.U = np.zeros((self.num_states,))
        self.Pi = np.zeros((self.num_states,), dtype=int)

    ## children need to implement this
    def solve(self):
        raise NotImplementedError("Implement this!")
    
    def maximum_value_approx(self, start_s, end_s, q):
        T_row = []
        T_col = []
        T_data = []

        R_row = []
        R_col = []
        R_data = []

        state = self.state.copy()
        state_next = self.state_next.copy()
        action = self.action.copy()
        reward = self.reward.copy()
        n_action = self.num_action

        for s in tqdm(range(start_s,end_s)):
            for a in range(n_action):
                for sp in state_next[np.logical_and(state==s, action==a)][:]:
                    if (sp in T_row and ((s)*n_action+a) in T_col):
                        continue
                    T_data.append(state_next[np.logical_and(action==a,np.logical_and(state==s,state_next==sp))].shape[0])
                    T_data[-1]=T_data[-1]/state_next[np.logical_and(action==a,state==s)].shape[0]
                    T_row.append(sp)
                    T_col.append((s)*n_action+a)
                R_s_a = np.sum(reward[np.logical_and(state==s, action==a)])
                if not R_s_a==0:
                    R_data.append(R_s_a/state[np.logical_and(state==s, action==a)].shape[0])
                    R_row.append(s)
                    R_col.append(a)
        
        q.put([[T_data, T_row, T_col],[R_data, R_row, R_col]], 10)

    def gen_model(self):
    
        # Estimate Transition and Reward functions
        # Use Threaded fn
        n_threads = 4
        procs=[]
        q = Queue()
        
        for i in range(n_threads):
            start_s = i*(self.num_states//n_threads)
            if i < n_threads-1:
                end_s = start_s + (self.num_states//n_threads)
            else:
                end_s = self.num_states 
            procs.append(Process(target=self.maximum_value_approx,args=(start_s,end_s,q)))

        for proc in procs:
            proc.start()
        
        T_data = []; T_row = []; T_col = []
        R_data = []; R_row = []; R_col = [] 
        i = 0
        while(any(proc.is_alive() for proc in procs)):
            time.sleep(1)
            ret = q.get(1)
            if ret:
                i += 1
                T_data.extend(ret[0][0])
                T_row.extend(ret[0][1])
                T_col.extend(ret[0][2])

                R_data.extend(ret[1][0])
                R_row.extend(ret[1][1])
                R_col.extend(ret[1][2])
            if i == n_threads:
                q.close()
                for proc in procs:
                    proc.terminate()
                break

        print("MAKING SPARSE ARRAY")
        T = sparse.csr_array((np.asarray(T_data),(np.asarray(T_row),np.asarray(T_col))),
                            shape=(self.num_states,self.num_states*self.num_action), dtype=np.float32,)
        R = sparse.csr_array((np.asarray(R_data),(np.asarray(R_row),np.asarray(R_col))),
                            shape=(self.num_states,self.num_action), dtype=np.float32,)
        
        sparse.save_npz("data/"+self.track+"_T_fn.npz", T)
        sparse.save_npz("data/"+self.track+"_R_fn.npz", R)

        return T, R
    
class GreedyMB(PolicyGen):

    def gen_model(self):
        try:
            T = sparse.load_npz("data/"+self.track+"_T_fn.npz")
            R = sparse.load_npz("data/"+self.track+"_R_fn.npz")
        except:
            super().gen_model()
        return T, R

    def bellman_update_utility(self, T, R, iters):
        sort = np.argsort(np.asarray(np.sum(R,axis=1)).flatten())
        for i in range(iters):
            sort = np.flip(sort.flatten())
            for s in tqdm(sort):
                U_s = R[[s],:] + self.disc*\
                    np.sum(T[:,(s)*self.num_action:(s+1)*self.num_action].multiply(\
                    self.U.reshape((self.num_states,1))),axis=0)
                self.U[s] = np.max(U_s)
            print(np.max(self.U))
            sort = np.argsort(self.U)

    def policy(self,T,R):
        for s in tqdm(range(self.num_states)):
            U_s = R[[s],:] + self.disc*\
                    np.sum(T[:,(s)*self.num_action:(s+1)*self.num_action].multiply(\
                    self.U.reshape((self.num_states,1))),axis=0)
            self.Pi[s] = np.argmax(U_s)+1

    def solve(self, iter):
        
        T, R = self.gen_model()
        print("STARTING ITERS")        
        self.bellman_update_utility(T, R, iter)
        self.policy(T,R)

        return self.Pi, self.U
    