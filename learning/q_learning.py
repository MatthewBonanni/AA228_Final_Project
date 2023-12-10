import pandas as pd
import numpy as np
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

class PolicyGenMF():
    def __init__(self, data, track, num_states=None, disc=0.95):
        race = data[data["TrackID"] == track]
        #race = race[race["PrevLapTime"] > pd.Timedelta(0)]
        race.loc[:,"PrevLapTime"] = race.loc[:,"PrevLapTime"].mask(race.loc[:,"PrevLapTime"] == pd.Timedelta(0),
                                                        race.loc[:,"LapTime"])
        
        self.start_tire = race.loc[:,"CompoundID"][race.loc[:,"LapNumber"]==1].mean()
        self.start_tire = round(self.start_tire)
        cols = lap_cols+events
        num_laps = race["LapNumber"].max()+1
        num_tires = 5
        num_events = len(events)
        self.disc = disc

        lt = np.array([i.total_seconds() for i in race.loc[:,"LapTime"]])
        prev_lt = np.array([i.total_seconds() for i in race.loc[:,"PrevLapTime"]])
        self.reward = -(lt-prev_lt)

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
                self.reward[i] = self.reward[i] - 10
            else:
                self.action[i] = 0

        self.ravel_shape = np.array([num_laps,num_laps,num_tires] + [2 for i in range(num_events)],dtype=int)
        self.state = np.ravel_multi_index(state.T, self.ravel_shape)
        self.state_next = np.ravel_multi_index(next_state.T, self.ravel_shape)
        
        self.num_states = int((num_laps**2)*num_tires*(2**num_events))
        self.num_action = num_tires + 1       

    def solve():
        raise NotImplementedError("Needs to be Implemented")
    
class QLearn(PolicyGenMF):

    def approx_q(self,start_s,end_s,Q,q):
        Q_row = []
        Q_col = []
        Q_dat = []
        idx = np.arange(start_s, end_s, dtype=int)

        s=self.state[idx]; sp = self.state_next[idx]; a = self.action[idx]
        r=self.reward[idx]
        Q_sa = Q[(s),(a)]
        Q_sap = Q_sa + self.alpha*(r + self.disc*(Q[sp,:].max(axis=1).todense().flatten()) - Q_sa)

        out_ids = np.abs(Q_sap[:]) > 1e-9
        Q_row.extend(s[out_ids])
        Q_col.extend(a[out_ids])
        Q_dat.extend(Q_sap[out_ids])  

        q.put([Q_dat, Q_row, Q_col], 10)

    def q_update(self, iter):
        Q = self.Q.copy()
        Q_prev = self.Q.copy()
        err_prev = 0

        # Estimate Q functions
        # Use Threaded fn
        n_threads = self.state.shape[0]//(50000-1) + 1

        q = Queue()
        for it in tqdm(range(iter)):
            procs=[]

            for i in range(n_threads):
                start_s = i*(self.state.shape[0]//n_threads)
                if i < n_threads-1:
                    end_s = start_s + (self.state.shape[0]//n_threads)
                else:
                    end_s = self.state.shape[0]
                procs.append(Process(target=self.approx_q,args=(start_s,end_s,Q.copy(),q)))

            for proc in procs:
                proc.start()

        
            Q_row = []
            Q_col = []
            Q_dat = []
            i = 0
            while(any(proc.is_alive() for proc in procs)):
                time.sleep(1)
                ret = q.get(1)
                if ret:
                    i += 1
                    Q_dat.extend(ret[0])
                    Q_row.extend(ret[1])
                    Q_col.extend(ret[2])
                if i == n_threads:
                    for proc in procs:
                        proc.terminate()
                    break
            
            #print("MAKING SPARSE ARRAY")
            # Check for duplicates
            Q_dat, Q_ind = self.fix_duplicates(Q_dat, Q_row, Q_col)             

            Q = sparse.csr_array((Q_dat,(Q_ind[0],Q_ind[1])), 
                                      shape=(self.num_states,self.num_action), dtype=np.float32)
            
            Q_err = (np.abs(Q - Q_prev)).mean()
            Q_prev = Q.copy()
            if np.abs(Q_err-err_prev) < 1e-4*Q_err:
                print("Iters:", it, " err:", Q_err)
                break
            else:
                err_prev = Q_err
                #print("err:", Q_err)
        return Q
            
    def fix_duplicates(self, data, row, col, method='mean'):
        dat_in = np.asarray(data).flatten()
        
        dat_out = []
        indices = []

        in_indx = np.vstack((row,col))
        order = np.lexsort((col,row))
        in_indx = in_indx[:,order]
        dat_in = dat_in[order]

        for id in range(dat_in.shape[0]):
            s = in_indx[0,id]; a = in_indx[1,id]
            if (s,a) not in indices:
                mask = np.logical_and((in_indx[0]==s), (in_indx[1]==a))
                if method == 'mean':
                    val = np.mean(dat_in[mask])
                elif method == 'sum':
                    val = np.sum(dat_in[mask])
                else:
                    raise ValueError("Choose a valid method.")
                
                dat_out.append(val)
                indices.append((s,a))
        return np.asarray(dat_out), np.asarray(indices).T

    def greedy_pol(self):
        pi = self.Q.argmax(axis=1).flatten()
        return pi


    def solve(self, iter, alpha=None):
        if alpha == None:
            self.alpha = 1/(np.power(self.state.shape[0],0.5))
        else:
            self.alpha=alpha

        # Init Q to zeros
        self.Q = sparse.csr_array(([0],([0],[0])), shape=(self.num_states,self.num_action), dtype=np.float32)
        # Update Q for n-iterations
        self.Q = self.q_update(iter)
        self.start_tire = 2
        max_Q = -np.Inf
        for i in range(self.num_action-1):
            state = np.array([[2,2,i,0,0,0,0,0,0]]).T
            Q_tire = self.Q[np.ravel_multi_index(state,self.ravel_shape),[0]].item()
            if  Q_tire > max_Q:
                self.start_tire = i
                max_Q = Q_tire
        breakpoint()

        pi = self.greedy_pol()

        return pi
    
class QLambda(QLearn):

    def approx_q(self,start_s,end_s,Q,N,q):
        Q_row = []
        Q_col = []
        Q_dat = []

        N_row = []
        N_col = []
        N_dat = []

        idx = np.arange(start_s, end_s, dtype=int)

        s=self.state[idx]; sp = self.state_next[idx]; a = self.action[idx]
        r=self.reward[idx]
        Q_sa = Q[(s),(a)]; N_sa = N[(s),(a)]+1
        Q_sap = Q_sa + self.alpha*(r + self.disc*Q[(sp),:].max(axis=1).todense().flatten() - Q_sa)*N_sa
        
        if N_sa.any():
            N_row.extend(s); N_col.extend(a), N_dat.extend(np.ones(idx.shape)*self.alpha*self.lmda)

        out_ids = np.abs(Q_sap[:]) > 1e-6
        Q_row.extend(s[out_ids])
        Q_col.extend(a[out_ids])
        Q_dat.extend(Q_sap[out_ids])

        q.put([[Q_dat, Q_row, Q_col],[N_dat, N_row, N_col]], 10)

    def q_update(self, iter):
        Q = self.Q.copy()
        Q_prev = self.Q.copy()
        N_prev = self.N.copy()
        err_prev = 0.0

        # Estimate Q functions
        # Use Threaded fn
        n_threads = self.state.shape[0]//(50000-1)+1
        q = Queue()
        for it in tqdm(range(iter)):
            procs=[]

            for i in range(n_threads):
                start_s = i*(self.state.shape[0]//n_threads)
                if i < n_threads-1:
                    end_s = start_s + (self.state.shape[0]//n_threads)
                else:
                    end_s = self.state.shape[0]
                procs.append(Process(target=self.approx_q,args=(start_s,end_s,Q.copy(),N_prev.copy(),q)))

            for proc in procs:
                proc.start()

        
            Q_row = []
            Q_col = []
            Q_dat = []

            N_row = []
            N_col = []
            N_dat = []
            i = 0
            while(any(proc.is_alive() for proc in procs)):
                time.sleep(1)
                ret = q.get(1)
                if ret:
                    i += 1
                    Q_dat.extend(ret[0][0])
                    Q_row.extend(ret[0][1])
                    Q_col.extend(ret[0][2])

                    N_dat.extend(ret[1][0])
                    N_row.extend(ret[1][1])
                    N_col.extend(ret[1][2])

                if i == n_threads:
                    for proc in procs:
                        proc.terminate()
                    break
            # same every iteration bc we're processing all the data at once.
            if N_prev.nnz < 2:
                N_dat, N_ind = self.fix_duplicates(N_dat, N_row, N_col,method='sum')             
                N = sparse.csr_array((N_dat,(N_ind[0],N_ind[1])), 
                                        shape=(self.num_states,self.num_action), dtype=np.float32)
                N_prev = N + N_prev*self.alpha*self.lmda
            else:
                N_prev = N_prev*(1+self.alpha*self.lmda)

            Q_dat, Q_ind = self.fix_duplicates(Q_dat, Q_row, Q_col)
            Q = sparse.csr_array((Q_dat,(Q_ind[0],Q_ind[1])), 
                                      shape=(self.num_states,self.num_action), dtype=np.float32)
            
            
            Q_err = (np.abs(Q - Q_prev)).mean()
            
            Q_prev = Q.copy()
            if np.abs(Q_err-err_prev) < 1e-4*Q_err:
                print("Iters:", it, " err:", Q_err)
                break
            else:
                err_prev = Q_err
                #print("err:", Q_err)
        self.N = N
        return Q


    def solve(self, iter, alpha=None, lmda=0.95):
        if alpha == None:
            self.alpha = 1/(np.power(self.state.shape[0],0.5))
        else:
            self.alpha=alpha
        self.lmda=lmda

        # Init Q and N to zeros
        self.Q = sparse.csr_array(([0],([0],[0])), shape=(self.num_states,self.num_action), dtype=np.float32)
        self.N = sparse.csr_array(([0],([0],[0])), shape=(self.num_states,self.num_action), dtype=np.float32)
        # Update Q for n-iterations
        self.Q = self.q_update(iter)

        state = np.repeat(np.array([[2,2,0,0,0,0,0,0,0]]),self.num_action-1, axis=0)
        for i in range(self.num_action-1):
            state[i,2] = i

        Q_tire = self.Q[np.ravel_multi_index(state.T,self.ravel_shape),[0]]
        try:
            self.start_tire = Q_tire.nonzero()[0][np.argmax(Q_tire[Q_tire.nonzero()])]
        except:
            pass
        pi = self.greedy_pol()
        return pi