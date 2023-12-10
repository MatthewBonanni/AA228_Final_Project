import pdb
import os
from copy import deepcopy
import numpy as np
import torch
from typing import Union
import scipy.sparse as sparse

from estimator.RaceNet import RaceNetBranched
from .state import RaceState
from .strategy import Policy

rng = np.random.default_rng(228)

class RaceMDP():
    def __init__(self,
                 NN : torch.nn.Module,
                 gamma : float = 0.9,
                 t_p_estimate : float = 30.0,
                 num_laps : int = 50):
        self.NN = NN
        self.gamma = gamma
        self.t_p_estimate = t_p_estimate
        self.num_laps = num_laps
        self.init_state = None
        self.state = None
        num_events = 6
        self.event_ravel=np.array([2 for i in range(6)])
        self.num_actions = 6

        self.t_i = 0.0
    
    def set_init_state(self,
                       state : RaceState) -> None:
        self.init_state = deepcopy(state)
        self.t_i = state.t_im1
        track_id = self.init_state.constants.track_id
        filename = "data/"+str(track_id)+"_T_fn.npz"
        if os.path.exists(filename):
            self.T = sparse.load_npz(filename)
        else:
            self.T = None
    
    def set_state(self,
                  state : RaceState) -> None:
        self.state = deepcopy(state)
        if self.init_state == None:
            self.set_init_state(state)
    
    def reset_state(self) -> None:
        if self.init_state is not None:
            self.state = deepcopy(self.init_state)
            self.t_i = self.init_state.t_im1
        else:
            raise ValueError("No Initial State Set")
    
    def __eval_NN(self) -> None:
        with torch.no_grad():
            self.t_i = self.NN(*self.state.to_NN_args()).item()
    
    def pit_time(self,
                 action : int) -> float:
        if action > 0:
            return self.t_p_estimate
        return 0.0
    
    def reward(self,
               action : int) -> float:
        self.__eval_NN()

        # Strongly penalize if we didn't use at least 2 types of tires
        if ((self.state.lap_number >= self.num_laps) and
            len(np.unique(self.state.tire_ids_used + [self.state.tire_id])) < 2):
            # print("Not enough tires...")
            # breakpoint()
            return -1e6

        return -1 * ((self.t_i - self.state.t_im1) +
                     self.pit_time(action))

    def get_event_state_int(self, action:int):
        event_state = self.state.events.to_array().astype(int)
        event_state = event_state.reshape((1,event_state.shape[0]))
        event_state_int = np.ravel_multi_index(event_state.T,self.event_ravel)
        return event_state_int*self.num_actions + action

    def next_event_state(self, action:int):
        event_state_int = self.get_event_state_int(action)
        T_probs = self.T[:,event_state_int].toarray().squeeze()
        if T_probs.sum() == 0:
            #print(self.state.events.to_list(), "a:", action)
            T_probs = self.T[:,event_state_int-action].toarray().squeeze()
        if T_probs.sum() == 0: #Check again if there is not a known transition
            T_probs[(event_state_int-action)//self.num_actions] = 1.0
        T_probs = T_probs/T_probs.sum()
        next_event = rng.choice(T_probs.shape[0],p=T_probs)
        next_event = np.array(np.unravel_index(next_event,self.event_ravel))
        if action > 0:
            next_event[0] = 1
        else:
            next_event[0] = 0
        return next_event

    
    def transition(self,
                   action : int) -> None:
        self.state.t_im1 = self.t_i
        self.state.lap_number += 1
        if self.T is not None:
            next_event = self.next_event_state(action)
            self.state.events.set_state(next_event)      
        if action > 0:
            self.state.tire_ids_used += [self.state.tire_id]
            self.state.tire_id = action-1
            self.state.constants.stint += 1
            self.state.tire_age = 1
            self.state.events.pit_stop = True
        else:
            self.state.tire_age += 1
    
    def rollout(self,
                policy : Policy,
                depth : int,
                reset : bool = True) -> float:
        ret = 0.0
        for i in range(depth):
            action = policy.eval(self.state)
            self.transition(action)
            r = self.reward(action)
            ret += self.gamma**(i-1) * r
        
        if reset:
            self.reset_state()

        return ret
    
    def mc_rollout(self,
                policy : Policy,
                depth : int,
                num_rollouts : int,
                reset : bool = True) -> float:
        ret = 0.0
        for m in range(num_rollouts):
            self.reset_state()
            for i in range(depth):
                action = policy.eval(self.state)
                self.transition(action)
                r = self.reward(action)
                ret += self.gamma**(i-1) * r
        
        if reset:
            self.reset_state()
        return ret/num_rollouts
    
    def traj_rollout(self,
                policy : Policy,
                depth : int,
                in_events : Union[np.ndarray,None] = None,
                reset : bool = True) -> np.ndarray:
        
        traj = np.zeros((9,depth))
        ret = 0.0
        for i in range(depth):
            action = policy.eval(self.state)
            self.transition(action)
            if in_events is not None:
                ps = 1 if action > 0 else 0
                self.state.events.set_state(
                    np.concatenate([[ps],in_events[:,i]],
                                   axis=0))
            r = self.reward(action)
            ret += self.gamma**(i-1) * r
            traj[:,i] = np.concatenate([[self.state.lap_number,
                    self.t_i, self.state.tire_id],
                    self.state.events.to_array()],axis=0)
        
        if reset:
            self.reset_state()

        return ret, traj