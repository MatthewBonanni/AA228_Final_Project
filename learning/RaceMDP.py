import pdb
from copy import deepcopy
import numpy as np
import torch

from estimator.RaceNet import RaceNetBranched
from .state import RaceState
from .strategy import Policy

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

        self.t_i = 0.0
    
    def set_init_state(self,
                       state : RaceState) -> None:
        self.init_state = deepcopy(state)
    
    def set_state(self,
                  state : RaceState) -> None:
        self.state = deepcopy(state)
    
    def reset_state(self) -> None:
        self.state = deepcopy(self.init_state)
    
    def __eval_NN(self) -> None:
        self.t_i = self.NN(*self.state.to_NN_args())
    
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
    
    def transition(self,
                   action : int) -> None:
        self.state.t_im1 = self.t_i
        self.state.lap_number += 1
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
            for i in range(depth):
                action = policy.eval(self.state)
                self.transition(action)
                r = self.reward(action)
                ret += self.gamma**(i-1) * r
        
        if reset:
            self.reset_state()

        return ret/num_rollouts