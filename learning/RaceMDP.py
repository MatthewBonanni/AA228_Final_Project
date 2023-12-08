import pdb
import numpy as np
import torch
from estimator.RaceNet import RaceNetBranched

N_A = 6

class RaceEvents():
    def __init__(self,
                 pit_stop           : bool,
                 yellow_flag        : bool,
                 red_flag           : bool,
                 safety_car         : bool,
                 virtual_safety_car : bool,
                 rainfall           : bool):
        self.pit_stop = pit_stop
        self.yellow_flag = yellow_flag
        self.red_flag = red_flag
        self.safety_car = safety_car
        self.virtual_safety_car = virtual_safety_car
        self.rainfall = rainfall
    
    def to_list(self) -> list:
        return [self.pit_stop,
                self.yellow_flag,
                self.red_flag,
                self.safety_car,
                self.virtual_safety_car,
                self.rainfall]
    
    def to_array(self) -> np.array:
        return np.array(self.to_list(), dtype=bool)
    
    def to_tensor(self) -> torch.tensor:
        arr = self.to_array()
        if arr.any():
            arr = np.concatenate([[0], arr])
        else:
            arr = np.concatenate([[1], arr])
        return torch.tensor([arr])

class RaceWeather():
    def __init__(self,
                 air_temp       : np.float16,
                 humidity       : np.float16,
                 pressure       : np.float16,
                 track_temp     : np.float16,
                 wind_direction : np.float16,
                 wind_speed     : np.float16):
        self.air_temp = air_temp
        self.humidity = humidity
        self.pressure = pressure
        self.track_temp = track_temp
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
    
    def to_list(self) -> list:
        return [self.air_temp,
                self.humidity,
                self.pressure,
                self.track_temp,
                self.wind_direction,
                self.wind_speed]
    
    def to_array(self) -> np.array:
        return np.array(self.to_list(), dtype=np.float16)
    
    def to_tensor(self) -> torch.tensor:
        return torch.tensor([self.to_list()])

class RaceConstants():
    def __init__(self,
                 year      : np.int8,
                 stint     : np.int8,
                 track_id  : np.int8,
                 driver_id : np.int8,
                 team_id   : np.int8):
        self.year = year
        self.stint = stint
        self.track_id = track_id
        self.driver_id = driver_id
        self.team_id = team_id
    
    def to_list(self) -> list:
        return [self.year,
                self.stint,
                self.track_id,
                self.driver_id,
                self.team_id]
    
    def to_array(self) -> np.array:
        return np.array(self.to_list(), dtype=np.int8)
    
    def to_tensor(self) -> torch.tensor:
        return torch.tensor([[0.0,
                              self.year-21,
                              self.stint]])
    
    def id_tensor(self) -> torch.tensor:
        return torch.tensor([[self.track_id,
                              self.driver_id,
                              self.team_id,
                              self.year-21,
                              0]])

class RaceState():
    def __init__(self,
                 t_im1 : float,
                 tire_age : int,
                 lap_number : int,
                 tire_id : int,
                 events : RaceEvents,
                 weather : RaceWeather,
                 constants : RaceConstants):
        self.t_im1 = t_im1
        self.tire_age = tire_age
        self.lap_number = lap_number
        self.tire_id = tire_id
        self.events = events
        self.weather = weather
        self.constants = constants
    
    def to_list(self) -> list:
        return [self.t_im1,
                self.tire_age,
                self.lap_number,
                self.tire_id,
                self.events,
                self.weather,
                self.constants]
    
    def to_NN_args(self):
        return [torch.tensor([[self.lap_number,
                               self.tire_age,
                               self.tire_id - 1]]),
                self.constants.id_tensor(),
                self.weather.to_tensor(),
                self.events.to_tensor()]

class Policy():
    def __init__(self):
        return
    
    def eval(self,
             mdp_state: RaceState) -> int:
        raise NotImplementedError()
    def get_state(self):
        raise NotImplementedError()
    def set_state(self, state: list):
        raise NotImplementedError()

class RandomPolicy():
    def eval(self,
             mdp_state : RaceState) -> int:
        return np.random.randint(6)

class AgeBasedRandomTirePolicy(Policy):
    def __init__(self,
                 pit_ages : list):
        self.pit_ages = pit_ages
    
    def eval(self,
             mdp_state: RaceState) -> int:
        tire_age = mdp_state.tire_age
        tire_id = mdp_state.tire_id
        rainfall = mdp_state.events.rainfall
        if tire_age > self.pit_ages[tire_id]:
            if rainfall: #Check if raining
                return np.random.randint(5,6+1)
            
            next_tire = np.random.randint(1,4+1)
            while next_tire == tire_id+1:
                next_tire = np.random.randint(1,4+1)
            return next_tire
        return 0
    
    def get_state(self):
        return np.array(self.pit_ages,dtype=int)
    
    def set_state(self, state: list):
        self.pit_ages = state

class RaceMDP():
    def __init__(self,
                 NN : torch.nn.Module,
                 gamma : float = 0.9,
                 num_laps : int = 50):
        self.NN = NN
        self.gamma = gamma
        self.t_p_estimate = 30.0

        self.t_i = 0.0
    
    def set_state(self,
                  state : RaceState) -> None:
        self.state = state
    
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
        return -1 * ((self.t_i - self.state.t_im1) +
                     self.pit_time(action))
    
    def transition(self,
                   action : int) -> None:
        self.state.t_im1 = self.t_i
        self.state.lap_number += 1
        if action > 0:
            self.state.tire_id = action-1
            self.state.constants.stint += 1
            self.state.tire_age = 1
            self.state.events.pit_stop = True
        else:
            self.state.tire_age += 1
    
    def rollout(self,
                policy : Policy,
                depth : int):
        ret = 0.0
        for i in range(depth):
            action = policy.eval(self.state)
            self.transition(action)
            r = self.reward(action)
            ret += self.gamma**(i-1) * r
        return ret
    
    def mc_rollout(self,
                policy : Policy,
                depth : int,
                num_rollouts : int):
        ret = 0.0
        for m in range(num_rollouts):
            for i in range(depth):
                action = policy.eval(self.state)
                self.transition(action)
                r = self.reward(action)
                ret += self.gamma**(i-1) * r
        return ret/num_rollouts