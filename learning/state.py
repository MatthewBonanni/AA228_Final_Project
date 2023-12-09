import numpy as np
import torch

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
                 tire_ids_used : list,
                 events : RaceEvents,
                 weather : RaceWeather,
                 constants : RaceConstants):
        self.t_im1 = t_im1
        self.tire_age = tire_age
        self.lap_number = lap_number
        self.tire_id = tire_id
        self.tire_ids_used = tire_ids_used
        self.events = events
        self.weather = weather
        self.constants = constants
    
    def to_list(self) -> list:
        return [self.t_im1,
                self.tire_age,
                self.lap_number,
                self.tire_id,
                self.tire_ids_used,
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