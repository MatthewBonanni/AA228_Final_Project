import numpy as np

from .state import RaceState

class Policy():
    def __init__(self):
        return
    
    def eval(self,
             mdp_state: RaceState) -> int:
        raise NotImplementedError()
    
    def get_parameters(self):
        raise NotImplementedError()
    
    def set_parameters(self,
                       parameters : np.array):
        raise NotImplementedError()

class RandomPolicy():
    def eval(self,
             mdp_state : RaceState) -> int:
        return np.random.randint(6)

class AgeBasedRandomTirePolicy(Policy):
    def __init__(self,
                 pit_ages : np.array):
        self.pit_ages = pit_ages
    
    def eval(self,
             mdp_state : RaceState) -> int:
        if mdp_state.tire_age > self.pit_ages[0, mdp_state.tire_id]:
            # If raining, use intermediate or wet
            if mdp_state.events.rainfall:
                return np.random.randint(5,6+1)
            
            next_tire = np.random.randint(1,4+1)
            while next_tire == mdp_state.tire_id+1:
                next_tire = np.random.randint(1,4+1)
            return next_tire
        return 0
    
    def get_parameters(self):
        return self.pit_ages
    
    def set_parameters(self,
                       parameters : list):
        self.pit_ages = parameters

class AgeSequencePolicy(Policy):
    def __init__(self,
                 ids_ages : np.array):
        self.ids_ages = ids_ages
    
    def eval(self,
             mdp_state : RaceState) -> int:
        pit_id = np.argwhere(mdp_state.lap_number == np.cumsum(self.ids_ages[0]))
        if len(pit_id > 0):
            return 1 + self.ids_ages[1, int(pit_id)]
        return 0

    def get_parameters(self):
        return self.ids_ages
    
    def set_parameters(self,
                       parameters: np.array):
        self.ids_ages = parameters
