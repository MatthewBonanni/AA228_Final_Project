import pdb
import numpy as np
from collections.abc import Callable
from copy import copy, deepcopy

from .RaceMDP import Policy

class PolicyOptimizer():
    def __init__(self,
                 learning_rates : list,
                 iters : int, 
                 eps : int):
        self.learning_rates = learning_rates
        self.iters = iters
        self.eps = eps

    def eval(self):
        raise NotImplementedError
    
class HookeJeeves(PolicyOptimizer):
    def __init__(self, 
                 step_sizes : list,
                 bounds : list,
                 iters : int, 
                 eps : int,
                 scale_factors : list):
        
        super().__init__(step_sizes, iters, eps)
        self.bounds = bounds
        self.scale_factors = scale_factors
        

    def eval(self,
             policy : Policy, 
             eval_fn: Callable, 
             eval_fn_args : list):
        
        policy_opt = deepcopy(policy)

        x = policy_opt.get_parameters()
        best_x = x
        best_y = eval_fn(policy_opt, *eval_fn_args)

        learning_rates = self.learning_rates

        for i_iter in range(self.iters):
            improvement = False
            for i_row in range(x.shape[0]):
                for i_col in range(x.shape[1]):
                    for sign in [-1,1]:
                        x_new = deepcopy(x)
                        x_new[i_row, i_col] += learning_rates[i_row] * sign
                        x_new[i_row, i_col] = max(self.bounds[i_row][0], x_new[i_row, i_col])
                        x_new[i_row, i_col] = min(self.bounds[i_row][1], x_new[i_row, i_col])
                        policy_opt.set_parameters(x_new)
                        y_new = eval_fn(policy_opt, *eval_fn_args)
                        if y_new > best_y:
                            best_x = x_new
                            best_y = y_new
                            improvement = True

            if improvement == False:
                learning_rates = [learning_rates[i_row] // self.scale_factors[i_row]
                                  for i_row in range(x.shape[0])]
                if np.all(np.array(learning_rates) <= self.eps):
                    break
            
            x = best_x
            print(best_x)
         
        policy_opt.set_parameters(best_x)
        return policy_opt