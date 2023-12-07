import pdb
import numpy as np
from collections.abc import Callable
from copy import copy, deepcopy

from .RaceMDP import Policy

class PolicyOpt():
    def __init__(self,
                 step_size : int, 
                 iters : int, 
                 eps : int):
        self.lr = step_size
        self.iters = iters
        self.eps = eps

    def eval(self):
        raise NotImplementedError
    
class HookeJeeves(PolicyOpt):
    def __init__(self, 
                 step_size : int, 
                 iters : int, 
                 eps : int,
                 scale_factor : int):
        
        super().__init__(step_size,iters, eps)
        self.scale_factor = scale_factor
        

    def eval(self,
            policy : Policy, 
            eval_fn: Callable, 
            eval_fn_args : list):
        
        best_x = np.array(policy.get_state())
        x = best_x
        dims = x.shape[0]

        pol = deepcopy(policy)
        best_y = eval_fn(pol, *eval_fn_args)

        alpha = self.lr
        
        i = 0
        for i in range(self.iters):
            imp = False
            for i in range(dims):
                for sign in [-1,1]:
                    xt = x.copy()
                    xt[i] = xt[i] + alpha*sign
                    policy.set_state([s for s in xt])
                    y = eval_fn(pol, *eval_fn_args)
                    if y > best_y:
                        best_x = xt
                        best_y = y
                        imp = True

            if imp == False:
                alpha = alpha//self.scale_factor
                if alpha < self.eps:
                    break

            x = best_x
            print(best_x)
            
        pol.set_state([s for s in best_x])
        return pol
                