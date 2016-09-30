"""
@created on Wed Sep 21 09:17:10 2016
@package: samplepy
@author: Elena Sharova
@purpose: Implements importance sampling for unknown true distribution specified by f over an interval[,]
@open source distribution licence: MIT

"""

import numpy as np
import copy

from base import _BaseSampling
from scipy.misc import derivative
from types import LambdaType
from math import isnan, log


class Importance(_BaseSampling):
    # Implements importance sampling for a specified function over a specififed interval
    
    
    def __init__(self, f, interval=[0.0,1.0]):
        # Constructor accepts the lambda function and the interval over which it should be defined
        
        if (self.__check_function__(f)):
            self._f = f
        else:
            raise ValueError("Invalid function provided.")
        if(self.__check_defined_over_interval__(f,interval)):
            self._interval = interval
        else:
            raise ValueError("The function is not defined over the entire interval.")
        
        self._sample = np.array([])

    
    def __repr__(self):
        
        return "samplepy univariate Importance Sampling object"
        
    def __len__(self):
        
        return len(self._sample)
        
    def __getitem__(self, position):
        
        return self._sample[position]
        
    def sample(self, N,  quantile, weight, seed=1):
        # Create a sample set of N samples 
        
        assert N > 0, ValueError("N is not > 0.")
        
        assert seed > 0, ValueError("Seed is not > 0.")
        
        assert (quantile > 0.0 and quantile <1.0), ValueError("Quantile not in (0,1).")
        
        assert (weight > 0.0 and weight <1.0), ValueError("Quantile weight not in (0,1).")
        
        # seed the random number generator
        np.random.seed(seed)
        
        # calculate the mean f(x) to scale u2
        x = np.arange(self._interval[0],self._interval[1],(abs(self._interval[1]-self._interval[0]))/N)
        fx = self._f(x)
        max_fx = np.max(fx)
        min_fx = np.min(fx)        

        self._sample = np.empty(N)
        i=0
        
        threshold= (1.0-quantile)*N
        w=N*weight
        
        q = lambda x, u: u >= x  if (quantile < 0.5) else u<=x
            
        while i<N:
            u1=np.random.uniform(self._interval[0], self._interval[1], size=1)
            u2=np.random.uniform(min_fx, max_fx, size=1)
            
            # collect a full sample here
            if u2 <=self._f(u1):
                self._sample[i]= u1
                i+=1
            # if this is the quantile we need to oversample, then sample from it unless we have reached the desired quantile weight
            elif (q(x[threshold], u1) and w>0):
                self._sample[i]= u1
                i+=1
                w-=1
        
        return copy.copy(self._sample)
 