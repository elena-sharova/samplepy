"""
@created on Wed Sep 26 09:17:10 2016
@package: samplepy
@author: Elena Sharova
@purpose: Implements Metropolis-Hastings sampling for unknown true distribution specified by f over an interval[,]
@         Based on a tutorial "The Metropolis-Hastings Algorithm" by Dan Navarro
@open source distribution licence: MIT

"""

import numpy as np
import copy

from base import _BaseSampling
from scipy.misc import derivative
from types import LambdaType
from math import isnan, log


class MH(_BaseSampling):
    # Implements the Metropolis-Hastings sampling for a specified function over a specififed interval
    
    
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
        
        return "samplepy univariat Metropolis-Hastings Sampling object"
        
    def __len__(self):
        
        return len(self._sample)
        
    def __getitem__(self, position):
        
        return self._sample[position]

        
    def sample(self, N, burn_in=100, lag=50,  seed=1):
        # Create a sample set of approximately N samples 
        
        assert N > 0, ValueError("N is not > 0.")
        
        assert seed > 0, ValueError("Seed is not > 0.")
        
        assert burn_in >= 0, ValueError("burn-in is not >= 0.")
        
        assert lag >= 1, ValueError("Lag is not >= 1.")
        
        # seed the random number generator
        np.random.seed(seed)
        
        x = np.arange(self._interval[0],self._interval[1],abs(self._interval[1]-self._interval[0])/N)
        
        fx = self._f(x)
        
        x_mean = np.mean(fx)
        x_std = np.std(fx)
        
        self._sample = np.empty(N)
        i=0
        
        next_x= x_mean
        
        # burn-in period
        for l in xrange(burn_in):
            xproposed = np.random.normal(loc=next_x, scale=x_std, size=1)
            acceptance_prob = self._f(xproposed)
            u1=np.random.uniform()
            if u1 <= acceptance_prob:
                next_x=xproposed  
                
        while i<N:
            
            # lag period
            for l in xrange(lag):
                xproposed = np.random.normal(loc=next_x, scale=x_std, size=1)
                acceptance_prob = self._f(xproposed)
                u1=np.random.uniform()
                if u1 <= acceptance_prob:
                    next_x=xproposed
            
            self._sample[i]=next_x
            i+=1
        
        return copy.copy(self._sample)
    