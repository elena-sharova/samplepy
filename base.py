"""
@created on Wed Sep 26 09:17:10 2016
@package: samplepy
@author: Elena Sharova
@purpose: Base class for sampling package
@open source distribution licence: MIT

"""
import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import derivative
from types import LambdaType
from math import isnan, log


class _BaseSampling(object):
    # Base class for samplepy sampling package
    
   
    def __check_function__(self, f):
        # Checks if the passed function is a lambda function
        
        return isinstance(f, LambdaType) and f.__name__ == "<lambda>"
        
        
    def __check_defined_over_interval__(self, f, interval):
        # Checks if the passed lambda is defined over the specified interval
        
        if interval[0]< interval[1]:
            try:
                if not isnan(derivative(f, interval[1], dx=1e-6)) and not isnan(derivative(f, interval[0], dx=1e-6)):
                    return True
                else:
                    return False
                
            except:
                raise ValueError("The function is not defined over the entire interval.")
        else:
            raise ValueError("Invalid interval: interval[0] >= interval[1].")