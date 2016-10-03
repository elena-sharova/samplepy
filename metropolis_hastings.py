"""
@created on Wed Sep 26 09:17:10 2016
@package: samplepy
@author: Elena Sharova
@purpose: Implements Metropolis-Hastings sampling for unknown
@         true distribution specified by f over an interval[,]
@open source distribution licence: MIT

"""
import copy

import numpy as np

from samplepy.base import _BaseSampling


class MH(_BaseSampling):
    """
    Implements the Metropolis-Hastings sampling for a specified function over a specififed interval
    """

    def __init__(self, f, interval=[0.0, 1.0]):
        # Constructor accepts the lambda function and the interval over which it should be defined

        if self.__check_function__(f):
            self._f = f
        else:
            raise ValueError("Invalid function provided.")
        if self.__check_defined_over_interval__(f, interval):
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


    def sample(self, sample_size, burn_in=100, lag=50, seed=1):
        """
        Create a sample set of sample_size size
        burn-in period length (optional, default = 100)
        lag period length (optional, default = 50)
        seed (optional, default = 1) to seed the random number generator
        """

        if sample_size < 0:
            raise ValueError("N is not > 0.")

        if int(seed) < 0:
            raise ValueError("Seed is not > 0.")

        if burn_in < 0:
            raise ValueError("burn-in is not >= 0.")

        if lag < 1:
            raise ValueError("Lag is not >= 1.")

        # seed the random number generator
        np.random.seed(seed)

        x = np.arange(self._interval[0], self._interval[1],
                      abs(self._interval[1]-self._interval[0])/float(sample_size))

        fx = self._f(x)

        min_fx = np.min(fx)
        max_fx = np.max(fx)

        if min_fx == max_fx:
            raise ValueError("Cannot generate a sample for a single point.")

        self._sample = np.empty(sample_size)
        i = 0

        next_x = np.random.uniform(self._interval[0], self._interval[1])
        c = 1.0/(self._interval[1]-self._interval[0])

        # burn-in period
        for l in xrange(burn_in):
            xproposed = np.random.uniform(self._interval[0], self._interval[1])
            alpha = (self._f(xproposed)/self._f(next_x))*c
            acceptance_prob = min(1, alpha)
            u1 = np.random.uniform()
            if u1 <= acceptance_prob:
                next_x = xproposed

        while i < sample_size:

            # lag period
            for l in xrange(lag):
                xproposed = np.random.uniform(self._interval[0], self._interval[1])
                alpha = (self._f(xproposed)/self._f(next_x))*c
                acceptance_prob = min(1, alpha)
                u1 = np.random.uniform()
                if u1 <= acceptance_prob:
                    next_x = xproposed

                self._sample[i] = next_x
                i += 1

        return copy.copy(self._sample)
    