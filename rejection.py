# -*- coding: utf-8 -*-
"""
@created on Wed Sep 21 09:17:10 2016
@package: samplepy
@author: Elena Sharova
@purpose: Implements simple rejection sampling for
@         unknown true distribution specified by f over an interval[,]
@open source distribution licence: MIT
"""
import copy

import numpy as np

from samplepy.base import _BaseSampling


class Rejection(_BaseSampling):
    """
       Implements rejection sampling for a specified function over a specififed interval
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

        return "samplepy univariate Rejection Sampling object"

    def __len__(self):

        return len(self._sample)

    def __getitem__(self, position):

        return self._sample[position]

    def sample(self, sample_size, seed=1):
        """
         Create a sample set of sample_size samples
         seed (optional, default = 1) to seed the random number generator
        """

        if sample_size < 1:
            raise ValueError("N is not > 1.")

        if int(seed) < 0:
            raise ValueError("Seed is not > 0.")

        # seed the random number generator
        np.random.seed(seed)

        # calculate the min&max of f(x) to scale u2
        x = np.arange(self._interval[0], self._interval[1],
                      (abs(self._interval[1]-self._interval[0]))/float(sample_size))
        fx = self._f(x)
        max_fx = np.max(fx)
        min_fx = np.min(fx)

        if min_fx == max_fx:
            raise ValueError("Cannot generate a sample for a single point.")

        self._sample = np.empty(sample_size)

        i = 0
        while i < sample_size:
            u1 = np.random.uniform(self._interval[0], self._interval[1])
            u2 = np.random.uniform(min_fx, max_fx)

            # collect a full sample here
            if u2 <= self._f(u1):

                self._sample[i] = u1
                i += 1

        return copy.copy(self._sample)
        