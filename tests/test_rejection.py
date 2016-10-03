
import unittest
import numpy as np
from samplepy import Rejection

class samplepyRejectionTestCase(unittest.TestCase):
    
    def setUp(self):
        self.f = lambda x: 2.0*np.exp(-2.0*x)
        self.interval = [-1, 1]
        self.rej = Rejection(self.f, self.interval)
        
    def tearDown(self):
        self.rej = None
        self.f = None
    
    def test_sample_size(self):
        self.assertEqual(len(self.rej.sample(100)), 100, "sample set not equal to requested size")
        
    def test_output_max_min(self):
        self.assertGreaterEqual(max(self.interval), max(self.rej.sample(100)), "generated sample outside the max interval point")
        self.assertLessEqual(min(self.interval), min(self.rej.sample(100)), "generate sample outisde the min interval point")
        
    def test_raised_exception_bad_interval(self):
        self.f2 = lambda x: 3.*np.log(abs(x))/x
        self.interval2 = [1, -1]

        with self.assertRaises(ValueError):
            Rejection(self.f2, self.interval2) 
            
    def test_raised_bad_function(self):
        self.f2 = lambda x: 3.0 # single point function
        self.interval2 = [-1.0, 1.0]
        
        with self.assertRaises(ValueError):
            self.rej2 = Rejection(self.f2, self.interval2)            
            self.rej2.sample(100)
        
    
        
if __name__ == '__main__':
    unittest.main()