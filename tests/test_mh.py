import unittest
import numpy as np
from samplepy import MH

class samplepyMHTestCase(unittest.TestCase):
    
    def setUp(self):
        self.f = lambda x: 2.0*np.exp(-2.0*x)
        self.interval = [-1, 1]
        self.mh = MH(self.f, self.interval)
        
    def tearDown(self):
        self.mh = None
        self.f = None
    
    def test_sample_size(self):
        self.assertEqual(len(self.mh.sample(100)), 100, "sample set not equal to requested size")

    def test_raised_exception_bad_interval(self):
        self.f2 = lambda x: 3.*np.log(abs(x))/x
        self.interval2 = [1, -1]

        with self.assertRaises(ValueError):
            MH(self.f2, self.interval2) 
            
    def test_raised_bad_function(self):
        self.f2 = lambda x: 3.0 # single point function
        self.interval2 = [-1.0, 1.0]
        
        with self.assertRaises(ValueError):
            self.mh2 = MH(self.f2, self.interval2)            
            self.mh2.sample(100)
    
    def test_raised_bad_paramteres(self):
        
        with self.assertRaises(ValueError):
            self.mh.sample(100, -5, -2)
        
    
        
if __name__ == '__main__':
    unittest.main()