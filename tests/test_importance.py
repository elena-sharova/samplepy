import unittest
import numpy as np
from samplepy import Importance

class samplepyImportanceTestCase(unittest.TestCase):
    
    def setUp(self):
        self.f = lambda x: 2.0*np.exp(-2.0*x)
        self.interval = [-1, 1]
        self.imp = Importance(self.f, self.interval)
        
    def tearDown(self):
        self.imp = None
        self.f = None
    
    def test_sample_size(self):
        self.assertEqual(len(self.imp.sample(100, 0.01, 0.01)), 100, "sample set not equal to requested size")
        
    def test_output_max_min(self):
        self.assertGreaterEqual(max(self.interval), max(self.imp.sample(100, 0.01, 0.01)), "generated sample outside the max interval point")
        self.assertLessEqual(min(self.interval), min(self.imp.sample(100, 0.01, 0.01)), "generate sample outisde the min interval point")
        
    def test_raised_exception_bad_interval(self):
        self.f2 = lambda x: 3.*np.log(abs(x))/x
        self.interval2 = [1, -1]

        with self.assertRaises(ValueError):
            Importance(self.f2, self.interval2) 
            
    def test_raised_bad_function(self):
        self.f2 = lambda x: 3.0 # single point function
        self.interval2 = [-1.0, 1.0]
        
        with self.assertRaises(ValueError):
            self.imp2 = Importance(self.f2, self.interval2)            
            self.imp2.sample(100, 0.01, 0.01, 3)
    
    def test_raised_bad_paramteres(self):
        
        with self.assertRaises(ValueError):
            self.imp.sample(100, -5, 2)
        
    
        
if __name__ == '__main__':
    unittest.main()