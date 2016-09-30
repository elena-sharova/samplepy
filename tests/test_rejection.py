from __future__ import absolute_import

import unittest
from ..rejection import Rejection

class samplepyRejectionTestCase(unittest.TestCase):
    
    def setUp(self):
        self.f = lambda x: 2.0*np.exp(-2.0*x)
        self.rej = Rejection(f, [-1, 1])
        
    def tearDown(self):
        self.rej.dispose()
        self.rej = None
        self.f.dispose()
        self.f = None
    
    def test_sample_size(self):
        self.assertEqual(len(self.rej(100, 1)), 100, "sample set not equal to requested size")
        

if __name__ == '__main__':
    unittest.main()