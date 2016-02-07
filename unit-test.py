import unittest
import nn
from instance import Instance

class TestStringMethods(unittest.TestCase):

    def test_vector_length(self):
        ins = Instance( [3,4] )
        self.assertEqual( ins.length, 5 )

    def test_compute_label( self ):
        v1 = {
            "features": (3,4),
            "label_array": (0,0,0,0,0,1,0,0,0)
        }
        ins = Instance( v1['features'], v1['label_array'] )
        self.assertEqual( ins.label, 5 )

        v1 = {
            "features": (3,4),
            "label": 7
        }
        ins = Instance( v1['features'], label=v1['label'] )
        self.assertEqual( ins.label, 7 )

    def test_cosine_dist(self):
        v1 = Instance( (5, 0, 3, 0, 2, 0, 0, 2, 0, 0) )
        v2 = Instance( (3, 0, 2, 0, 1, 1, 0, 1, 0, 1) )

        self.assertEqual( nn.cosine_dist(v1,v2), 0.06439851429360033 )

        v1 = Instance( ( 3, 4 ) )
        v2 = Instance( ( 6, 8 ) )
        self.assertEqual( nn.cosine_dist(v1,v2), 0 )

if __name__ == '__main__':
    unittest.main()
