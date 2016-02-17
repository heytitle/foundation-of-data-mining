import unittest
import nn
from instance import Instance
from simplejson import dumps

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

    def test_confustion_matrix(self):
        data = [
            Instance( (), label=7, predicted_label=7 ),
            Instance( (), label=7, predicted_label=7 ),
            Instance( (), label=7, predicted_label=7 ),
            Instance( (), label=7, predicted_label=2 ),
            Instance( (), label=7, predicted_label=1 ),
            Instance( (), label=7, predicted_label=5 ),
            Instance( (), label=2, predicted_label=7 ),
            Instance( (), label=2, predicted_label=2 ),
            Instance( (), label=2, predicted_label=0 ),
            Instance( (), label=1, predicted_label=0 ),
            Instance( (), label=6, predicted_label=6 ),
            Instance( (), label=6, predicted_label=6 ),
            Instance( (), label=4, predicted_label=9 ),
            Instance( (), label=4, predicted_label=8 ),
            Instance( (), label=4, predicted_label=9 )
        ]

        matrix = nn.confusion_matrix(data)
        expected = [
            [0,0,0,0,0,0,0,0,0,0], # 0
            [1,0,0,0,0,0,0,0,0,0], # 1
            [1,0,1,0,0,0,0,1,0,0], # 2
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,2],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,2,0,0,0],
            [0,1,1,0,0,1,0,3,0,0], # 7
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
        ]
        self.assertEqual( dumps(matrix), dumps(expected) )

        self.assertEqual( nn.precision( matrix, 7 ), 0.75 );
        self.assertEqual( nn.precision( matrix, 6 ), 1 );
        self.assertEqual( nn.precision( matrix, 4 ), 0 );

        self.assertEqual( nn.recall( matrix, 7 ), 0.5 );
        self.assertEqual( nn.recall( matrix, 6 ), 1 );
        self.assertEqual( nn.recall( matrix, 4 ), 0 );
        self.assertEqual( nn.recall( matrix, 2 ), 0.3333333333333333 );

        self.assertEqual( nn.accuracy( matrix ), 6.0/15 );

if __name__ == '__main__':
    unittest.main()
