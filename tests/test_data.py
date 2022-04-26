import unittest
import sys
import numpy as np

sys.path.append('../')


from dataGen.DataMaker import DataMaker  # noqa


class TestDataMaker(unittest.TestCase):
    def setUp(self):
        # set sample size, feature number, num classes, and true beta values
        n_samples = 100000
        n_features = 4
        n_classes = 2
        beta_vector = [.2, -.3, .1, .4, -.2]

        # create data maker object
        self.dataMaker = DataMaker(n_samples, n_features, n_classes)

        # create data
        self.X, self.y = self.dataMaker.make_binary_data(beta_vector)

    def test_make_binary_data(self):
        # test that y is between zero and one
        self.assertTrue(np.all(self.y >= 0) and np.all(self.y <= 1))

        # test that X has the correct number of features
        self.assertEqual(self.X.shape[1], 4 + 1)


if __name__ == '__main__':
    unittest.main()
