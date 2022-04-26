import unittest
import sys
from sklearn.linear_model import LogisticRegression
sys.path.append('../')  # noqa

from dataGen.DataMaker import DataMaker  # noqa
from models.logit import Logit  # noqa


class TestLogit(unittest.TestCase):
    def setUp(self):
        n_sample = 1000
        n_feature = 3
        self.data_maker = DataMaker(n_sample, n_feature)
        self.X, self.y = self.data_maker.make_binary_data([.03, -.08, .02, .07])

    def test_prediction_range(self):
        predictions = Logit().predict(self.X, self.y)
        print(self.X)
        self.assertTrue(all(predictions >= 0) and all(predictions <= 1))


if __name__ == '__main__':
    unittest.main()
