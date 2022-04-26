import sys
import unittest
import statsmodels.api as sm
import numpy as np
sys.path.append('..')  # need for local imports

from dataGen.DataMaker import DataMaker  # noqa
from models.lpm import LPM  # noqa


class TestLPM(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100000
        self.n_features = 4
        self.n_classes = 2

        self.coef_vector = [.2, .2, .4, .3, -.2]

        self.data_maker = DataMaker(self.n_samples, self.n_features, self.n_classes)
        self.X, self.y = self.data_maker.make_linear_data(self.coef_vector)

        self.sm_est = sm.OLS(self.y, self.X).fit()

    def test_estimate(self):
        """
        Test that the estimate is the same as the statsmodels estimate
        """
        coef_est = LPM.estimate(self.X, self.y)
        sm_coef_est = self.sm_est.params
        self.assertTrue(np.allclose(coef_est, sm_coef_est, atol=.001))

        # print(f"coef_est: {coef_est}")
        # print("sm coef est: {}".format(sm_coef_est))

    def test_se(self):
        """
        Test that the standard error is the same as the statsmodels estimate
        """
        se_est = LPM.estimate_std_err(self.X, self.y, self.n_samples)
        self.assertTrue(np.allclose(se_est, self.sm_est.bse, atol=.001))

        # print(f"se_est: {se_est}")
        # print("sm se est: {}".format(self.sm_est.bse))


if __name__ == '__main__':
    unittest.main()
