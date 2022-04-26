import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import sys
sys.path.append('../')


class Probit:
    def normal_cdf(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        :param X: feature matrix
        :param beta: model coefficients
        :return: standard normal cdf evaluated at X'b
        """
        return norm.cdf(X @ beta)

    def log_likelihood(self, beta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
        """
        :param beta: model coefficients
        :param X: feature matrix
        :param y: outcome variable
        :return: sum of log likelihood function
        :rtype: float
        """
        ll = -np.sum(y * np.log(self.normal_cdf(X, beta)) + (1 - y) * np.log(1 - self.normal_cdf(X, beta)))
        return ll

    def probit_model(self, X, y,):
        """
        :param X: feature matrix
        :param y: outcome variable
        :return: coefficients of the probit model
        :rtype: scipy.optimize.OptimizeResult
        """
        beta0 = np.zeros(X.shape[1])
        beta0.fill(.1)

        bnds = ((-.9999, .9999),) * X.shape[1]
        beta_hat = minimize(self.log_likelihood, beta0, args=(y, X,), bounds=bnds,
                            method='SLSQP', tol=1e-10, options={'maxiter': 100000, 'ftol': 1e-6})  # e-4 is good
        return beta_hat

    def predict(self, X, y, decision_boundary=.5):
        """
        :param X: feature matrix
        :param y: outcome variable
        :param decision_boundary: decision boundary
        :return: predicted values
        """
        beta_hat = self.probit_model(X, y)
        return (self.normal_cdf(X, beta_hat.x) >= decision_boundary) * 1

    def rsq(self, X, y):
        """
        Compute McFadden's R^2

        :param y: The outcome vector.
        :param yhat: The predicted outcome vector.

        :return: The R^2 value.
        :rtype: float
        """
        # compute R^2
        beta_hat = self.probit_model(X, y)
        L0 = self.log_likelihood(np.array(beta_hat.x[0]).reshape(-1, 1), y, np.array(X[:, 0]).reshape(-1, 1))
        L1 = self.log_likelihood(beta_hat.x, y, X)
        return 1 - (np.log(L1) / np.log(L0))


if __name__ == "__main__":
    from dataGen.DataMaker import DataMaker  # noqa
    from dataGen.o_data import OData
    from sklearn.datasets import make_classification  # noqa
    from sklearn.preprocessing import StandardScaler  # noqa

    # set number of samples and number of features to include in the model
    n_samples = 10000
    n_features = 3

    # generate data from DataMaker
    # dataGen = DataMaker(n_samples, n_features)
    # X, y = dataGen.make_data(30, 70, .3, "uniform", "uniform", low=-5, high=5)

    # generate sklearn data
#    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features,
#                               n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=0)

    # generate data from DataMaker
    y, X, *extra = OData.make_data(n_samples, n_features)

    # standardize data and add column of ones
    X = StandardScaler().fit_transform(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # fit model
    est = Probit().probit_model(X, y)
    print(est)

    # get predictions
    y_hat = Probit().predict(X, y)

    # compute R^2
    print(f"McFadden's R^2: {Probit().rsq(X, y)}")
