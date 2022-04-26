import numpy as np
from scipy.optimize import minimize  # powell works
import sys
sys.path.append('../')


class Logit:
    def log_func(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        :param X: feature matrix
        :param beta: coefficients
        :return: output of the link function
        """
        return 1/(1 + np.e ** (-np.matmul(X, beta)))

    def log_likelihood(self, beta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
        """
        :param beta: model coefficients
        :param X: feature matrix
        :param y: outcome variable
        :return: sum of log-likelihood function for the given parameters
        """
        ll = -np.sum(y * np.log(self.log_func(X, beta)) + (1 - y) * np.log(1 - self.log_func(X, beta)))
        return ll

    def logit_model(self, X, y,):
        """
        :param X: feature matrix
        :param y: outcome variable
        :return: coefficients of the logistic regression model
        :rtype: scipy.optimize.OptimizeResult
        """
        beta0 = np.zeros(X.shape[1])
        beta0.fill(0.05)

        bnds = ((-.99, .99),) * X.shape[1]
        beta_hat = minimize(self.log_likelihood, beta0, args=(y, X,), bounds=bnds,
                            method="L-BFGS-B", tol=1e-8, options={'maxiter': 100000, 'ftol': 1e-6})  # e-4 is good
        return beta_hat

    def predict(self, X, y, decision_boundary=.5):
        """
        :param X: feature matrix
        :param y: outcome variable
        :return: predicted values
        """
        beta_hat = self.logit_model(X, y)
        return (self.log_func(X, beta_hat.x) >= decision_boundary) * 1

    def rsq(self, X, y):
        """
        Compute McFadden's R-squared

        :param y: The outcome vector.
        :param yhat: The predicted outcome vector.

        :return: The R^2 value.
        :rtype: float
        """
        # compute R^2
        beta_hat = self.logit_model(X, y)
        L0 = self.log_likelihood(np.array(beta_hat.x[0]).reshape(-1, 1), y, np.array(X[:, 0]).reshape(-1, 1))
        L1 = self.log_likelihood(beta_hat.x, y, X)
        return 1 - (np.log(L1) / np.log(L0))


if __name__ == "__main__":
    from dataGen.DataMaker import DataMaker  # noqa
    from dataGen.o_data import OData
    from sklearn.preprocessing import StandardScaler  # noqa
    from sklearn.datasets import make_classification  # noqa

    # set data parameters
    n_samples = 10000
    n_features = 3

    # generate data
    # dataGen = DataMaker(n_samples, n_features)
    # X, y = dataGen.make_data(30, 70, .9, "uniform", "uniform", low=-10, high=10)

    # generate sklearn data
    # X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2,
    #                             n_redundant=0, n_classes=2, random_state=0, shuffle=True)

    # generate opossum data
    y, X, *extra = OData.make_data(n_samples, n_features)

    # scale data
    X = StandardScaler().fit_transform(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # fit model
    est = Logit().logit_model(X, y)
    print(est)

    # get predictions
    # y_hat = Logit().predict(X, y)

    # display r2
    print(f"McFadden's R^2: {Logit().rsq(X, y)}")
