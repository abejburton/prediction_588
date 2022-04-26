import sys
import numpy as np
sys.path.append('..')  # need for DataMaker import


class LPM:
    @staticmethod
    def estimate(X: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate the linear probability of the data given the model parameters.

        :param X: The feature matrix.
        :param y: The outcome vector.

        :return: The estimated linear probability coefficients, beta-hat.
        :rtype: float
        """
        # estimate coefficients
        beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
        return beta_hat.T

    @staticmethod
    def predict_y(X: np.ndarray, y: np.ndarray):
        """
        Predict y hat given X and y

        :param X: The feature matrix.
        :param y: The outcome vector.

        :return: The predicted y-hat.
        """
        # predict y values
        return (np.matmul(X, LPM.estimate(X, y).T) >= .5) * 1

    @staticmethod
    def estimate_std_err(X: np.ndarray, y: np.ndarray, n: int):
        """
        Estimate the standard error of the linear probability of the data given the model parameters.
        """

        # predict y-hat
        yhat = LPM.predict_y(X, y)

        # estimate residuals
        resid = (y - yhat).reshape(-1, 1)

        # compute (X.T @ X)^-1
        term1 = np.linalg.inv(np.matmul(X.T, X))

        # X.T @ X * u' @ u
        term2 = np.matmul(X.T, X) * np.matmul(resid.T, resid)

        # compute variance covariance matrix
        var_cov = term1 @ term2 @ term1

        # return standard error
        return np.diag(np.sqrt(var_cov / (n)))

    @staticmethod
    def rsq(y, yhat):
        """
        Compute the R^2 value of the linear probability model.

        :param y: The outcome vector.
        :param yhat: The predicted outcome vector.

        :return: The R^2 value.
        :rtype: float
        """
        # compute R^2
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


if __name__ == '__main__':
    from dataGen.DataMaker import DataMaker
    from dataGen.o_data import OData
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    # set data parameters
    n_samples = 10000
    n_features = 3

    # generate data from DataMaker
    # dataGen = DataMaker(n_samples, n_features)
    # X, y = dataGen.make_multiclass_data()

    # generate sklearn data
#    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=0,
#                               n_classes=2, n_clusters_per_class=1, random_state=0)

    # generate data from OData
    y, X, *extra = OData.make_data(n_samples, n_features)

    # standardize data and add column of ones
    X = StandardScaler().fit_transform(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # estimate coeffiencents
    coef_est = LPM.estimate(X, y)
    print("Coefficients: ", coef_est)

    # get predictions
    yhat = LPM.predict_y(X, y)
    print(sum(y))
    print(sum(yhat))

    # compute R^2
    print(f"R^2: {LPM.rsq(y, LPM.predict_y(X, y))}")
