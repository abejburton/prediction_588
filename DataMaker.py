import numpy as np


class DataMaker:
    def __init__(self, n_samples, n_features, n_classes=2):
        """
        :param n_samples: number of samples
        :param n_features: number of features (covariates)
        :param n_classes: number of outcome classes (set to 2 for binary classification)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

    def make_linear_data(self, beta_arr: np.ndarray):
        """
        :param beta_arr: array of true beta values; len(beta_vector) == n_features + 1
        :return: X, y
        :rtype: np.array, np.array
        """

        # check to make sure beta_vector is the right length
        if (len(beta_arr) - 1) != self.n_features:
            raise ValueError("beta_arr must have length n_features + 1")

        # translate beta_vector to np.array if needed
        if not isinstance(beta_arr, np.ndarray):
            beta_arr = np.array(beta_arr)

        # generate random X values
        X = np.random.rand(self.n_samples, self.n_features)

        # generate y values using linear method
        # the error term is normally distributed with mean 0 and variance .125
        # the decision boundary for y=1 is set by np.round()
        y = np.round(
            beta_arr[0] + np.dot(X, beta_arr[1:]) + np.random.normal(0, .125, size=self.n_samples)
        )

        # ensure all y greater than or equal to zero
        for i, val in enumerate(y):
            dist_from_one = 1 - val
            if abs(0 < dist_from_one):
                y[i] = 0

        # add a column of ones to the X matrix for use in future estimation
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        return X, y

    def make_multiclass_data(self):
        """
        Generates X and y data for use in binary outcome estimation models.
        Error term is default set to uniform, but error paramaters can be altered.

        :return: X, y
        :rtype: np.array, np.array
        """
        # generate features
        def gen_features(boundary, upper):
            if upper:
                return np.random.normal(boundary + boundary / 2, opt_stdev(boundary), self.n_samples // 2)
            else:
                return np.random.normal(x1_boundary - x1_boundary / 2, opt_stdev(x1_boundary), self.n_samples // 2)

        def opt_stdev(boundary):
            """
            :param boundary: the artificial decision boundary for the feature
            :return: the standard deviation to use in creating data
            """
            boundary = int(boundary)
            return int(boundary // (boundary - boundary / 2))

        # create error term
        uni_error = np.random.uniform(-10, 10, self.n_samples).reshape(-1, 1)

        # pick some center for x1
        x1_boundary = 20

        # distribute x1 around the boundary
        feature_1_class_1 = gen_features(x1_boundary, upper=False)
        feature_1_class_2 = gen_features(x1_boundary, upper=True)

        # pick some center for x2
        x2_boundary = 40

        # distribute x2 around the boundary
        feature_2_class_1 = gen_features(x2_boundary, upper=False)
        feature_2_class_2 = gen_features(x2_boundary, upper=True)

        # merge features
        x1 = np.concatenate((feature_1_class_1, feature_1_class_2)).reshape(-1, 1)
        x2 = np.concatenate((feature_2_class_1, feature_2_class_2)).reshape(-1, 1)
        ones_column = np.ones((self.n_samples, 1))

        # create feature matrix
        X = np.column_stack((ones_column, x1+uni_error, x2+uni_error))

        # generate y with decision bounds
        y = np.zeros(self.n_samples)

        # decision boundary set by boundary for feature one
        for i in range(y.shape[0]):
            if X[i, 0] < x1_boundary:
                y[i] = 0
            else:
                y[i] = 1

        # ensure all values in X matrix >= 0
        for i, val in enumerate(X):
            for j, val2 in enumerate(val):
                if val2 < 0:
                    X[i, j] = 0

        return X, y

    def make_data(self, x1_boundary, x2_boundary, class_proportion, feature_dist, error_dist, **kwargs):
        """
        Generates X and y data for use in binary outcome estimation models.

        Available feature distributions:
            - uniform
            - normal

        Available error distributions:
            - uniform
            - normal

        Available kwargs:
            - low: the lower bound for the uniform error term
            - high: the upper bound for the uniform error term
            - mean: the mean for the normal error term
            - stdev: the standard deviation for the normal error term

        :param x1_boundary: the "true" decision boundary for feature 1
        :type x1_boundary: int

        :param x2_boundary: the "true" decision boundary for feature 2
        :type x2_boundary: int

        :param class_proportion: the proportion of samples in each class
        :type class_proportion: float

        :param feature_dist: the distribution to use for feature value generation
        :type feature_dist: str

        :param error_dist: the distribution to use for error term generation
        :type error_dist: str

        :param kwargs: additional parameters for uniform error generation
        :type kwargs: dict

        :return: X, y
        :rtype: np.array, np.array
        """

        error_dict = {
            "uniform": np.random.uniform,
            "normal": np.random.normal
        }

        # generate features
        def gen_features(boundary, upper, size, dist=feature_dist):
            if dist == "normal":
                if upper:
                    return np.random.normal(boundary + boundary / 2, opt_stdev(boundary), size)
                else:
                    return np.random.normal(x1_boundary - x1_boundary / 2, opt_stdev(x1_boundary), size)
            elif dist == "uniform":
                if upper:
                    return np.random.uniform(boundary, boundary * 2,  size)
                else:
                    return np.random.uniform(0, boundary, size)
            else:
                raise ValueError("feature_dist must be 'uniform' or 'normal'")

        def opt_stdev(boundary):
            """
            :param boundary: the artificial decision boundary for the feature
            :return: the standard deviation to use in creating data
            """
            boundary = int(boundary)
            return int(boundary // (boundary - boundary / 2))

        # create error term
        if error_dist == "uniform":
            error = error_dict[error_dist](kwargs["low"], kwargs["high"], self.n_samples).reshape(-1, 1)
        else:
            error = error_dict[error_dist](kwargs["mean"], kwargs["stdev"], self.n_samples).reshape(-1, 1)

        # calculate appropriate class size
        class_1_size = int(self.n_samples * class_proportion)
        class_2_size = self.n_samples - class_1_size

        # distribute x1 around the boundary
        feature_1_class_1 = gen_features(x1_boundary, upper=False, size=class_1_size)
        feature_1_class_2 = gen_features(x1_boundary, upper=True, size=class_2_size)

        # distribute x2 around the boundary
        feature_2_class_1 = gen_features(x2_boundary, upper=False, size=class_1_size)
        feature_2_class_2 = gen_features(x2_boundary, upper=True, size=class_2_size)

        # merge features
        x1 = np.concatenate((feature_1_class_1, feature_1_class_2)).reshape(-1, 1)
        x2 = np.concatenate((feature_2_class_1, feature_2_class_2)).reshape(-1, 1)
        ones_column = np.ones((self.n_samples, 1))

        # create feature matrix
        X = np.column_stack((ones_column, x1+error, x2+error))

        # generate y with decision bounds
        y = np.zeros(self.n_samples)

        # decision boundary set by boundary for feature one
        for i in range(y.shape[0]):
            if X[i, 0] < x1_boundary:
                y[i] = 0
            else:
                y[i] = 1

        # ensure all values in X matrix >= 0
        for i, val in enumerate(X):
            for j, val2 in enumerate(val):
                if val2 < 0:
                    X[i, j] = 0

        return X, y


if __name__ == '__main__':
    data_maker = DataMaker(1000, 4)
    X, y = data_maker.make_data(20, 30, .4, "normal", "normal", mean=8, stdev=2)
    print(X.shape)
    print(y.shape)
