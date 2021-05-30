import numpy as np
import pandas as pd


class MNB:
    """
    Description
    -----------
    Multinomial Naive Bayes algorithm implementation
    """

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X_train, y_train):
        """
        Description
        -----------
          Train the model on X,y

        Parameters
        ----------
        X_train : array like (m,n)
          Set of features to train on

        y_train : array like (m,)
          Set of labels

        Returns
        -------
        None
        """
        self.classes = np.unique(y_train)
        self.log_priors = np.log(np.bincount(y_train) / len(y_train))

        t = pd.DataFrame(X_train)
        t['y'] = y_train
        sum_table = t.groupby(by='y').sum()
        self.log_likelihoods = np.log(
            ((sum_table.T + self.alpha) / (sum_table + self.alpha).sum(axis=1))).T.values

    def predict(self, X_test):
        """
        Description
        -----------
          Predict given set

        Parameters
        ----------
        X_test : array like (m,n)
          Set of features to be tested

        Returns
        -------
        list
          Predictions of X_test
        """
        return [self._predict(x_test) for x_test in X_test]

    def _predict(self, x_test):
        log_likelihoods_x = self.log_likelihoods * x_test
        posteriors_x = log_likelihoods_x.sum(axis=1) + self.log_priors
        return self.classes[np.argmax(posteriors_x)]
