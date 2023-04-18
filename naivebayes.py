import numpy as np
from scipy.stats import norm

class NaiveBayes:
    def __init__(self, num_cont_feat):
        self.num_cont_feat = num_cont_feat
    
    def fit(self, X, y):
        n_examples, n_features = X.shape

        y_labels, y_counts = np.unique(y, return_counts=True)

        # nd.array of dictionaries mapping values to the likelihood
        X_params = np.ndarray(shape=(len(y_labels), n_features), dtype=dict)
        # prior
        y_params = y_counts/n_examples

        # For every class label
        for i, y_label in enumerate(y_labels):
            # For every feature column
            for j, col in enumerate(X.T):
                if j < 6:
                    # gaussian distribution for continuous features.
                    mu = np.sum(col[y == y_label]) / y_counts[i]
                    sigma = np.sqrt(np.sum((col[y == y_label] - mu)**2) / y_counts[i])
                    thetas = np.array([mu, sigma])
                    x_labels = np.array(["mu", "sigma"])
                else:
                    x_labels, x_counts = np.unique(col[y == y_label], return_counts=True)

                    # MLE prob parameters with m-estimate
                    # p = 1/len(np.unique(col))
                    # m = 1/p
                    # thetas = (x_counts+m*p)/(y_counts[i]+m)

                    # MLE probability parameters without m-estimate
                    thetas = x_counts/y_counts[i]
                    thetas = np.log(thetas)

                # Store a dictionary as a parameter map
                # Which maps from the feature value to the MLE probability parameter
                X_params[i][j] = dict(zip(x_labels, thetas))

        return X_params, y_params

    def predict(self, X, X_params, y_params):
        y_hat = np.empty(len(X))
        # For each example x, find the most probable hypothesis
        for i, x in enumerate(X):
            # For each hypothesis (class label), consider the likelihood that the hypothesis holds for x
            hypotheses = np.copy(y_params)
            for j in range(len(hypotheses)):
                # For each feature in x, consider the likelihood that the hypothesis holds for that feature
                for k, feature in enumerate(x):
                    if k < self.num_cont_feat:
                        hypotheses[j] += norm.pdf(x[k],loc=X_params[j][k]["mu"],scale=X_params[j][k]["sigma"])
                    else:
                        try:
                            hypotheses[j] += X_params[j][k][feature]
                        except:
                            # if there is no probabilty for feature then add zero
                            hypotheses[j] += 0
            # Pick the most probable hypothesis
            y_hat[i] = np.argmax(hypotheses)

        return y_hat