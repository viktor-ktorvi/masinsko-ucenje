import numpy as np
from domaci_1.data_loading import load_data, add_bias


def get_W(x_sample, X_train, tau):
    W = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        W[i, i] = np.exp(-np.sum((x_sample - X_train[i, :]) ** 2) / 2 / tau ** 2)

    return W


if __name__ == '__main__':
    y_train, X_train = load_data('data_train.csv')
    y_val, X_val = load_data('data_val.csv')

    x_sample = X_val[0, :]
    tau = 0.1
    W = get_W(x_sample, X_train, tau)
    X = add_bias(X_train)

    theta = (X.T @ W @ X) ** (-1) @ X.T @ W @ y_train

    y_sample = add_bias(x_sample.reshape(1, len(x_sample))) @ theta
