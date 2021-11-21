import numpy as np
import pandas as pd
from data_loading import split_and_save, load_data, add_bias


# LR, GDA

class Klasa:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def h(theta, x):
    # print(x @ theta)
    return 1 / (1 + np.exp(-x @ theta))


def log_reg(x_train, y_train, epochs, alpha, mb):
    theta = np.random.randn(x_train.shape[1])
    d_theta = np.zeros_like(theta)
    for i in range(epochs):
        n = 0
        while True:
            if n + mb < len(y_train):
                y = y_train[n:n + mb]
                x = x_train[n:n + mb, :]
            else:
                y = y_train[n:]
                x = x_train[n:, :]

            n += mb

            for j in range(len(d_theta)):
                d_theta[j] = (y_train - h(theta, x_train)).dot(x_train[:, j])

            theta += alpha * d_theta

            if n > len(y_train):
                break

    return theta


def accuracy(y_infr, y):
    y_inference = y_infr
    y_inference[y_inference < 0.5] = 0
    y_inference[y_inference >= 0.5] = 1

    return np.sum(y_inference == y) / len(y)


if __name__ == '__main__':
    y, x = load_data('data_train.csv')
    x = (x - np.mean(x)) / np.std(x)
    x = add_bias(x)

    i = 0
    j = 1
    k = 2

    target_class = Klasa(x=x[y == i], y=np.ones_like(y[y == i]))
    other_class = Klasa(x=x[np.logical_or(y == j, y == k)], y=np.zeros_like(y[np.logical_or(y == j, y == k)]))

    x_train = np.vstack((target_class.x, other_class.x))
    y_train = np.concatenate((target_class.y, other_class.y))

    theta = log_reg(x_train, y_train, epochs=100, alpha=1e-2, mb=100)
    print('Accuracy = {:.3f}'.format(accuracy(h(theta, x), y_train)))
