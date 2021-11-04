import pandas as pd
import numpy as np


def load_data(filename):
    data = pd.read_csv(filename, header=None).to_numpy()

    y = data[:, -1].reshape(data.shape[0], 1)
    X = data[:, :-1]

    return y, X


def add_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def split_and_save(filename, train_ratio, val_ratio):
    data = pd.read_csv(filename, header=None).to_numpy()

    # shuffle the data rows
    rng = np.random.default_rng()
    rng.shuffle(data)

    train_len = np.round(data.shape[0] * train_ratio).astype(np.int32)
    val_len = np.round(data.shape[0] * val_ratio).astype(np.int32)

    data_train = data[0:train_len]
    data_val = data[train_len:train_len + val_len]
    data_test = data[train_len + val_len:]

    np.savetxt('data_train.csv', data_train, delimiter=',')
    np.savetxt('data_val.csv', data_val, delimiter=',')
    np.savetxt('data_test.csv', data_test, delimiter=',')


if __name__ == '__main__':
    filename = 'data.csv'
    train_ratio = 0.7
    val_ratio = 0.15

    split_and_save(filename, train_ratio, val_ratio)
