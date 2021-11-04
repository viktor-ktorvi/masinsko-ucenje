import pandas as pd
import numpy as np


def load_data(filename):
    data = pd.read_csv(filename, header=None).to_numpy()

    y = data[:, -1].reshape(data.shape[0], 1)
    X = data[:, :-1]
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    return y, X


if __name__ == '__main__':
    filename = 'data.csv'

    y, X = load_data(filename)
