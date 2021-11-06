import numpy as np
from data_loading import load_data
from locally_weighted_lr import lwlr


def model(X):
    y_train, x_train = load_data('data_train.csv')

    return lwlr(X, x_train, y_train, tau=0.102)


if __name__ == '__main__':
    y, x = load_data('data_test.csv')

    y_ = model(x)

    rms = np.sqrt(np.sum((y - y_) ** 2) / len(y))

    print('RMSError = {:2.2f}'.format(rms))
