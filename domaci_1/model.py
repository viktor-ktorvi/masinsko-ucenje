from domaci_1.data_loading import load_data
from domaci_1.locally_weighted_lr import lwlr


def model(X):
    y_train, x_train = load_data('data_train.csv')

    return lwlr(X, x_train, y_train, tau=0.1)


if __name__ == '__main__':
    y_val, x_val = load_data('data_val.csv')

    y_ = model(x_val)
