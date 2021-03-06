import numpy as np
from matplotlib import pyplot as plt
from data_loading import load_data, add_bias


def get_W(x_sample, X_train, tau):
    W = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        W[i, i] = np.exp(-np.sum((x_sample - X_train[i, :]) ** 2) / 2 / tau ** 2)

    return W


def lwlr(x_samples, x_train, y_train, tau):
    y_samples = np.zeros(x_samples.shape[0])
    for i in range(x_samples.shape[0]):
        W = get_W(x_samples[i, np.newaxis], x_train, tau=tau)

        X_train = add_bias(x_train)

        theta = np.linalg.inv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train
        # theta = np.linalg.inv(X_train.T.dot(W).dot(X_train)).dot(X_train.T).dot(W).dot(y_train)
        X_sample = add_bias(x_samples[i, np.newaxis])
        y_samples[i] = X_sample @ theta

    return y_samples


def hyperparameter_search():
    y_train, x_train = load_data('data_train.csv')
    y_val, x_val = load_data('data_val.csv')

    tau = np.logspace(start=-2, stop=0, num=100)
    rms = np.zeros_like(tau)
    for i in range(len(tau)):
        y_samples = lwlr(x_val, x_train, y_train, tau=tau[i])

        rms[i] = np.sqrt(np.sum((y_val - y_samples) ** 2) / len(y_val))

    plt.figure()
    plt.grid()
    plt.plot(tau, rms)
    plt.xscale('log')
    # plt.yscale('log')
    plt.title(r'RMS error with respect to $\tau$')
    plt.xlabel(r'$\tau$')
    plt.ylabel('rms error')
    plt.show()

    print('Lowest RMSError at tau_min_rms = {:5.3f}, RMSError(tau_min_rms) = {:2.2f}'.format(tau[np.argmin(rms)],
                                                                                             np.min(rms)))


if __name__ == '__main__':
    hyperparameter_search()
