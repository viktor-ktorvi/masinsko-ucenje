import numpy as np
from matplotlib import pyplot as plt
from locally_weighted_lr import get_W
from data_loading import add_bias


def get_example(size):
    # return np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=size)
    return np.linspace(start=-2 * np.pi, stop=2 * np.pi, num=size)


if __name__ == '__main__':
    x_train = get_example(size=500)
    y_train = np.exp(x_train) + 0.1 * np.random.randn(len(x_train))

    x_train = x_train.reshape((len(x_train), 1))

    plt.figure()
    plt.grid()
    plt.plot(x_train, y_train, 'o')
    plt.title('Example')
    plt.xlabel('x')
    plt.ylabel('y')

    x_samples = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=100)
    x_samples = x_samples.reshape((len(x_samples), 1))
    y_samples = np.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        W = get_W(x_samples[i].reshape(x_samples[i].shape[0], 1), x_train, tau=0.1)

        X = add_bias(x_train)

        # Mora da se deli sa 4, ne znam zasto
        theta = ((X.T @ W @ X) ** (-1)) @ X.T @ W @ y_train / 4
        X_sample = add_bias(x_samples[i].reshape(x_samples[i].shape[0], 1))
        y_samples[i] = X_sample @ theta

    plt.plot(x_samples, y_samples, 'rx')

    plt.show()

    # TODO Probati za 2D