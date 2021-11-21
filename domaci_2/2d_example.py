import numpy as np
from matplotlib import pyplot as plt
from domaci_2.main import h, log_reg
if __name__ == '__main__':
    np.random.seed(26)
    N = 100
    x1 = 10 + np.random.randn(N, 2)
    y1 = np.ones(x1.shape[0])

    x2 = -2 + 2 * np.random.randn(N, 2)
    y2 = np.zeros(x2.shape[0])

    # plt.figure()
    # plt.plot(x1[:, 0], x1[:, 1], 'o')
    # plt.plot(x2[:, 0], x2[:, 1], 'o')
    # plt.show()

    x_train = np.vstack((x1, x2))
    y_train = np.concatenate((y1, y2))

    theta = np.random.randn(x_train.shape[1])
    d_theta = np.zeros_like(theta)


    theta = log_reg(x_train, y_train, epochs=100, alpha=1e-2, mb=100)


    y_infr = h(theta, x_train)
    y_infr[y_infr < 0.5] = 0
    y_infr[y_infr > 0.5] = 1
    print('Accuracy = {:.3f}'.format(np.sum(y_infr == y_train) / len(y_train)))

    plt.figure()
    plt.plot(x_train[y_infr == 1, 0], x_train[y_infr == 1, 1], 'ro')
    plt.plot(x_train[y_infr == 0, 0],x_train[y_infr == 0, 1], 'bo')