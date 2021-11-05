import numpy as np
from matplotlib import pyplot as plt
from locally_weighted_lr import get_W, lwlr
from data_loading import add_bias
from mpl_toolkits.mplot3d import Axes3D


def example_1d():
    x_train = np.linspace(start=-2 * np.pi, stop=2 * np.pi, num=500)
    y_train = np.sin(x_train) + 0.1 * np.random.randn(len(x_train))

    x_train = x_train.reshape((len(x_train), 1))

    plt.figure()
    plt.grid()
    plt.plot(x_train, y_train, 'o')
    plt.title('Example')
    plt.xlabel('x')
    plt.ylabel('y')

    x_samples = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=100)
    x_samples = x_samples.reshape((len(x_samples), 1))

    y_samples = lwlr(x_samples, x_train, y_train, tau=0.1)
    plt.plot(x_samples, y_samples, 'rx')

    plt.show()


def example_2d():
    nx, ny = (40, 40)
    x = np.linspace(-2 * np.pi, 2 * np.pi, nx)
    y = np.linspace(-2 * np.pi, 2 * np.pi, ny)
    X, Y = np.meshgrid(x, y)

    z = np.cos(X) + np.cos(Y) + 0.1 * np.random.randn(nx, ny)
    # z = np.cos(X + Y) + 0.01 * np.random.randn(nx, ny)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, z)
    # ax.plot_surface(xv, yv, np.cos(xv) + np.cos(yv), color='r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x_samples = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(500, 2))
    x_train = np.hstack((X.flatten().reshape(nx * ny, 1), Y.flatten().reshape(nx * ny, 1)))
    y_train = z.flatten()

    y_samples = lwlr(x_samples, x_train, y_train, tau=0.1)

    ax.scatter(x_samples[:, 0], x_samples[:, 1], y_samples, color='r')

    plt.show()


if __name__ == '__main__':
    example_1d()
    example_2d()
