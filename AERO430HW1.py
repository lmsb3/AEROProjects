import numpy as np
import matplotlib.pyplot as plt


def case1(n, a):
    dx = 1 / (2 ** n)
    kap = 2 + ((a ** 2) * (dx ** 2))
    fd_mat = np.eye(2 ** n - 1)*kap + np.roll(np.eye(2**n)*-1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2**n)*-1, -(2**n))[0:-1, 0:-1]
    fd_mat[0][1] = -1

    return fd_mat


def case2(n, a):
    dx = 1 / (2 ** n)
    kap = 2 + ((a ** 2) * (dx ** 2))
    h = a ** 2 * k * R / 2
    k_p = ((dx * h) / k) + (kap / 2)

    fd_mat = np.eye(2**n)*kap + np.roll(np.eye(2**n+1)*-1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2**n+1)*-1, -(2**n+1))[0:-1, 0:-1]
    fd_mat[0, 0] = k_p

    return fd_mat


# Setting up left hand matrix for Case 3
def case3(n, a):
    dx = 1 / (2 ** n)
    kap = 2 + ((a ** 2) * (dx ** 2))
    h = a ** 2 * k * R / 2
    k_p = ((dx * h) / k) + (kap / 2)

    fd_mat = np.eye(2 ** n) * kap + np.roll(np.eye(2 ** n + 1) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n + 1) * -1, -(2 ** n + 1))[0:-1, 0:-1]
    fd_mat[0, 0] = k_p

    return fd_mat


def simpson(a, b, n, input_array):
    sum = 0
    for k in range(n + 1):
        summand = input_array[k]
        if (k != 0) and (k != n):
            summand *= (2 + (2 * (k % 2)))
        sum += summand
    # sum = np.sum(
    #     [input_array[k] * (2 + (2 * (k % 2))) if (k != 0 and k != n) else input_array[k]] for k in range(n + 1)) possible alterante method
    return ((b - a) / (3 * n)) * sum


def case1_FDM(n, a):
    dx = 1 / (2 ** n)
    kap = 2 + ((a ** 2) * (dx ** 2))
    fd_mat = np.eye(2 ** n - 1) * kap + np.roll(np.eye(2 ** n) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n) * -1, -(2 ** n))[0:-1, 0:-1]
    fd_mat[0][1] = -1
    b = np.zeros((2**n-1, 1))
    b[-1, 0] = 100
    T = np.dot(np.linalg.inv(fd_mat), b)
    T = np.concatenate((np.array([0]), T.T[0], np.array([100])))
    x = np.linspace(0, 1, 2**n+1)
    return x, T




if __name__ == '__main__':
    # Probelm 3
    # Bar Paramaters
    k = 0.5  # Thermal conductivity
    R = 0.1  # Cross section radius
    Ac = np.pi * R ** 2  # Cross sectional area
    L = 1  # Bar length
    As = 2 * np.pi * R * L
    a = 4
    h = a ** 2 * k * R / 2
    case1_FDM(4, 6)
    # print(case3(3))