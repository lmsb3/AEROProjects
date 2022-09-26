import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# Analytical solution
def temp_ana(x, y, k):
    return 100*np.sinh(k*np.pi*y)*np.sin(np.pi*x)/np.sinh(k*np.pi)


def temp_fdm_second_order(n, k):
    # Create A matrix
    A_dim = n**2
    A = -np.eye(A_dim)*2*(k**2+1) + np.eye(A_dim, k=1)*k**2 + np.eye(A_dim, k=-1)*k**2 + np.eye(A_dim, k=-n) + np.eye(A_dim, k=n)
    ind1 = np.array(range(A_dim-1))
    ind2 = np.array(range(1, A_dim))
    A[ind1[n-1::n], ind2[n-1::n]] = A[ind2[n-1::n], ind1[n-1::n]] = 0

    # Create B matrix
    b = np.zeros((A_dim, 1))
    b[:n, 0] = -100 * np.sin((np.array(range(n))+1)/(n+1) * np.pi)
    return solve(A, b)


def temp_fdm_fourth_order(n, k):
    # Craeting A
    A_dim = (n - 1) ** 2
    A = np.eye(A_dim) * 5 / 3 * (k ** 2 + 1)
    A += (np.eye(A_dim, k=1) + np.eye(A_dim, k=-1)) * (1 / 6 * (k ** 2 + 1) - k ** 2)
    A += (np.eye(A_dim, k=-(n - 1)) + np.eye(A_dim, k=(n - 1))) * (1 / 6 * (k ** 2 + 1) - 1)
    A += (np.eye(A_dim, k=-(n - 1) + 1) + np.eye(A_dim, k=(n - 1) - 1) + np.eye(A_dim, k=(n - 1) + 1) +
          np.eye(A_dim, k=-(n - 1) - 1)) * -1 / 12 * (k ** 2 + 1)

    ind1 = np.array(range(A_dim - 1))
    ind2 = ind1 + 1
    ind3 = np.array(range(A_dim - n + 2))
    ind4 = ind3 + n - 2
    ind5 = np.array(range(A_dim - n))
    ind6 = ind5 + n
    A[ind1[n - 2::n - 1], ind2[n - 2::n - 1]] = A[ind2[n - 2::n - 1], ind1[n - 2::n - 1]] = 0
    A[ind3[::n - 1], ind4[::n - 1]] = A[ind4[::n - 1], ind3[::n - 1]] = 0
    A[ind5[n - 2::n - 1], ind6[n - 2::n - 1]] = A[ind6[n - 2::n - 1], ind5[n - 2::n - 1]] = 0

    # Creating B
    C = 1 / 6 * (k ** 2 + 1) - 1
    D = -1 / 12 * (k ** 2 + 1)
    b = np.zeros((A_dim, 1))
    i_vals = np.array(range(n - 1))
    b[:n - 1, 0] = -100 * (C * np.sin(np.pi * (i_vals + 1) / n) + D * (
            np.sin(np.pi * i_vals / n) + np.sin(np.pi * (i_vals + 2) / n)))

    return solve(A, b)


def data_from_fdm(u, n):
    x = np.linspace(0, 1, n+2)
    y = np.linspace(0, 1, n+2)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n+2, n+2))
    Z[-1] = 100 * np.sin(np.pi * x)
    Z[1:-1, 1:-1] = u[::-1].reshape((n, n))
    return X, Y, Z


def heat_loss_ana(k):
    return -(200 * k * np.cosh(k * np.pi))/np.sinh(k*np.pi)


def heat_loss_fdm(U, n, order, k):
    if order == 2:
        dt_dy = -(U[-3, :] - 4*U[-2, :] + 3*U[-1, :])*n/2
    else:
        dt_dy = (-3*U[-5, :] + 16*U[-4, :] - 36*U[-3, :] + 48*U[-2, :] - 25*U[-1, :])*n/12
    q = 1/3/n*(dt_dy[0]+dt_dy[-1] + np.sum(dt_dy[1:-1:2]*4 + dt_dy[2::2]*2))
    return q


def roc_to_extrap(values):
    q_mesh = values[:-2]
    q_half_mesh = values[1:-1]
    q_quarter_mesh = values[2:]
    extrap = (q_half_mesh**2 - q_mesh*q_quarter_mesh)/(2*q_half_mesh-q_quarter_mesh-q_mesh)
    roc = np.log2(np.abs((extrap-q_mesh)/(extrap-q_half_mesh)))
    return roc, np.abs((extrap-values[2:])/extrap)


def roc_to_exact(values, exact):
    return np.log2(np.abs((exact-values[:-1])/(exact-values[1:]))), np.abs(exact-values)/np.abs(exact)


if __name__ == '__main__':
    K = np.array([0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10, 12, 14])
    print('K values:', K)
    ns = 2**np.array(range(2, 7))
    print('N values:', ns)
    print('dx values:', 1/ns)

    # ========== Analytical ==========
    print('='*20, 'Analytical', '='*20)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    temp_analytical = temp_ana(X, Y, K[0])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ana = ax.plot_surface(X, Y, temp_analytical, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature (deg C)')
    ax.set_title('Analytical, k={}'.format(K[0]))

    q_exact = heat_loss_ana(K)

    print('X values:', X[::5, 10])
    print('Y values:', Y[::5, 10])
    print('Temp values:', temp_analytical[::5, 10])

    print('Exact heat loss values:', q_exact, '\n')

    # ========== 2nd Order ==========
    print('=' * 20, '2nd Order', '=' * 20)
    q_second_order_n = np.zeros((len(ns)))
    plt.figure()
    plt.title('Second Order Rate of Convergence to Exact')
    plt.xlabel('dxs')
    plt.ylabel('Percent Error')
    plt.figure()
    plt.title('Second Order Rate of Convergence to Extrap')
    plt.xlabel('dxs')
    plt.ylabel('Percent Error')
    for i in range(len(K)):
        for n in ns:
            u = temp_fdm_second_order(n, K[i])
            X, Y, U = data_from_fdm(u, n)
            q_second_order_n[int(np.log2(n)) - 2] = heat_loss_fdm(U, n, 2, K[i])
            if n == 64 and K[i] == 0.1:
                fig2 = plt.figure()
                ax = fig2.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, U, cmap='coolwarm')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('Temperature (deg C)')
                ax.set_title('Second Order FDM, k={}, n={}'.format(K[i], n))
                print('For X, Y, and Temp, values are every 4th value at n={}\nX values:\n'.format(n), X[::4, 10])
                print('Y values:\n', Y[::4, 10])
                print('Temp values:\n', temp_analytical[::4, 10])
                print('Heat Loss value:', q_second_order_n[-1])
        roc_exact, perr_exact = roc_to_exact(q_second_order_n, q_exact[i])
        roc_extrap, perr_extrap = roc_to_extrap(q_second_order_n)
        plt.figure(2)
        plt.loglog(1/ns, perr_exact, '.-', label='k={}'.format(K[i]))
        plt.figure(3)
        plt.loglog(1/ns[2:], perr_extrap, '.-', label='k={}'.format(K[i]))

        print('k = {}:'.format(K[i]))
        if K[i] == 0.1:
            print('Rate of Convergence wrt to Exact:', roc_exact)
            print('Rate of Convergence wrt to Exact:', roc_extrap)
        print('Percent Error wrt Exact:', perr_exact)
        print('Percent Error wrt Extrapolated values:', perr_extrap)
        print('Heat Loss values for every N:', q_second_order_n, '\n')

    plt.figure(2)
    plt.legend()
    plt.figure(3)
    plt.legend()

    # ========== 4th Order ==========
    print('=' * 20, '4th Order', '=' * 20)
    q_fourth_order_n = np.zeros((len(ns)))
    plt.figure()
    plt.title('Fourth Order Rate of Convergence to Exact')
    plt.xlabel('dxs')
    plt.ylabel('Percent Error')
    plt.figure()
    plt.title('Fourth Order Rate of Convergence to Extrap')
    plt.xlabel('dxs')
    plt.ylabel('Percent Error')
    for i in range(len(K)):
        for j in range(len(ns)):
            n = ns[j]
            u = temp_fdm_fourth_order(n, K[i])
            X, Y, U = data_from_fdm(u, n-1)
            q_fourth_order_n[j] = heat_loss_fdm(U, n, 4, K[i])
            if n == 64 and K[i] == 0.1:
                fig2 = plt.figure()
                ax = fig2.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, U, cmap='coolwarm')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('Temperature (deg C)')
                ax.set_title('Fourth Order FDM, k={}, n={}'.format(K[i], n))
                print('For X, Y, and Temp, values are every 4th value at n={}\nX values:\n'.format(n), X[::4, 10])
                print('Y values:\n', Y[::4, 10])
                print('Temp values:\n', temp_analytical[::4, 10])
                print('Heat Loss value:', q_fourth_order_n[-1])
        roc_exact, perr_exact = roc_to_exact(q_fourth_order_n, q_exact[i])
        roc_extrap, perr_extrap = roc_to_extrap(q_fourth_order_n)
        plt.figure(5)
        plt.loglog(1 / ns, perr_exact, '.-', label='k={}'.format(K[i]))
        plt.figure(6)
        plt.loglog(1 / ns[2:], perr_extrap, '.-', label='k={}'.format(K[i]))
        print('k = {}:'.format(K[i]))
        if K[i] == 0.1:
            print('Rate of Convergence wrt to Exact:', roc_exact)
            print('Rate of Convergence wrt to Exact:', roc_extrap)
        print('Percent Error wrt Exact:', perr_exact)
        print('Percent Error wrt Extrapolated values:', perr_extrap)
        print('Heat Loss values for every N:', q_second_order_n, '\n')

    plt.figure(5)
    plt.legend()
    plt.figure(6)
    plt.legend()

    plt.show()
