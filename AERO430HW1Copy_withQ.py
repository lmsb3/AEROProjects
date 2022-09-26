import numpy as np
import matplotlib.pyplot as plt


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


def q_dot_calc_analytical(C, D, alphas):
    return -k*Ac*alphas*(C*np.cosh(alphas*L)+D*np.sinh(alphas*L))


def q_dot_calc_FDM(T, dx, a):
    return -k*Ac*((T[-1] - T[-2])/dx + a**2*dx*T[-1]/2)


def analytical_sol(C, D, alphas, case_num, plot=True, x_len=100):
    x = np.array([np.linspace(0, 1, x_len)])
    T = C*np.sinh(alphas.T*x)+D*np.cosh(alphas.T*x)
    qps = q_dot_calc_analytical(C, D, alphas.T)
    print(qps)
    if plot:
        plt.figure()
        plt.title('Analytical, Case {}'.format(case_num))
        plt.xlabel('X')
        plt.ylabel('Temperature')
        plt.plot(np.array([np.linspace(0, 1, x_len)] * len(T)).T, T.T)
        plt.legend(['a = {}'.format(a) for a in alphas[0]])
    return x, T, qps


def case1_FDM(n, a, find_q=False):
    dx = 1 / 2**n
    kap = 2 + a**2 * dx**2
    fd_mat = np.eye(2 ** n - 1) * kap + np.roll(np.eye(2 ** n) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n) * -1, -(2 ** n))[0:-1, 0:-1]
    fd_mat[0][1] = -1
    b = np.zeros((2**n-1, 1))
    b[-1, 0] = 100
    T = np.dot(np.linalg.inv(fd_mat), b)
    T = np.concatenate((np.array([0]), T.T[0], np.array([100])))
    x = np.linspace(0, 1, 2**n+1)
    qp = q_dot_calc_FDM(T, dx, a)
    return (x, T, qp) if find_q else (x, T)


def case2_FDM(n, a, find_q=False):
    dx = 1 / 2**n
    kap = 2 + a**2 * dx**2
    k_p = kap/2

    fd_mat = np.eye(2 ** n) * kap + np.roll(np.eye(2 ** n + 1) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n + 1) * -1, -(2 ** n + 1))[0:-1, 0:-1]
    fd_mat[0, 0] = k_p
    b = np.zeros((2 ** n, 1))
    b[-1, 0] = 100
    T = np.dot(np.linalg.inv(fd_mat), b)
    T = np.concatenate((T.T[0], np.array([100])))
    x = np.linspace(0, 1, 2 ** n + 1)
    qp = q_dot_calc_FDM(T, dx, a)
    return (x, T, qp) if find_q else (x, T)


def case3_FDM(n, a, find_q=False):
    dx = 1 / 2**n
    kap = 2 + a**2 * dx**2
    h = a ** 2 * k * R / 2
    k_p = ((dx * h) / k) + (kap / 2)

    fd_mat = np.eye(2 ** n) * kap + np.roll(np.eye(2 ** n + 1) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n + 1) * -1, -(2 ** n + 1))[0:-1, 0:-1]
    fd_mat[0, 0] = k_p
    b = np.zeros((2 ** n, 1))
    b[-1, 0] = 100
    T = np.dot(np.linalg.inv(fd_mat), b)
    T = np.concatenate((T.T[0], np.array([100])))
    x = np.linspace(0, 1, 2 ** n + 1)
    qp = q_dot_calc_FDM(T, dx, a)

    return (x, T, qp) if find_q else (x, T)


def FDM_mult_alphas(case_func, n, alphas, find_q=False, plot=False, case_num=1):
    results = []
    if plot:
        plt.figure()
        plt.ylabel('Temperature')
        plt.xlabel('Distance')
        plt.title('FDM Case {} n = {}'.format(case_num, n))

    for alpha in alphas:
        result = case_func(n, alpha, find_q=find_q)
        plt.plot(result[0], result[1], label='a = {}'.format(alpha)) if plot else ''
        results.append(result)

    if plot:
        plt.legend(loc='lower right')

    return np.array(results)


def FDM_mult_n(case_func, ns, alpha, find_q=False, plot=False, case_num=1):
    results = []
    if plot:
        plt.figure()
        plt.legend(loc='upper left')
        plt.ylabel('Temperature')
        plt.xlabel('Distance')
        plt.title('FDM Case {} a = {}'.format(case_num, alpha))

    for n in ns:
        result = case_func(n, alpha, find_q=find_q)
        plt.plot(result[0], result[1], label='n = {}'.format(n)) if plot else ''
        results.append(result)

    return results


def roc_to_extrap(values):
    q_mesh = values[:-2]
    q_half_mesh = values[1:-1]
    q_quarter_mesh = values[2:]
    extrap = (q_half_mesh**2 - q_mesh*q_quarter_mesh)/(2*q_half_mesh-q_quarter_mesh-q_mesh)
    roc = np.log2((extrap-q_mesh)/(extrap-q_half_mesh))
    return roc, np.abs((extrap-values[:-2])/extrap)


def roc_to_exact(values, exact):
    return np.log2((exact-values[:-1])/(exact-values[1:])), np.abs(exact-values)/np.abs(exact)


if __name__ == '__main__':
    # Bar Paramaters
    k = 0.5  # Thermal conductivity
    R = 0.1  # Cross section radius
    a = 4
    Ac = np.pi * R ** 2  # Cross sectional area
    L = 1  # Bar length
    A_surface = 2 * np.pi * R * L
    alphas = np.array([[0.25, 0.5, 0.55, 0.75, 0.85, 1, 2, 4, 6, 6.5, 8, 10, 15]])
    h = alphas.T ** 2 * k * R / 2
    alpha = 4
    ns = np.array([2, 3, 4, 5, 6, 7, 8])
    n = 4
    dxs = 1 / 2 ** ns
    dx = 1/2**n
    plotting = input('Would you like to show plots? (y/n): ')
    plotting = True if plotting == 'y' else False

    # Problem 1: Analytical Solution
    # Case 1
    C_c1 = 100/np.sinh(alphas.T*L)
    D_c1 = 0
    T1_ana, q1_ana = analytical_sol(C_c1, D_c1, alphas, 1, plot=plotting)[1:]

    # Case 2
    C_c2 = 0
    D_c2 = 100/np.cosh(alphas.T*L)
    T2_ana, q2_ana = analytical_sol(C_c2, D_c2, alphas, 2, plot=plotting)[1:]

    # Case 3
    C_c3 = h / k / alphas.T * (100 / (h / k / alphas.T * np.sinh(L * alphas.T) + np.cosh(L * alphas.T)))
    D_c3 = 100 / (h / k / alphas.T * np.sinh(L * alphas.T) + np.cosh(L * alphas.T))
    T3_ana, q3_ana = analytical_sol(C_c3, D_c3, alphas, 3, plot=plotting)[1:]

    # Problem 2: FDM Solution
    # Case 1 FDM Plotting
    c1 = FDM_mult_alphas(case1_FDM, n, alphas[0], plot=plotting, case_num=1)
    # Case 2 FDM Plotting
    c2 = FDM_mult_alphas(case2_FDM, n, alphas[0], plot=plotting, case_num=2)
    # Case 3 FDM Plotting
    c3 = FDM_mult_alphas(case3_FDM, n, alphas[0], plot=plotting, case_num=3)

    # Richardson for Problem 2
    q1 = np.array([result[-1]
                   for result in FDM_mult_n(case1_FDM, ns, alpha, find_q=True)])
    roc_q1, perr_q1 = roc_to_extrap(q1)
    roc_q1_exact, perr_q1_exact = roc_to_exact(q1, q1_ana[7])

    T02_fdm, q2 = np.array([np.array([result[1][0], result[-1]])
                            for result in FDM_mult_n(case2_FDM, ns, alpha, find_q=True)]).T
    roc_q2, perr_q2 = roc_to_extrap(q2)
    roc_q2_exact, perr_q2_exact = roc_to_exact(q2, q2_ana[7])
    roc_T2, perr_T2 = roc_to_extrap(T02_fdm)
    roc_T2_exact, perr_T2_exact = roc_to_exact(T02_fdm, T2_ana[7][0])

    T03_fdm, q3 = np.array([np.array([result[1][0], result[-1]])
                            for result in FDM_mult_n(case3_FDM, ns, alpha, find_q=True)]).T
    roc_q3, perr_q3 = roc_to_extrap(q3)
    roc_q3_exact, perr_q3_exact = roc_to_exact(q3, q3_ana[7])
    roc_T3, perr_T3 = roc_to_extrap(T03_fdm)
    roc_T3_exact, perr_T3_exact = roc_to_exact(T03_fdm, T3_ana[7][0])


    if plotting:
        # plt.figure()
        # plt.title('Rate of Convergence of q to extrapolated q')
        # plt.plot(range(len(roc_q1)), roc_q1, label='Case 1')
        # plt.plot(range(len(roc_q2)), roc_q2, '--', label='Case 2')
        # plt.plot(range(len(roc_q3)), roc_q3, ':', label='Case 3')
        # plt.legend()
        # plt.figure()
        # plt.title('Rate of Convergence of q to exact')
        # plt.plot(range(len(roc_q1_exact)), roc_q1_exact, label='Case 1')
        # plt.plot(range(len(roc_q2_exact)), roc_q2_exact, '--', label='Case 2')
        # plt.plot(range(len(roc_q3_exact)), roc_q3_exact, ':', label='Case 3')
        # plt.legend()
        # plt.figure()
        # plt.title('Rate of Convergence of T(0) to extrapolated T(0)')
        # plt.plot(range(len(roc_q2)), roc_T2, label='Case 2')
        # plt.plot(range(len(roc_q3)), roc_T3, '--', label='Case 3')
        # plt.legend()
        # plt.figure()
        # plt.title('Rate of Convergence of T(0) to exact T(0)')
        # plt.plot(range(len(roc_T2_exact)), roc_T2_exact, label='Case 2')
        # plt.plot(range(len(roc_T3_exact)), roc_T3_exact, '--', label='Case 3')
        # plt.legend()

        plt.figure()
        plt.title('Rate of Convergence of q to extrapolated q')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs[:-2], perr_q1, label='Case 1')
        plt.loglog(dxs[:-2], perr_q2, '--', label='Case 2')
        plt.loglog(dxs[:-2], perr_q3, ':', label='Case 3')
        plt.legend()
        plt.figure()
        plt.title('Rate of Convergence of q to exact')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs, perr_q1_exact, label='Case 1')
        plt.loglog(dxs, perr_q2_exact, '--', label='Case 2')
        plt.loglog(dxs, perr_q3_exact, ':', label='Case 3')
        plt.legend()
        plt.figure()
        plt.title('Rate of Convergence of T(0) to extrapolated T(0)')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs[:-2], perr_T2, label='Case 2')
        plt.loglog(dxs[:-2], perr_T3, '--', label='Case 3')
        plt.legend()
        plt.figure()
        plt.title('Rate of Convergence of T(0) to exact T(0)')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs, perr_T2_exact, label='Case 2')
        plt.loglog(dxs, perr_T3_exact, '--', label='Case 3')
        plt.legend()
        plt.show()

    # Printing Data
    print('Alpha Values:\n{}'.format(alphas))
    print('-'*51)
    print('-'*15, 'Analytical Solution', '-'*15)
    print('-'*51)
    x = np.array([np.linspace(0, 1, 11)])
    print('X values: \n{}'.format(x[0]))
    T1 = analytical_sol(C_c1, D_c1, alphas, 1, x_len=10)
    print('Temperature Case 1:\n{}'.format(T1))
    T2 = analytical_sol(C_c2, D_c2, alphas, 2, x_len=10)
    print('Temperature Case 2:\n{}'.format(T2))
    T3 = analytical_sol(C_c3, D_c3, alphas, 3, x_len=10)
    print('Temperature Case 3:\n{}'.format(T3))

    print('-' * 58)
    print('-' * 11, 'Finite Difference Method Solution', '-' * 11)
    print('-' * 58)
    print('Alpha = 4')
    x, T1_fdm = case1_FDM(2, 4)
    T2_fdm = case2_FDM(2, 4)[-1]
    T3_fdm = case3_FDM(2, 4)[-1]
    print('X values:\n{}'.format(x))
    print('Temperature Case 1:\n{}'.format(T1_fdm))
    print('Temperature Case 2:\n{}'.format(T2_fdm))
    print('Temperature Case 3:\n{}'.format(T3_fdm))

    print('-' * 58)
    print('-' * 18, 'Convergence of T(0)', '-' * 18)
    print('-' * 58)
    print('T(0) Case 2 exact: {}'.format(T2[0]))
    print('T(0) Case 2 FDM:\n{}'.format(T02_fdm))
    print('T(0) Case 3 exact: {}'.format(T3[0]))
    print('T(0) Case 3 FDM:\n{}'.format(T03_fdm))

    # Problem 3
    # Ais = np.array([np.array(range(2**n+1)) % 2 * 2.0 + 2 for n in ns], dtype=object)
    # for i in range(len(Ais)):
    #     Ais[i][0] = 1
    #     Ais[i][-1] = 1
    # Ais *= dxs/3.0
    # h_p3 = 4 ** 2 * k * Ac / A_surface * dxs
    # # Case 1
    # T1_fdms = np.array([result[1] for result in FDM_mult_n(case1_FDM, ns, 4)], dtype=object)
    # qp_1 = np.array([2 * np.pi * h_p3[i] * np.dot(Ais[i], T1_fdms[i]) for i in range(len(ns))])
    # b, pe = roc_to_extrap(qp_1)
    # plt.figure()
    # plt.loglog(dxs[2:], pe)
    #
    #
    # # Case 2
    # T2_fdms = np.array([result[1] for result in FDM_mult_n(case2_FDM, ns, 4)], dtype=object)
    # # 2 * np.pi * h[5][0] * np.dot(Ai, T2_fdm)
    # qp_2 = np.array([2 * np.pi * h_p3[i] * np.dot(Ais[i], T2_fdms[i]) for i in range(len(ns))])
    # b, pe = roc_to_extrap(qp_2)
    # plt.figure()
    # plt.loglog(dxs[2:], pe)
    #
    # # Case 3
    # T3_fdms = np.array([result[1] for result in FDM_mult_n(case3_FDM, ns, 4)], dtype=object)
    # # 2 * np.pi * h[5][0] * np.dot(Ai, T3_fdm)
    # qp_3 = np.array([2 * np.pi * h_p3[i] * np.dot(Ais[i], T3_fdms[i]) for i in range(len(ns))])
    # b, pe = roc_to_extrap(qp_3)
    # plt.figure()
    # plt.loglog(dxs[2:], pe)
