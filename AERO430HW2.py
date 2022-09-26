import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def q_dot_calc_analytical(C, D, alphas):
    return -k*Ac*alphas*(C*np.cosh(alphas*L)+D*np.sinh(alphas*L))


def q_dot_calc_FDM(T, dx, a, order=4):
    if order == 4:
        return -k*Ac*((T[-1]*(1 + dx**2/2*a**2 + dx**4/24*a**4) - T[-2])/(dx + dx**3/6*a**2))
    elif order == 2:
        return -k*Ac*((T[-1] - T[-2])/dx + a**2*dx*T[-1]/2)
    elif order == 1:
        return -k*Ac*((T[-1] - T[-2])/dx)


def analytical_sol(C, D, alphas, case_num, plot=True, x_len=1000):
    x = np.array([np.linspace(0, 1, x_len)])
    T = C*np.sinh(alphas.T*x)+D*np.cosh(alphas.T*x)
    qps = q_dot_calc_analytical(C, D, alphas.T)
    if plot:
        plt.figure()
        plt.title('Analytical, Case {}'.format(case_num))
        plt.xlabel('Distance (cm)')
        plt.ylabel('Temperature (deg C)')
        plt.plot(np.array([np.linspace(0, 1, x_len)] * len(T)).T, T.T)
        plt.legend(['a = {}'.format(a) for a in alphas[0]])
    return x, T, qps


def case1_FDM(n, a, find_q=False):
    dx = 1 / 2**n
    kap = (-24-10*dx**2*a**2)/(-12+dx**2*a**2)
    fd_mat = np.eye(2 ** n - 1) * kap + np.roll(np.eye(2 ** n) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n) * -1, -(2 ** n))[0:-1, 0:-1]
    fd_mat[0][1] = -1
    b = np.zeros((2**n-1, 1))
    b[-1, 0] = 100
    T = np.dot(np.linalg.inv(fd_mat), b)
    T = np.concatenate((np.array([0]), T.T[0], np.array([100])))
    x = np.linspace(0, 1, 2**n+1)
    qp4 = q_dot_calc_FDM(T, dx, a)
    qp2 = q_dot_calc_FDM(T, dx, a, order=2)
    qp1 = q_dot_calc_FDM(T, dx, a, order=1)
    return (x, T, qp1, qp2, qp4) if find_q else (x, T)


def case2_FDM(n, a, find_q=False):
    dx = 1 / 2**n
    kap = (-24-10*dx**2*a**2)/(-12+dx**2*a**2)
    k_p = kap/2

    fd_mat = np.eye(2 ** n) * kap + np.roll(np.eye(2 ** n + 1) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(2 ** n + 1) * -1, -(2 ** n + 1))[0:-1, 0:-1]
    fd_mat[0, 0] = k_p
    b = np.zeros((2 ** n, 1))
    b[-1, 0] = 100
    T = np.dot(np.linalg.inv(fd_mat), b)
    T = np.concatenate((T.T[0], np.array([100])))
    x = np.linspace(0, 1, 2 ** n + 1)
    qp4 = q_dot_calc_FDM(T, dx, a)
    qp2 = q_dot_calc_FDM(T, dx, a, order=2)
    qp1 = q_dot_calc_FDM(T, dx, a, order=1)
    return (x, T, qp1, qp2, qp4) if find_q else (x, T)


def FDM_mult_alphas(case_func, n, alphas, find_q=False, plot=False, case_num=1):
    results = []
    if plot:
        plt.figure()
        plt.ylabel('Temperature (deg C)')
        plt.xlabel('Distance (cm)')
        plt.title('FDM Case {} dx = {}'.format(case_num, 1/2**n))

    for alpha in alphas:
        result = case_func(n, alpha, find_q=find_q)
        plt.plot(result[0], result[1], label='a = {}'.format(alpha)) if plot else ''
        results.append(result)

    if plot:
        plt.legend(loc='lower right')

    return np.array(results)


def FDM_mult_n(case_func, ns, alpha, find_q=False, plot=False, case_num=1, axs = None):
    results = []

    for i in range(len(ns)):
        n = ns[i]
        result = case_func(n, alpha, find_q=find_q)
        if plot:
            axs[i].plot(result[0], result[1], label='dx = {}'.format(1/2**n))
        results.append(result)

    return results


def roc_to_extrap(values, case_num=1):
    q_mesh = values[:-2]
    q_half_mesh = values[1:-1]
    q_quarter_mesh = values[2:]
    extrap = (q_half_mesh**2 - q_mesh*q_quarter_mesh)/(2*q_half_mesh-q_quarter_mesh-q_mesh)
    roc = np.log2((extrap-q_mesh)/(extrap-q_half_mesh))
    print('Extrapolated values:\n{}'.format(extrap))
    return roc, np.abs((extrap-values[2:])/extrap)


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
    dxs = 1/2**ns
    dx = 1/2**n
    plotting = input('Would you like to show plots? (y/n): ')
    plotting = True if plotting.lower() == 'y' else False

    # Problem 1: Analytical Solution
    # Case 1
    C_c1 = 100/np.sinh(alphas.T*L)
    D_c1 = 0
    T1_ana, q1_ana = analytical_sol(C_c1, D_c1, alphas, 1, plot=plotting)[1:]

    # Case 2
    C_c2 = 0
    D_c2 = 100/np.cosh(alphas.T*L)
    T2_ana, q2_ana = analytical_sol(C_c2, D_c2, alphas, 2, plot=plotting)[1:]

    # Problem 2: FDM Solution
    # Case 1 FDM Plotting
    c1 = FDM_mult_alphas(case1_FDM, n, alphas[0], plot=plotting, case_num=1)
    # Case 2 FDM Plotting
    c2 = FDM_mult_alphas(case2_FDM, n, alphas[0], plot=plotting, case_num=2)

    # Richardson for Problem 2
    # Case 1 FDM
    T1_fdm_half, q1_1, q1_2, q1_4 = np.array([np.array([result[1][int(len(result[1])/2)], result[-3], result[-2], result[-1]])
                                              for result in FDM_mult_n(case1_FDM, ns, alpha, find_q=True)]).T
    print('q Case 1 4th Order ', end='')
    roc_q1_4, perr_q1_4 = roc_to_extrap(q1_4)
    roc_q1_exact_4, perr_q1_exact_4 = roc_to_exact(q1_4, q1_ana[7])
    print('q Case 1 2nd Order ', end='')
    roc_q1_2, perr_q1_2 = roc_to_extrap(q1_2)
    roc_q1_exact_2, perr_q1_exact_2 = roc_to_exact(q1_2, q1_ana[7])
    print('q Case 1 1st Order ', end='')
    roc_q1_1, perr_q1_1 = roc_to_extrap(q1_1)
    roc_q1_exact_1, perr_q1_exact_1 = roc_to_exact(q1_1, q1_ana[7])
    print('T(1/2) Case 1 ', end='')
    roc_T1, perr_T1 = roc_to_extrap(T1_fdm_half)
    T1_exact = case1_FDM(9, 4.0)[1]
    roc_T1_exact, perr_T1_exact = roc_to_exact(T1_fdm_half, T1_exact[int(len(T1_exact)/2)])

    # Case 2 FDM
    T02_fdm, q2_1, q2_2, q2_4 = np.array([np.array([result[1][int(len(result[1])/2)], result[-3], result[-2], result[-1]])
                                          for result in FDM_mult_n(case2_FDM, ns, alpha, find_q=True)]).T
    print('q Case 2 4th Order ', end='')
    roc_q2_4, perr_q2_4 = roc_to_extrap(q1_4)
    roc_q2_exact_4, perr_q2_exact_4 = roc_to_exact(q1_4, q1_ana[7])
    print('q Case 2 2nd Order ', end='')
    roc_q2_2, perr_q2_2 = roc_to_extrap(q1_2)
    roc_q2_exact_2, perr_q2_exact_2 = roc_to_exact(q1_2, q1_ana[7])
    print('q Case 2 1st Order ', end='')
    roc_q2_1, perr_q2_1 = roc_to_extrap(q1_1)
    roc_q2_exact_1, perr_q2_exact_1 = roc_to_exact(q1_1, q1_ana[7])
    print('T(1/2) Case 2 ', end='')
    roc_T2, perr_T2 = roc_to_extrap(T02_fdm)
    T2_exact = case2_FDM(9, 4.0)[1]
    roc_T2_exact, perr_T2_exact = roc_to_exact(T02_fdm, T2_exact[int(len(T2_exact)/2)])

    if plotting:
        plt.figure()
        plt.title('Case 1 Rate of Convergence to extrap of q for different order Taylor Series')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs[2:], perr_q1_4, label='4th Order')
        plt.loglog(dxs[2:], perr_q1_2, label='2nd Order')
        plt.loglog(dxs[2:], perr_q1_1, label='1st Order')
        plt.legend()

        plt.figure()
        plt.title('Case 1 Rate of Convergence to exact of q for different order Taylor Series')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs, perr_q1_exact_4, label='4th Order')
        plt.loglog(dxs, perr_q1_exact_2, label='2nd Order')
        plt.loglog(dxs, perr_q1_exact_1, label='1st Order')
        plt.legend()

        plt.figure()
        plt.title('Case 2 Rate of Convergence to extrap of q for different order Taylor Series')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs[2:], perr_q2_4, label='4th Order')
        plt.loglog(dxs[2:], perr_q2_2, label='2nd Order')
        plt.loglog(dxs[2:], perr_q2_1, label='1st Order')
        plt.legend()

        plt.figure()
        plt.title('Case 2 Rate of Convergence to exact of q for different order Taylor Series')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs, perr_q2_exact_4, label='4th Order')
        plt.loglog(dxs, perr_q2_exact_2, label='2nd Order')
        plt.loglog(dxs, perr_q2_exact_1, label='1st Order')
        plt.legend()

        plt.figure()
        plt.title('Rate of Convergence of q to extrapolated q')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs[2:], perr_q1_4, label='Case 1')
        plt.loglog(dxs[2:], perr_q2_4, '--', label='Case 2')
        plt.legend()

        plt.figure()
        plt.title('Rate of Convergence of q to exact')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs, perr_q1_exact_4, label='Case 1')
        plt.loglog(dxs, perr_q2_exact_4, '--', label='Case 2')
        plt.legend()

        plt.figure()
        plt.title('Rate of Convergence of T(1/2) to extrapolated T(1/2)')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs[2:], perr_T1, label='Case 1')
        plt.loglog(dxs[2:], perr_T2, '--', label='Case 2')
        plt.legend()

        plt.figure()
        plt.title('Rate of Convergence of T(1/2) to exact T(1/2)')
        plt.xlabel('dx')
        plt.ylabel('Percent Error')
        plt.loglog(dxs, perr_T1_exact, label='Case 1')
        plt.loglog(dxs, perr_T2_exact, '--', label='Case 2')
        plt.legend()

        fig, axs = plt.subplots(len(alphas[0]), len(ns), sharex='all', sharey='all')
        fig2, axs2 = plt.subplots(len(alphas[0]), len(ns), sharex='all', sharey='all')
        for i in range(len(alphas[0])):
            a = alphas[0][i]
            FDM_mult_n(case1_FDM, ns, a, case_num=1, plot=True, axs=axs[i])
            FDM_mult_n(case2_FDM, ns, a, case_num=2, plot=True, axs=axs2[i])
        plt.show()

    # Printing Data
    np.set_printoptions(suppress=True)
    print('Alpha Values:\n{}'.format(alphas))
    print('-'*51)
    print('-'*15, 'Analytical Solution', '-'*15)
    print('-'*51)
    x = np.array([np.linspace(0, 1, 11)])
    print('X values: \n{}'.format(x[0]))
    T1 = analytical_sol(C_c1, D_c1, alphas, 1, x_len=11, plot=False)[1]
    print('Temperature Case 1:\n{}'.format(T1))
    T2 = analytical_sol(C_c2, D_c2, alphas, 2, x_len=11, plot=False)[1]
    print('Temperature Case 2:\n{}'.format(T2))
    print('q Case 1:\n{}'.format(q1_ana))
    print('q Case 2:\n{}'.format(q2_ana))
    table1 = [['Temperature Case 1', 'Temperature Case 2'], T1, T2]
    # print(tabulate(T1.T, headers=alphas[0]))
    # print(tabulate(T2.T, headers=alphas[0]))

    print('-' * 58)
    print('-' * 11, 'Finite Difference Method Solution', '-' * 11)
    print('-' * 58)
    print('Alpha = 4')
    x, T1_fdm = case1_FDM(2, 4)
    T2_fdm = case2_FDM(2, 4)[-1]
    print('X values:\n{}'.format(x))
    print('Temperature Case 1:\n{}'.format(T1_fdm))
    print('Temperature Case 2:\n{}'.format(T2_fdm))

    print('-' * 58)
    print('-' * 18, 'Convergence of T(1/2)', '-' * 18)
    print('-' * 58)
    print('dxs:\n{}'.format(dxs))
    print('T(1/2) Case 1 exact: {}'.format(T1_exact[int(len(T1_exact)/2)]))
    print('T(1/2) Case 1 FDM:\n{}'.format(T1_fdm_half))
    print('Percent Error Case 1 wrt exact:\n{}'.format(perr_T1_exact))
    print('Percent Error Case 1 wrt extrap:\n{}'.format(perr_T1))
    print('Rate of Convergence wrt extrap case 1:\n{}'.format(roc_T1))
    print('Rate of Convergence wrt exact case 1:\n{}'.format(roc_T1_exact))
    print('T(1/2) Case 2 exact: {}'.format(T2_exact[int(len(T2_exact)/2)]))
    print('T(1/2) Case 2 FDM:\n{}'.format(T02_fdm))
    print('Percent Error Case 2 wrt exact:\n{}'.format(perr_T2_exact))
    print('Percent Error Case 2 wrt extrap:\n{}'.format(perr_T2))
    print('Rate of Convergence wrt extrap case 2:\n{}'.format(roc_T2))
    print('Rate of Convergence wrt exact case 2:\n{}'.format(roc_T2_exact))

    print('-' * 58)
    print('-' * 18, 'Convergence of q', '-' * 18)
    print('-' * 58)
    print('-' * 18, 'FOURTH ORDER', '-' * 18)
    print('dxs:\n{}'.format(dxs))
    print('q Case 1:\n{}'.format(q1_4))
    print('q Case 2:\n{}'.format(q2_4))
    print('Percent Error q1 fdm wrt extrap:\n{}'.format(perr_q1_4))
    print('Percent Error q1 fdm wrt exact:\n{}'.format(perr_q1_exact_4))
    print('Rate of Convergence wrt extrap case 1:\n{}'.format(roc_q1_4))
    print('Rate of Convergence wrt exact case 1:\n{}'.format(roc_q1_exact_4))
    print('Percent Error q2 fdm wrt extrap:\n{}'.format(perr_q2_4))
    print('Percent Error q2 fdm wrt exact:\n{}'.format(perr_q2_exact_4))
    print('Rate of Convergence wrt extrap case 2:\n{}'.format(roc_q2_4))
    print('Rate of Convergence wrt exact case 2:\n{}'.format(roc_q2_exact_4))
    print('-' * 18, 'SECOND ORDER', '-' * 18)
    print('dxs:\n{}'.format(dxs))
    print('q Case 1:\n{}'.format(q1_2))
    print('q Case 2:\n{}'.format(q2_2))
    print('Percent Error q1 fdm wrt extrap:\n{}'.format(perr_q1_2))
    print('Percent Error q1 fdm wrt exact:\n{}'.format(perr_q1_exact_2))
    print('Rate of Convergence wrt extrap case 1:\n{}'.format(roc_q1_2))
    print('Rate of Convergence wrt exact case 1:\n{}'.format(roc_q1_exact_2))
    print('Percent Error q2 fdm wrt extrap:\n{}'.format(perr_q2_2))
    print('Percent Error q2 fdm wrt exact:\n{}'.format(perr_q2_exact_2))
    print('Rate of Convergence wrt extrap case 2:\n{}'.format(roc_q2_2))
    print('Rate of Convergence wrt exact case 2:\n{}'.format(roc_q2_exact_2))
    print('-' * 18, 'FIRST ORDER', '-' * 18)
    print('dxs:\n{}'.format(dxs))
    print('q Case 1:\n{}'.format(q1_1))
    print('q Case 2:\n{}'.format(q2_1))
    print('Percent Error q1 fdm wrt extrap:\n{}'.format(perr_q1_1))
    print('Percent Error q1 fdm wrt exact:\n{}'.format(perr_q1_exact_1))
    print('Rate of Convergence wrt extrap case 1:\n{}'.format(roc_q1_1))
    print('Rate of Convergence wrt exact case 1:\n{}'.format(roc_q1_exact_1))
    print('Percent Error q2 fdm wrt extrap:\n{}'.format(perr_q2_1))
    print('Percent Error q2 fdm wrt exact:\n{}'.format(perr_q2_exact_1))
    print('Rate of Convergence wrt extrap case 2:\n{}'.format(roc_q2_1))
    print('Rate of Convergence wrt exact case 2:\n{}'.format(roc_q2_exact_1))


