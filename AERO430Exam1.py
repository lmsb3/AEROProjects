import numpy as np
import matplotlib.pyplot as plt


def exact_sol_c1(n, Tm, k0, k1):
    x = np.linspace(0, 1, n+1)
    # Alpha Definition
    a0 = (h*P/(k0*A))**.5
    a1 = (h*P/(k1*A))**.5
    T = np.zeros(n+1)
    split = x > np.pi/5
    not_split = np.invert(split)
    T[split] = Tm*np.sinh(a1*(1-x[split]))/np.sinh((1-np.pi/5)*a1)+T_L*np.sinh(a1*(x[split]-np.pi/5))/np.sinh((1-np.pi/5)*a1)
    T[not_split] = Tm*np.sinh(a0*x[not_split])/np.sinh(a0*np.pi/5)
    return x, T


def fdm_sol_c1(n, k0, k1):
    b = np.zeros(n+1)
    b[0], b[-1] = T_0, T_L
    x = np.linspace(0, 1, n+1)
    diag = np.ones((n+1))
    diag[x <= x_split] *= 2+h*P/k0/A*(1/n)**2
    diag[x > x_split] *= 2+h*P/k1/A*(1/n)**2
    fd_mat = np.eye(n+1) * diag + np.roll(np.eye(n+2) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(n+2) * -1, -(n+2))[0:-1, 0:-1]
    split_index = len(diag[x <= x_split])-1
    fd_mat[split_index, split_index-1:split_index+2] = [-k0*A*n, A*n*(k0+k1) + h*P/n, -k1*A*n]
    fd_mat[0, 0], fd_mat[-1, -1] = 1, 1
    fd_mat[0, 1], fd_mat[-1, -2] = 0, 0
    temp = np.linalg.solve(fd_mat, b)
    return x, temp


def fem_sol_c1(n, k0, k1):
    b = np.zeros(n + 1)
    b[0], b[-1] = T_0, T_L
    x = np.linspace(0, 1, n + 1)
    diag = np.ones((n + 1))
    w0_sq = h*P/A/k0
    w1_sq = h*P/A/k1
    diag[x <= x_split] *= (-2*n-w0_sq*2/3/n)/(-n+w0_sq/n/6)
    diag[x > x_split] *= (-2*n-w1_sq*2/3/n)/(-n+w1_sq/n/6)
    fe_mat = np.eye(n + 1) * diag + np.roll(np.eye(n + 2) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(n + 2) * -1, -(n + 2))[0:-1, 0:-1]
    split_index = len(diag[x <= x_split])-1
    fe_mat[split_index, split_index - 1:split_index + 2] = [k0*A*n-h/6*P/n, -A*n*(k0+k1) - h*P/n*2/3, k1*A*n-h/6*P/n]
    fe_mat[0, 0], fe_mat[-1, -1] = 1, 1
    fe_mat[0, 1], fe_mat[-1, -2] = 0, 0
    temp = np.linalg.solve(fe_mat, b)
    return x, temp


def fdm_sol_c2(n, k0, k1):
    b = np.zeros(n+1)
    b[0], b[-1] = T_0, T_L
    x = np.linspace(0, 1, n+1)
    diag = np.ones((n+1))
    diag[x <= x_split] *= 2+h*P/k0/A*(1/n)**2
    diag[x > x_split] *= 2+h*P/k1/A*(1/n)**2
    fd_mat = np.eye(n+1) * diag + np.roll(np.eye(n+2) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(n+2) * -1, -(n+2))[0:-1, 0:-1]
    split_index = len(diag[x <= x_split])-1
    fd_mat[split_index, split_index-1:split_index+2] = [-k0*A*n, A*n*(k0+k1) + h*P/n, -k1*A*n]
    fd_mat[-1, -1] = 1
    fd_mat[-1, -2] = 0
    fd_mat[0, 0] = (2+h*P/k0/A*(1/n)**2)/2
    temp = np.linalg.solve(fd_mat, b)
    return x, temp


def fem_sol_c2(n, k0, k1):
    b = np.zeros(n + 1)
    b[0], b[-1] = T_0, T_L
    x = np.linspace(0, 1, n + 1)
    diag = np.ones((n + 1))
    w0_sq = h*P/A/k0
    w1_sq = h*P/A/k1
    diag[x <= x_split] *= (-2*n-w0_sq*2/3/n)/(-n+w0_sq/n/6)
    diag[x > x_split] *= (-2*n-w1_sq*2/3/n)/(-n+w1_sq/n/6)
    fe_mat = np.eye(n + 1) * diag + np.roll(np.eye(n + 2) * -1, -1)[0:-1, 0:-1] + \
             np.roll(np.eye(n + 2) * -1, -(n + 2))[0:-1, 0:-1]
    split_index = len(diag[x <= x_split])-1
    fe_mat[split_index, split_index - 1:split_index + 2] = [k0*A*n-h/6*P/n, -A*n*(k0+k1) - h*P/n*2/3, k1*A*n-h/6*P/n]
    fe_mat[-1, -1] = 1
    fe_mat[-1, -2] = 0
    fe_mat[0, 0] = (-2*n-w0_sq*2/3/n)/(-n+w0_sq/n/6)/2
    temp = np.linalg.solve(fe_mat, b)
    return x, temp


def calc_q_dot(n, temp, k):
    w = h*P/A/k
    return -k*A*((temp[-1]-temp[-2])*n + w*temp[-1]/2/n)


def find_extrap_values(values):
    values = np.asanyarray(values)
    q_mesh = values[:-2]
    q_half_mesh = values[1:-1]
    q_quarter_mesh = values[2:]
    return (q_half_mesh**2 - q_mesh*q_quarter_mesh)/(2*q_half_mesh-q_quarter_mesh-q_mesh)


def roc(values, exact, extrap=False):
    if not extrap:
        values = np.asanyarray(values)
        return np.log2(np.abs((exact-values[:-1])/(exact-values[1:]))), np.abs(exact-values)/np.abs(exact)
    else:
        roc = np.log2((exact - values[:-2]) / (exact - values[1:-1]))
        return roc, np.abs((exact - values[2:]) / exact)


def print_values(title, values):
    print('{}:\n{}'.format(title, values))


if __name__ == '__main__':
    print('Output:')
    appendix_table_string = ''
    # Bar Variables
    T_0 = 0
    T_L = 100
    h = 0.4
    r = 0.1
    L = 1
    A = np.pi*r**2
    P = 2*np.pi*r*L + 2*A
    x_split = round(np.pi/5, 6)
    # x_split = 1/2

    cs = np.array([1e-3, 1e-1, 1, 1e1, 1e3])
    k_0 = 0.5
    k1s = k_0*cs
    ns = 2**np.array(range(2, 10))
    dxs = 1/ns

    print_values('Ns', ns)
    # ------------------------------Case 1------------------------------
    print('-'*20, 'Case 1', '-'*20, sep='')
    # ---------------FDM---------------
    print('-'*20, 'FDM', '-'*20, sep='')
    temp_interface_fdm = []
    q_dots = []
    plt.figure()
    plt.grid()

    # Varying n
    print('----X and Temp values sampled at intervals for varied ns----\nk0 = 0.5; k1 = 5')
    for n in ns:
        x, temp = fdm_sol_c1(n, k_0, k_0*10)
        print('============\nN: ', n, '\nX:\n', x[::int(np.ceil(n/10))], '\nTemp:\n', temp[::int(np.ceil(n/10))], sep='')
        plt.plot(x, temp, label='n = {}'.format(n))
        temp_interface_fdm.append(temp[len(x[x <= x_split])-1])
        q_dots.append(calc_q_dot(n, temp, k_0*10))
    print('============')
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 1 FDM Varying N')

    # Exact
    x, temp = exact_sol_c1(100, temp_interface_fdm[-1], k_0, k_0 * 10)
    plt.plot(x, temp, label='Exact')
    plt.legend()

    # Rate of Convergence
    beta_extrap, rel_err_extrap = roc(q_dots[:-1], find_extrap_values(q_dots[:-1]), extrap=True)
    plt.figure()
    plt.grid()
    plt.loglog(1/ns[3:], rel_err_extrap)
    plt.title('Case 1 ROC Heat Loss FDM using Richardson Extrapolation')
    plt.xlabel('dx')
    plt.ylabel('Relative Error')

    # Varying k1
    plt.figure()
    print('----X and Temp values sampled at intervals for varied k1----')
    for k_1 in k1s:
        x, temp = fdm_sol_c1(ns[-1], k_0, k_1)
        print('============\nk1: ', k_1, '\nX:\n', x[::int(np.ceil(len(x)/10))], '\nTemp:\n', temp[::int(np.ceil(len(temp)/10))], sep='')
        plt.plot(x, temp, label='k0={}; k1={}'.format(k_0, k_1))
    print('============')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 1 FDM Varying k1 N={}'.format(ns[-1]))
    plt.grid()

    # Printing
    print('----Values of Interest----')
    print_values('Interface Temperatures', temp_interface_fdm)
    print_values('Heat Loss values wrt to n', q_dots)
    print_values('ROC for Heat Loss wrt extrap value', beta_extrap)
    print_values('Relative error wrt extrapolated value', rel_err_extrap)
    print('K0:', k_0)
    print_values('K1s', k1s)
    print('X and Temp values sampled at intervals:', )

    # ---------------FEM---------------
    print('-' * 20, 'FEM', '-' * 20, sep='')
    temp_interface_fem = []
    q_dots = []
    plt.figure()
    plt.grid()

    # Varying n
    print('----X and Temp values sampled at intervals for varied ns----\nk0 = 0.5; k1 = 5')
    for n in ns:
        x, temp = fem_sol_c1(n, k_0, k_0 * 10)
        print('============\nN: ', n, '\nX:\n', x[::int(np.ceil(n / 10))], '\nTemp:\n', temp[::int(np.ceil(n / 10))],
              sep='')
        plt.plot(x, temp, label='n = {}'.format(n))
        temp_interface_fem.append(temp[len(x[x <= x_split]) - 1])
        q_dots.append(calc_q_dot(n, temp, k_0 * 10))
    print('============')
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 1 FEM Varying N')

    # Exact
    x, temp = exact_sol_c1(100, temp_interface_fem[-1], k_0, k_0 * 10)
    plt.plot(x, temp, label='Exact')
    plt.legend()

    # Rate of Convergence
    beta_extrap, rel_err_extrap = roc(q_dots[:-1], find_extrap_values(q_dots[:-1]), extrap=True)
    plt.figure()
    plt.grid()
    plt.loglog(1 / ns[3:], rel_err_extrap)
    plt.title('Case 1 ROC Heat Loss FEM using Richardson Extrapolation')
    plt.xlabel('dx')
    plt.ylabel('Relative Error')

    # Varying k1
    plt.figure()
    print('----X and Temp values sampled at intervals for varied k1----')
    for k_1 in k1s:
        x, temp = fem_sol_c1(ns[-1], k_0, k_1)
        print('============\nk1: ', k_1, '\nX:\n', x[::int(np.ceil(len(x) / 10))], '\nTemp:\n',
              temp[::int(np.ceil(len(temp) / 10))], sep='')
        plt.plot(x, temp, label='k0={}; k1={}'.format(k_0, k_1))
    print('============')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 1 FEM Varying k1 N={}'.format(ns[-1]))
    plt.grid()

    # Printing
    print('----Values of Interest----')
    print_values('Interface Temperatures', temp_interface_fdm)
    print_values('Heat Loss values wrt to n', q_dots)
    print_values('ROC for Heat Loss wrt extrap value', beta_extrap)
    print_values('Relative error wrt extrapolated value', rel_err_extrap)
    print('K0:', k_0)
    print_values('K1s', k1s)
    print('X and Temp values sampled at intervals:', )

    # ------------------------------Case 2------------------------------
    print('-' * 20, 'Case 2', '-' * 20, sep='')
    # ---------------FDM---------------
    print('-' * 20, 'FDM', '-' * 20, sep='')
    temp_interface_fdm = []
    q_dots = []
    plt.figure()
    plt.grid()

    # Varying n
    print('----X and Temp values sampled at intervals for varied ns----\nk0 = 0.5; k1 = 5')
    for n in ns:
        x, temp = fdm_sol_c2(n, k_0, k_0 * 10)
        print('============\nN: ', n, '\nX:\n', x[::int(np.ceil(n / 10))], '\nTemp:\n', temp[::int(np.ceil(n / 10))],
              sep='')
        plt.plot(x, temp, label='n = {}'.format(n))
        temp_interface_fdm.append(temp[len(x[x <= x_split]) - 1])
        q_dots.append(calc_q_dot(n, temp, k_0 * 10))
    print('============')
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 2 FDM Varying N')

    # Rate of Convergence
    beta_extrap, rel_err_extrap = roc(q_dots[:-1], find_extrap_values(q_dots[:-1]), extrap=True)
    plt.figure()
    plt.grid()
    plt.loglog(1 / ns[3:], rel_err_extrap)
    plt.title('Case 2 ROC Heat Loss FDM using Richardson Extrapolation')
    plt.xlabel('dx')
    plt.ylabel('Relative Error')

    # Varying k1
    plt.figure()
    print('----X and Temp values sampled at intervals for varied k1----')
    for k_1 in k1s:
        x, temp = fdm_sol_c2(ns[-1], k_0, k_1)
        print('============\nk1: ', k_1, '\nX:\n', x[::int(np.ceil(len(x) / 10))], '\nTemp:\n',
              temp[::int(np.ceil(len(temp) / 10))], sep='')
        plt.plot(x, temp, label='k0={}; k1={}'.format(k_0, k_1))
    print('============')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 2 FDM Varying k1 N={}'.format(ns[-1]))
    plt.grid()

    # Printing
    print('----Values of Interest----')
    print_values('Interface Temperatures', temp_interface_fdm)
    print_values('Heat Loss values wrt to n', q_dots)
    print_values('ROC for Heat Loss wrt extrap value', beta_extrap)
    print_values('Relative error wrt extrapolated value', rel_err_extrap)
    print('K0:', k_0)
    print_values('K1s', k1s)
    print('X and Temp values sampled at intervals:', )

    # ---------------FEM---------------
    print('-' * 20, 'FEM', '-' * 20, sep='')
    temp_interface_fem = []
    q_dots = []
    plt.figure()
    plt.grid()

    # Varying n
    print('----X and Temp values sampled at intervals for varied ns----\nk0 = 0.5; k1 = 5')
    for n in ns:
        x, temp = fem_sol_c2(n, k_0, k_0 * 10)
        print('============\nN: ', n, '\nX:\n', x[::int(np.ceil(n / 10))], '\nTemp:\n', temp[::int(np.ceil(n / 10))],
              sep='')
        plt.plot(x, temp, label='n = {}'.format(n))
        temp_interface_fem.append(temp[len(x[x <= x_split]) - 1])
        q_dots.append(calc_q_dot(n, temp, k_0 * 10))
    print('============')
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 2 FEM Varying N')

    # Rate of Convergence
    beta_extrap, rel_err_extrap = roc(q_dots[:-1], find_extrap_values(q_dots[:-1]), extrap=True)
    plt.figure()
    plt.grid()
    plt.loglog(1 / ns[3:], rel_err_extrap)
    plt.title('Case 2 ROC Heat Loss FEM using Richardson Extrapolation')
    plt.xlabel('dx')
    plt.ylabel('Relative Error')

    # Varying k1
    plt.figure()
    print('----X and Temp values sampled at intervals for varied k1----')
    for k_1 in k1s:
        x, temp = fem_sol_c2(ns[-1], k_0, k_1)
        print('============\nk1: ', k_1, '\nX:\n', x[::int(np.ceil(len(x) / 10))], '\nTemp:\n',
              temp[::int(np.ceil(len(temp) / 10))], sep='')
        plt.plot(x, temp, label='k0={}; k1={}'.format(k_0, k_1))
    print('============')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Temperature')
    plt.title('Case 2 FEM Varying k1 N={}'.format(ns[-1]))
    plt.grid()

    # Printing
    print('----Values of Interest----')
    print_values('Interface Temperatures', temp_interface_fdm)
    print_values('Heat Loss values wrt to n', q_dots)
    print_values('ROC for Heat Loss wrt extrap value', beta_extrap)
    print_values('Relative error wrt extrapolated value', rel_err_extrap)
    print('K0:', k_0)
    print_values('K1s', k1s)

    plt.show()
