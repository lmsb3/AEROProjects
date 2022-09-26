# Blake Rogers
# 09/14/2021
# AERO 430

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# Functions for Problem 2

# Setting up left hand matrix for Case 1
def Case1(n):
    a = 4
    dx = 1 / (2 ** n)
    k = 2 + ((a ** 2) * (dx ** 2))

    fd_mat = np.zeros((2 ** n - 1, 2 ** n - 1))
    fd_mat[0][0:2] = [k, -1]
    row_FDM = [-1, k, -1]

    for i in range(1, 2 ** n - 2):
        fd_mat[i][i - 1:i + 2] = row_FDM

    fd_mat[2 ** n - 2][2 ** n - 3:2 ** n - 1] = [-1, k]

    return (fd_mat)


# Setting up left hand matrix for Case 2
def Case2(n):
    a = 4
    dx = 1 / (2 ** n)
    kap = 2 + ((a ** 2) * (dx ** 2))
    h = a ** 2 * k * R / 2
    k_prime = ((dx * h) / k) + (kap / 2)

    left_FDM_array = np.zeros((2 ** n, 2 ** n))
    left_FDM_array[0][0:2] = [k_prime, -1]
    row_FDM = [-1, kap, -1]

    for i in range(1, 2 ** n - 1):
        left_FDM_array[i][i - 1:i + 2] = row_FDM

    left_FDM_array[2 ** n - 1][2 ** n - 2:2 ** n] = [-1, kap]

    return (left_FDM_array)


# Setting up left hand matrix for Case 3
def Case3(n):
    a = 4
    dx = 1 / (2 ** n)
    kap = 2 + ((a ** 2) * (dx ** 2))
    k_prime = (kap / 2)

    left_FDM_array = np.zeros((2 ** n, 2 ** n))
    left_FDM_array[0][0:2] = [k_prime, -1]
    row_FDM = [-1, kap, -1]

    for i in range(1, 2 ** n - 1):
        left_FDM_array[i][i - 1:i + 2] = row_FDM

    left_FDM_array[2 ** n - 1][2 ** n - 2:2 ** n] = [-1, kap]

    return (left_FDM_array)


# Simpson rule function
def simpson(a, b, n, input_array):
    sum = 0
    for k in range(n + 1):
        summand = input_array[k]
        if (k != 0) and (k != n):
            summand *= (2 + (2 * (k % 2)))
        sum += summand
    # sum = np.sum([input_array[k]*(2 + (2 * (k % 2))) if (k != 0 and k != n) else input_array[k]] for k in range(n+1))
    return ((b - a) / (3 * n)) * sum


# Probelm 3 (Solved before to reference values in main loop)
# Bar Paramaters
k = 0.5  # Thermal conductivity
R = 0.1  # Cross section radius
Ac = np.pi * R ** 2  # Cross sectional area
L = 1  # Bar length
As = 2 * np.pi * R * L
a = 4
h = a ** 2 * k * R / 2

# Case paramaters
T_0 = 0  # T(0)
T_L = 100  # T(L)
Ta = 0  # Ambient temperature

# For loop to calculate analytical value for heat loss due to Newton's cooling
for case in [1, 2, 3]:
    if case == 1:
        C = (T_L - Ta - (T_0 - Ta) * np.cosh(a * L)) / np.sinh(a * L)
        D = T_0 - Ta
        qdotL1_exact = As * h * (((np.cosh(a * L) - 1) * (C / a)) + ((np.sinh(a * L)) * (D / a)) + (Ta * L))
    elif case == 2:
        C = (h / (k * a)) * (T_L / (((h / (k * a)) * np.sinh(a * L)) + np.cosh(a * L)))
        D = (T_L / (((h / (k * a)) * np.sinh(a * L)) + np.cosh(a * L)))
        qdotL2_exact = As * h * (((np.cosh(a * L) - 1) * (C / a)) + ((np.sinh(a * L)) * (D / a)) + (Ta * L))
    elif case == 3:
        C = 0
        D = T_L / np.cosh(a)
        qdotL3_exact = As * h * (((np.cosh(a * L) - 1) * (C / a)) + ((np.sinh(a * L)) * (D / a)) + (Ta * L))

# Problem 1

# Setting up storage arrays and x
x = np.linspace(0, L)
x_table = np.linspace(0, L, 11)
temp_table1 = []
temp_table2 = []
temp_table3 = []
qdot1 = []
qdot2 = []
qdot3 = []
alpha = np.asarray([0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10])
i = 0
alph4_T_table1 = []
alph4_T_table2 = []
alph4_T_table3 = []
for a in alpha:
    i += 1
    h = (a ** 2 * k * R) / 2
    for case in [1, 2, 3]:
        if case == 1:
            # Case 1 analytical solving and plotting
            C = (T_L - Ta - (T_0 - Ta) * np.cosh(a * L)) / np.sinh(a * L)
            D = T_0 - Ta
            T = C * np.sinh(a * x) + D * np.cosh(a * x) + Ta
            T_table1 = C * np.sinh(a * x_table) + D * np.cosh(a * x_table) + Ta
            if a == 4:
                alph4_T_table1 = T_table1
                qL1 = 2 * np.pi * 0.1 * 0.4 * (((C / a) * (np.cosh(a * L) - 1)) + ((D / a) * np.sinh(a * L)) + Ta * L)
            temp_table1.append(T_table1)
            qdot_case1 = -k * a * Ac * (C * np.cosh(a * L) + D * np.sinh(a * L))
            qdot1.append(qdot_case1)
            plt.figure(1)
            plt.plot(x, T, label='' + str(a) + ' \u03B1')
            plt.xlabel('Position (x)')
            plt.ylabel('Temperature (C)')
            plt.title('Case 1 Temperature vs. Position')
            plt.legend()
            plt.grid(True)
        elif case == 2:
            # Case 2 analytical solving and plotting
            C = (h / (k * a)) * (T_L / (((h / (k * a)) * np.sinh(a * L)) + np.cosh(a * L)))
            D = (T_L / (((h / (k * a)) * np.sinh(a * L)) + np.cosh(a * L)))
            T = C * np.sinh(a * x) + D * np.cosh(a * x) + Ta
            if a == 4:
                t02 = C * np.sinh(a * 0) + D * np.cosh(a * 0) + Ta
            T_table2 = C * np.sinh(a * x_table) + D * np.cosh(a * x_table) + Ta
            if a == 4:
                alph4_T_table2 = T_table2
                qL2 = 2 * np.pi * 0.1 * 0.4 * (((C / a) * (np.cosh(a * L) - 1)) + ((D / a) * np.sinh(a * L)) + Ta * L)
            temp_table2.append(T_table2)
            qdot_case2 = -k * a * Ac * (C * np.cosh(a * L) + D * np.sinh(a * L))
            qdot2.append(qdot_case2)
            plt.figure(2)
            plt.plot(x, T, label='' + str(a) + ' \u03B1')
            plt.xlabel('Position (x)')
            plt.ylabel('Temperature (C)')
            plt.title('Case 2 Temperature vs. Position')
            plt.legend()
            plt.grid(True)
        elif case == 3:
            # Case 3 analytical solving and plotting
            C = 0
            D = T_L / np.cosh(a)
            T = C * np.sinh(a * x) + D * np.cosh(a * x) + Ta
            T_table3 = C * np.sinh(a * x_table) + D * np.cosh(a * x_table) + Ta
            if a == 4:
                alph4_T_table3 = T_table3
                qL3 = 2 * np.pi * 0.1 * 0.4 * (((C / a) * (np.cosh(a * L) - 1)) + ((D / a) * np.sinh(a * L)) + Ta * L)
            temp_table3.append(T_table3)
            qdot_case3 = -k * a * Ac * (C * np.cosh(a * L) + D * np.sinh(a * L))
            qdot3.append(qdot_case3)
            plt.figure(3)
            plt.plot(x, T, label='' + str(a) + ' \u03B1')
            plt.xlabel('Position (x)')
            plt.ylabel('Temperature (C)')
            plt.title('Case 3 Temperature vs. Position')
            plt.legend()
            plt.grid(True)
        T = C * np.sinh(a * x) + D * np.cosh(a * x) + Ta
        plt.figure(i + 3)
        plt.plot(x, T, label="Case %i" % case)
        plt.xlabel("Position (x)")
        plt.ylabel("Temperature (T)")
        plt.title("Temperature vs. Position [alpha={:.2f}]".format(a))
        plt.legend()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.grid(True)

# Post Processing - Printing tables
# Temperature
print('------------- Case 1 Temperature for Varying Alpha -------------')
d = {'Alpha=0.25': temp_table1[0], 'Alpha=0.5': temp_table1[1], 'Alpha=0.75': temp_table1[2], 'Alpha=1': temp_table1[3],
     'Alpha=2': temp_table1[4]}
df = pd.DataFrame(data=d, index=x_table)
print(df)

d = {'Alpha=4': temp_table1[5], 'Alpha=6': temp_table1[6], 'Alpha=8': temp_table1[7], 'Alpha=10': temp_table1[8]}
df = pd.DataFrame(data=d, index=x_table)
print(df)

print('------------- Case 2 Temperature for Varying Alpha -------------')
d = {'Alpha=0.25': temp_table2[0], 'Alpha=0.5': temp_table2[1], 'Alpha=0.75': temp_table2[2], 'Alpha=1': temp_table2[3],
     'Alpha=2': temp_table2[4]}
df = pd.DataFrame(data=d, index=x_table)
print(df)

d = {'Alpha=4': temp_table2[5], 'Alpha=6': temp_table2[6], 'Alpha=8': temp_table2[7], 'Alpha=10': temp_table2[8]}
df = pd.DataFrame(data=d, index=x_table)
print(df)

print('------------- Case 3 Temperature for Varying Alpha -------------')
d = {'Alpha=0.25': temp_table3[0], 'Alpha=0.5': temp_table3[1], 'Alpha=0.75': temp_table3[2], 'Alpha=1': temp_table3[3],
     'Alpha=2': temp_table3[4]}
df = pd.DataFrame(data=d, index=x_table)
print(df)

d = {'Alpha=4': temp_table3[5], 'Alpha=6': temp_table3[6], 'Alpha=8': temp_table3[7], 'Alpha=10': temp_table3[8]}
df = pd.DataFrame(data=d, index=x_table)
print(df)

# Heat Loss
print('-------- Heat Loss vs. Alpha Table Case 1--------')
table_a_qdot1 = pd.DataFrame(qdot1, columns=['Heat Loss (qdot)'], index=alpha)
index = table_a_qdot1.index
index.name = 'Alpha'
print(table_a_qdot1)

print('\n-------- Heat Loss vs. Alpha Table Case 2--------')
table_a_qdot2 = pd.DataFrame(qdot2, columns=['Heat Loss (qdot)'], index=alpha)
index = table_a_qdot2.index
index.name = 'Alpha'
print(table_a_qdot2)

print('\n-------- Heat Loss vs. Alpha Table Case 3--------')
table_a_qdot3 = pd.DataFrame(qdot3, columns=['Heat Loss (qdot)'], index=alpha)
index = table_a_qdot3.index
index.name = 'Alpha'
print(table_a_qdot3)

# Problem 2/3 (Used same loop)

# Array intialization - a lot but most used to reference for report, graphs, or set up data to be used later
q_fdm_dot1 = []
q_fdm_dot2 = []
q_fdm_dot3 = []
a = 4
Q1_approx = []
Q2_approx = []
Q3_approx = []
beta1_t0_store = []
beta2_t0_store = []
beta3_t0_store = []
beta1_qd_store = []
beta2_qd_store = []
beta3_qd_store = []


# Richardson extrapolation for temperature (Reference from array a certain way)
def richardsons_temp(Q):
    i = len(Q) - 1
    Qe = (Q[i - 1][0] ** 2 - Q[i - 2][0] * Q[i][0]) / (2 * Q[i - 1][0] - Q[i - 2][0] - Q[i][0])
    beta = (np.log((Qe - Q[i - 2])[0] / (Qe - Q[i - 1][0]))) / np.log(2)
    return (Qe, beta)


# Richardson extrapolation for qdot (Reference from array a certain way)
def richardsons_qdot(Q):
    i = len(Q) - 1
    Qe = (Q[i - 1] ** 2 - Q[i - 2] * Q[i]) / (2 * Q[i - 1] - Q[i - 2] - Q[i])
    beta = (np.log((Qe - Q[i - 2]) / (Qe - Q[i - 1]))) / np.log(2)
    return (Qe, beta)


# Array intialization - a lot but most used to reference for report, graphs, or set up data to be used later
deltx_store = []
sol_vary_mesh_store1 = []
sol_vary_mesh_store2 = []
sol_vary_mesh_store3 = []
qdot_vary_mesh_store1 = []
qdot_vary_mesh_store2 = []
qdot_vary_mesh_store3 = []
percent_error_qd_store1 = []
percent_error_qd_store2 = []
percent_error_qd_store3 = []
percent_error_t0_store2 = []
percent_error_t0_store3 = []
Q1qd_etr = []
Q2qd_etr = []
Q3qd_etr = []
Q2t0_etr = []
Q3t0_etr = []
percent_error_qd_store1_ex = []
percent_error_qd_store2_ex = []
percent_error_qd_store3_ex = []
percent_error_t0_store2_ex = []
percent_error_t0_store3_ex = []
t0_store2 = []
t0_store3 = []
flux_array1 = []
flux_array2 = []
flux_array3 = []
storeQe1 = []
storeQe2 = []
storeQe3 = []
storeb1 = []
storeb2 = []
storeb3 = []
error_per_store1 = []
error_per_store2 = []
error_per_store3 = []
ext_error_per_store1 = []
ext_error_per_store2 = []
ext_error_per_store3 = []
storebeta1_exact_q = []
storebeta2_exact_q = []
storebeta3_exact_q = []
storebeta2_exact_T = []
storebeta3_exact_T = []
simp_storebeta1_exact_q = []
simp_storebeta2_exact_q = []
simp_storebeta3_exact_q = []
for n in range(2, 12):
    x_fdm = np.linspace(0, 1, num=2 ** n)
    for case in [1, 2, 3]:
        if case == 1:
            # Set up of x and right matrix
            deltx = 1 / (2 ** n)
            deltx_store.append(deltx)
            resultant_matrix = np.zeros((2 ** n - 1, 1))
            resultant_matrix[2 ** n - 2, 0] = 100

            # Solve for left matrix using function
            left_matrix = Case1(n)
            left_matrix_inverse = np.linalg.inv(Case1(n))

            # Solve for solution and add boundary conditions in
            solution_matrix = np.dot(left_matrix_inverse, resultant_matrix)
            boundary_conditions_list = [0]

            for d in np.nditer(solution_matrix):
                boundary_conditions_list.append(d.tolist())

            boundary_conditions_list.append(100)

            # Problem 3 - solving for heat loss due to Newtons method numerically
            perimeter = 2 * np.pi * 0.1
            test_array = boundary_conditions_list
            lateral_flux_val = 0.4 * perimeter * simpson(0, 1, len(test_array) - 1, test_array)
            flux_array1.append(lateral_flux_val)

            # Solving for qdot numerically
            T_n = boundary_conditions_list[len(boundary_conditions_list) - 1]
            T_n_min1 = boundary_conditions_list[len(boundary_conditions_list) - 2]
            q_dot_env_approx = (-k * Ac * (-T_n_min1 / deltx + T_n / deltx + a ** 2 * T_n * deltx ** 2 / (2 * deltx)))
            q_fdm_dot1.append(q_dot_env_approx)

            # Richardson Extrapolation (finiding percent error, beta, and extrapolated values for report/graphs)
            if n == 2:
                percent_error_qd1 = np.abs(((qdot1[5] - q_dot_env_approx) / qdot1[5]) * 100)
                percent_error_qd_store1.append(percent_error_qd1)

                er_exact = np.abs(((qdotL1_exact - lateral_flux_val) / qdotL1_exact) * 100)
                error_per_store1.append(er_exact)

                beta1_exact_q = np.abs(
                    np.log(np.abs(q_fdm_dot1[n - 2] - qdot1[5]) / (np.abs(q_fdm_dot1[n - 3] - qdot1[5])))) / np.log(2)
                storebeta1_exact_q.append(beta1_exact_q)

                sbeta1_exact_q = np.abs(
                    np.log(np.abs(flux_array1[n - 2] - qL1) / (np.abs(flux_array1[n - 3] - qL1)))) / np.log(2)
                simp_storebeta1_exact_q.append(sbeta1_exact_q)
            if n >= 3:
                Qe1_qd, b1_qd = richardsons_qdot(q_fdm_dot1)
                Q1qd_etr.append(Qe1_qd)
                beta1_qd_store.append(b1_qd)

                Qe1, b1 = richardsons_qdot(flux_array1)
                storeQe1.append(Qe1)
                storeb1.append(b1)

                beta1_exact_q = np.abs(
                    np.log(np.abs(q_fdm_dot1[n - 2] - qdot1[5]) / (np.abs(q_fdm_dot1[n - 3] - qdot1[5])))) / np.log(2)
                storebeta1_exact_q.append(beta1_exact_q)

                sbeta1_exact_q = np.abs(
                    np.log(np.abs(flux_array1[n - 2] - qL1) / (np.abs(flux_array1[n - 3] - qL1)))) / np.log(2)
                simp_storebeta1_exact_q.append(sbeta1_exact_q)

                er_exact = np.abs(((qdotL1_exact - lateral_flux_val) / qdotL1_exact) * 100)
                error_per_store1.append(er_exact)

                er_exact = np.abs(((Qe1 - lateral_flux_val) / Qe1) * 100)
                ext_error_per_store1.append(er_exact)

                percent_error_qd1_ex = np.abs(((Qe1_qd - q_dot_env_approx) / Qe1_qd) * 100)
                percent_error_qd_store1_ex.append(percent_error_qd1_ex)

                percent_error_qd1 = np.abs(((qdot1[5] - q_dot_env_approx) / qdot1[5]) * 100)
                percent_error_qd_store1.append(percent_error_qd1)

        elif case == 2:
            # Set up of x and right matrix
            resultant_matrix = np.zeros((2 ** n, 1))
            resultant_matrix[2 ** n - 1, 0] = 100

            # Solve for left matrix using function
            left_matrix_inverse = np.linalg.inv(Case2(n))

            # Solve for solution and add boundary conditions in
            solution_matrix = np.dot(left_matrix_inverse, resultant_matrix)
            solution_matrix = np.append(solution_matrix, [100])

            # Solve for solution and add boundary conditions in
            sol_vary_mesh_store2.append(solution_matrix)
            t0_2 = solution_matrix[0]
            t0_store2.append(t0_2)

            # Problem 3 - solving for heat loss due to Newtons method numerically
            perimeter = 2 * np.pi * 0.1
            test_array = solution_matrix
            lateral_flux_val = 0.4 * perimeter * simpson(0, 1, len(test_array) - 1, test_array)
            flux_array2.append(lateral_flux_val)

            # Solving for qdot numerically
            T_n2 = solution_matrix[len(solution_matrix) - 1]
            T_n_min12 = solution_matrix[len(solution_matrix) - 2]
            q_dot_env_approx2 = (
                        -k * Ac * (-T_n_min12 / deltx + T_n2 / deltx + a ** 2 * T_n2 * deltx ** 2 / (2 * deltx)))
            q_fdm_dot2.append(q_dot_env_approx2)

            # Richardson Extrapolation (finiding percent error, beta, and extrapolated values for report/graphs)
            if n == 2:
                percent_error_t0 = np.abs(((alph4_T_table2[0] - solution_matrix[0]) / alph4_T_table2[0]) * 100)
                percent_error_t0_store2.append(percent_error_t0)

                percent_error_qd2 = np.abs(((qdot2[5] - q_dot_env_approx2) / qdot2[5]) * 100)
                percent_error_qd_store2.append(percent_error_qd2)

                er_exact = np.abs(((qdotL2_exact - lateral_flux_val) / qdotL2_exact) * 100)
                error_per_store2.append(er_exact)

                beta2_exact_q = np.abs(
                    np.log(np.abs(q_fdm_dot2[n - 2] - qdot2[5]) / (np.abs(q_fdm_dot2[n - 3] - qdot2[5])))) / np.log(2)
                storebeta2_exact_q.append(beta2_exact_q)

                beta2_exact_T = np.abs(np.log(np.abs(t0_store2[n - 2] - alph4_T_table2[0]) / (
                    np.abs(t0_store2[n - 3] - alph4_T_table2[0])))) / np.log(2)
                storebeta2_exact_T.append(beta2_exact_T)

                sbeta2_exact_q = np.abs(
                    np.log(np.abs(flux_array2[n - 2] - qL2) / (np.abs(flux_array2[n - 3] - qL2)))) / np.log(2)
                simp_storebeta2_exact_q.append(sbeta2_exact_q)
            if n >= 3:
                Qe2_t0, b2_t0 = richardsons_temp(sol_vary_mesh_store2)
                Q2t0_etr.append(Qe2_t0)
                beta2_t0_store.append(b2_t0)

                Qe2_qd, b2_qd = richardsons_qdot(q_fdm_dot2)
                Q2qd_etr.append(Qe2_qd)
                beta2_qd_store.append(b2_qd)

                Qe2, b2 = richardsons_qdot(flux_array2)
                storeQe2.append(Qe2)
                storeb2.append(b2)

                beta2_exact_q = np.abs(
                    np.log(np.abs(q_fdm_dot2[n - 2] - qdot2[5]) / (np.abs(q_fdm_dot2[n - 3] - qdot2[5])))) / np.log(2)
                storebeta2_exact_q.append(beta2_exact_q)

                beta2_exact_T = np.abs(np.log(np.abs(t0_store2[n - 2] - alph4_T_table2[0]) / (
                    np.abs(t0_store2[n - 3] - alph4_T_table2[0])))) / np.log(2)
                storebeta2_exact_T.append(beta2_exact_T)

                sbeta2_exact_q = np.abs(
                    np.log(np.abs(flux_array2[n - 2] - qL2) / (np.abs(flux_array2[n - 3] - qL2)))) / np.log(2)
                simp_storebeta2_exact_q.append(sbeta2_exact_q)

                er_exact = np.abs(((qdotL2_exact - lateral_flux_val) / qdotL2_exact) * 100)
                error_per_store2.append(er_exact)

                er_exact = np.abs(((Qe2 - lateral_flux_val) / Qe2) * 100)
                ext_error_per_store2.append(er_exact)

                percent_error_t0_ex = np.abs(((Qe2_t0 - solution_matrix[0]) / Qe2_t0) * 100)
                percent_error_t0_store2_ex.append(percent_error_t0_ex)

                percent_error_t0 = np.abs(((alph4_T_table2[0] - solution_matrix[0]) / alph4_T_table2[0]) * 100)
                percent_error_t0_store2.append(percent_error_t0)

                percent_error_qd2_ex = np.abs(((Qe2_qd - q_dot_env_approx2) / Qe2_qd) * 100)
                percent_error_qd_store2_ex.append(percent_error_qd2_ex)

                percent_error_qd2 = np.abs(((qdot2[5] - q_dot_env_approx2) / qdot2[5]) * 100)
                percent_error_qd_store2.append(percent_error_qd2)

        elif case == 3:
            # Set up of x and right matrix
            resultant_matrix = np.zeros((2 ** n, 1))
            resultant_matrix[2 ** n - 1, 0] = 100

            # Solve for left matrix using function
            left_matrix_inverse = np.linalg.inv(Case3(n))

            # Solve for solution and add boundary conditions in
            solution_matrix = np.dot(left_matrix_inverse, resultant_matrix)
            solution_matrix = np.append(solution_matrix, [100])

            # Solve for solution and add boundary conditions in
            sol_vary_mesh_store3.append(solution_matrix)
            t0_3 = solution_matrix[0]
            t0_store3.append(t0_3)

            # Problem 3 - solving for heat loss due to Newtons method numerically
            perimeter = 2 * np.pi * 0.1
            test_array = solution_matrix
            lateral_flux_val = 0.4 * perimeter * simpson(0, 1, len(test_array) - 1, test_array)
            flux_array3.append(lateral_flux_val)

            # Solving for qdot numerically
            T_n3 = solution_matrix[len(solution_matrix) - 1]
            T_n_min13 = solution_matrix[len(solution_matrix) - 2]
            q_dot_env_approx3 = (
                        -k * Ac * ((-T_n_min13 / deltx + T_n3 / deltx) + 4 ** 2 * T_n3 * deltx ** 2 / (2 * deltx)))
            q_fdm_dot3.append(q_dot_env_approx3)

            # Richardson Extrapolation (finiding percent error, beta, and extrapolated values for report/graphs)
            if n == 2:
                q3_dx25_ex = np.abs((qdot3[5] - q_dot_env_approx) / qdot3[5]) * 100
                q3_dx25 = np.abs((qdot3[5] - q_dot_env_approx) / qdot3[5]) * 100

                T3_dx25_ed = np.abs((alph4_T_table3[0] - solution_matrix[0]) / alph4_T_table3[0]) * 100
                T3_dx25 = np.abs((T_table3[0] - solution_matrix[0]) / T_table3[0]) * 100

                er_exact = np.abs(((qdotL3_exact - lateral_flux_val) / qdotL3_exact) * 100)
                error_per_store3.append(er_exact)

                beta3_exact_q = np.abs(
                    np.log(np.abs(q_fdm_dot3[n - 2] - qdot3[5]) / (np.abs(q_fdm_dot3[n - 3] - qdot3[5])))) / np.log(2)
                storebeta3_exact_q.append(beta3_exact_q)

                beta3_exact_T = np.abs(np.log(np.abs(t0_store3[n - 2] - alph4_T_table3[0]) / (
                    np.abs(t0_store3[n - 3] - alph4_T_table3[0])))) / np.log(2)
                storebeta3_exact_T.append(beta3_exact_T)

                percent_error_t0 = np.abs((((100 / np.cosh(4)) - solution_matrix[0]) / (100 / np.cosh(4))) * 100)
                percent_error_t0_store3.append(percent_error_t0)

                percent_error_qd3 = np.abs(((qdot3[5] - q_dot_env_approx3) / qdot3[5]) * 100)
                percent_error_qd_store3.append(percent_error_qd3)

                sbeta3_exact_q = np.abs(
                    np.log(np.abs(flux_array3[n - 2] - qL3) / (np.abs(flux_array3[n - 3] - qL3)))) / np.log(2)
                simp_storebeta3_exact_q.append(sbeta3_exact_q)
            if n >= 3:
                Qe3_t0, b3 = richardsons_temp(sol_vary_mesh_store3)
                Q3t0_etr.append(Qe3_t0)
                beta3_t0_store.append(b3)

                Qe3_qd, b3_qd = richardsons_qdot(q_fdm_dot3)
                Q3qd_etr.append(Qe3_qd)
                beta3_qd_store.append(b3_qd)

                Qe3, b3 = richardsons_qdot(flux_array3)
                storeQe3.append(Qe3)
                storeb3.append(b3)

                beta3_exact_q = np.abs(
                    np.log(np.abs(q_fdm_dot3[n - 2] - qdot3[5]) / (np.abs(q_fdm_dot3[n - 3] - qdot3[5])))) / np.log(2)
                storebeta3_exact_q.append(beta3_exact_q)

                beta3_exact_T = np.abs(np.log(np.abs(t0_store3[n - 2] - alph4_T_table3[0]) / (
                    np.abs(t0_store3[n - 3] - alph4_T_table3[0])))) / np.log(2)
                storebeta3_exact_T.append(beta3_exact_T)

                sbeta3_exact_q = np.abs(
                    np.log(np.abs(flux_array3[n - 2] - qL3) / (np.abs(flux_array3[n - 3] - qL3)))) / np.log(2)
                simp_storebeta3_exact_q.append(sbeta3_exact_q)

                er_exact = np.abs(((qdotL3_exact - lateral_flux_val) / qdotL3_exact) * 100)
                error_per_store3.append(er_exact)

                extr_er_exact = np.abs(((Qe3 - lateral_flux_val) / Qe3) * 100)
                ext_error_per_store3.append(extr_er_exact)

                percent_error_t0_ex = np.abs(((Qe3_t0 - solution_matrix[0]) / Qe3_t0) * 100)
                percent_error_t0_store3_ex.append(percent_error_t0_ex)

                percent_error_t0 = np.abs((((100 / np.cosh(4)) - solution_matrix[0]) / (100 / np.cosh(4))) * 100)
                percent_error_t0_store3.append(percent_error_t0)

                percent_error_qd3_ex = np.abs(((Qe3_qd - q_dot_env_approx3) / Qe3_qd) * 100)
                percent_error_qd_store3_ex.append(percent_error_qd3_ex)

                percent_error_qd3 = np.abs(((qdot3[5] - q_dot_env_approx3) / qdot3[5]) * 100)
                percent_error_qd_store3.append(percent_error_qd3)

q1_exact = [qdot1[5], qdot1[5], qdot1[5], qdot1[5], qdot1[5], qdot1[5], qdot1[5], qdot1[5], qdot1[5], qdot1[5]]
q2_exact = [qdot2[5], qdot2[5], qdot2[5], qdot2[5], qdot2[5], qdot2[5], qdot2[5], qdot2[5], qdot2[5], qdot2[5]]
q3_exact = [qdot3[5], qdot3[5], qdot3[5], qdot3[5], qdot3[5], qdot3[5], qdot3[5], qdot3[5], qdot3[5], qdot3[5]]
T2_exact = [alph4_T_table2[0], alph4_T_table2[0], alph4_T_table2[0], alph4_T_table2[0], alph4_T_table2[0],
            alph4_T_table2[0], alph4_T_table2[0], alph4_T_table2[0], alph4_T_table2[0], alph4_T_table2[0]]
T3_exact = [alph4_T_table3[0], alph4_T_table3[0], alph4_T_table3[0], alph4_T_table3[0], alph4_T_table3[0],
            alph4_T_table3[0], alph4_T_table3[0], alph4_T_table3[0], alph4_T_table3[0], alph4_T_table3[0]]
# Post Processing
# Case 1
print('------------- Case 1 qdot_fdm against qdot_exact -------------')
d = {'qdot_fdm': q_fdm_dot1, 'qdot_exact': q1_exact, 'Percent Error': percent_error_qd_store1,
     'Beta': storebeta1_exact_q}
df = pd.DataFrame(data=d, index=deltx_store)
print(df)

print('------------- Case 1 qdot_fdm against qdot_extr -------------')
d = {'qdot_fdm': q_fdm_dot1[2:], 'qdot_extr': Q1qd_etr[1:], 'Percent Error': percent_error_qd_store1_ex[1:],
     'Beta': beta1_qd_store[1:]}
df = pd.DataFrame(data=d, index=deltx_store[2:])
print(df)

# Case 2 Tables
print('\n------------- Case 2 qdot_fdm against qdot_exact -------------')
d = {'qdot_fdm': q_fdm_dot2, 'qdot_exact': q2_exact, 'Percent Error': percent_error_qd_store2,
     'Beta': storebeta2_exact_q}
df = pd.DataFrame(data=d, index=deltx_store)
print(df)

print('------------- Case 2 qdot_fdm against qdot_extr -------------')
d = {'qdot_fdm': q_fdm_dot2[2:], 'qdot_extr': Q2qd_etr[1:], 'Percent Error': percent_error_qd_store2_ex[1:],
     'Beta': beta2_qd_store[1:]}
df = pd.DataFrame(data=d, index=deltx_store[2:])
print(df)

print('\n------------- Case 2 T_0_fdm against T_0_exact -------------')
d = {'T_0_fdm': t0_store2, 'T_0_exact': T2_exact, 'Percent Error': percent_error_t0_store2, 'Beta': storebeta2_exact_T}
df = pd.DataFrame(data=d, index=deltx_store)
print(df)

print('------------- Case 2 T_0_fdm against T_0_extr -------------')
d = {'T_0_fdm': t0_store2[2:], 'T_0_extr': Q2t0_etr[1:], 'Percent Error': percent_error_t0_store2_ex[1:],
     'Beta': beta2_t0_store[1:]}
df = pd.DataFrame(data=d, index=deltx_store[2:])
print(df)

# Case 3 Tables
print('\n------------- Case 3 qdot_fdm against qdot_exact -------------')
d = {'qdot_fdm': q_fdm_dot3, 'qdot_exact': q3_exact, 'Percent Error': percent_error_qd_store3,
     'Beta': storebeta3_exact_q}
df = pd.DataFrame(data=d, index=deltx_store)
print(df)

print('------------- Case 3 qdot_fdm against qdot_extr -------------')
d = {'qdot_fdm': q_fdm_dot3[2:], 'qdot_extr': Q3qd_etr[1:], 'Percent Error': percent_error_qd_store3_ex[1:],
     'Beta': beta3_qd_store[1:]}
df = pd.DataFrame(data=d, index=deltx_store[2:])
print(df)

print('\n------------- Case 3 T_0_fdm against T_0_exact -------------')
d = {'T_0_fdm': t0_store3, 'T_0_exact': T3_exact, 'Percent Error': percent_error_t0_store3, 'Beta': storebeta3_exact_T}
df = pd.DataFrame(data=d, index=deltx_store)
print(df)

print('------------- Case 3 T_0_fdm against T_0_extr -------------')
d = {'T_0_fdm': t0_store3[2:], 'T_0_extr': Q3t0_etr[1:], 'Percent Error': percent_error_t0_store3_ex[1:],
     'Beta': beta3_t0_store[1:]}
df = pd.DataFrame(data=d, index=deltx_store[2:])
print(df)

# Case 1 (dx = 1/4, a = 4)
# Solving for analytical and FDM solution at dx = 0.25 and alpha = 4 to compare methods (work done in previous for loop)
n = 2

resultant_matrix = np.zeros((2 ** n - 1, 1))
resultant_matrix[2 ** n - 2, 0] = 100

left_matrix_inverse = np.linalg.inv(Case1(n))

solution_matrix = np.dot(left_matrix_inverse, resultant_matrix)
boundary_conditions_list = [0]

for i in np.nditer(solution_matrix):
    boundary_conditions_list.append(i.tolist())

boundary_conditions_list.append(100)
x_ex = np.linspace(0, 1, num=2 ** n + 1)

x_table = np.linspace(0, L, 11)
a = 4
C = (T_L - Ta - (T_0 - Ta) * np.cosh(a * L)) / np.sinh(a * L)
D = T_0 - Ta
T = C * np.sinh(a * x_table) + D * np.cosh(a * x_table) + Ta

print('-------- Temperature vs. Position Case 1 [alpha={:.2f}]'.format(a) + ' dx = 1/4 --------')
table_a_temp1 = pd.DataFrame(boundary_conditions_list, columns=['Temperature (C)'], index=x_ex)
index = table_a_temp1.index
index.name = 'Position (cm)'
print(table_a_temp1)
print(type(table_a_temp1))

# Plotting
plt.figure(i + 4)
plt.plot(x_ex, boundary_conditions_list, '-o', label='' + str(a) + ' \u03B1')
plt.plot(x_table, T, '-o', label='' + str(a) + ' \u03B1')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (C)')
plt.title('Case 1 Temperature vs. Position FDM compared to Analytical')
plt.legend(['FDM', 'Analytical'])
plt.grid(True)
# plt.show()

plt.figure(i + 5)
plt.loglog(deltx_store, percent_error_qd_store1)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 1 Convergence of qdot')
plt.grid(True)
# plt.show()

plt.figure(i + 6)
plt.loglog(deltx_store[1:], percent_error_qd_store1_ex)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 1 Convergence of qdot against q_extrapolated')
plt.grid(True)
# plt.show()

# Case 2 (dx = 1/4, a = 4)
# Solving for analytical and FDM solution at dx = 0.25 and alpha = 4 to compare methods (work done in previous for loop)
resultant_matrix = np.zeros((2 ** n, 1))
resultant_matrix[2 ** n - 1, 0] = 100

left_matrix_inverse = np.linalg.inv(Case2(n))
case2 = Case2(n)

solution_matrix = np.dot(left_matrix_inverse, resultant_matrix)

solution_matrix2 = np.append(solution_matrix, [100])
x_ex = np.linspace(0, 1, num=2 ** n + 1)

x_table = np.linspace(0, L, 11)
a = 4
C = (h / (k * a)) * (T_L / (((h / (k * a)) * np.sinh(a * L)) + np.cosh(a * L)))
D = (T_L / (((h / (k * a)) * np.sinh(a * L)) + np.cosh(a * L)))
T = C * np.sinh(a * x_table) + D * np.cosh(a * x_table) + Ta

print('-------- Temperature vs. Position Case 2 [alpha={:.2f}]'.format(a) + ' dx = 1/4 --------')
table_a_temp1 = pd.DataFrame(solution_matrix2, columns=['Temperature (C)'], index=x_ex)
index = table_a_temp1.index
index.name = 'Position (cm)'
print(table_a_temp1)

# Plotting
plt.figure(i + 7)
plt.plot(x_ex, solution_matrix2, '-o', label='' + str(a) + ' \u03B1')
plt.plot(x_table, T, '-o', label='' + str(a) + ' \u03B1')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (C)')
plt.title('Case 2 Temperature vs. Position FDM compared to Analytical')
plt.legend(['FDM', 'Analytical'])
plt.grid(True)
# plt.show()

plt.figure(i + 8)
plt.loglog(deltx_store, percent_error_t0_store2)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 2 Convergence of T_0 against T_0_extrapolated')
plt.grid(True)
# plt.show()

plt.figure(i + 9)
plt.loglog(deltx_store[1:], percent_error_t0_store2_ex)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 2 Convergence of T_0 against T_0_extrapolated')
plt.grid(True)
# plt.show()

plt.figure(i + 10)
plt.loglog(deltx_store, percent_error_qd_store2)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 2 Convergence of qdot')
plt.grid(True)
# plt.show()

plt.figure(i + 11)
plt.loglog(deltx_store[1:], percent_error_qd_store2_ex)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 2 Convergence of qdot against q_extrapolated')
plt.grid(True)
# plt.show()

# Case 3 (dx = 1/4, a = 4)
# Solving for analytical and FDM solution at dx = 0.25 and alpha = 4 to compare methods (work done in previous for loop)
resultant_matrix = np.zeros((2 ** n, 1))
resultant_matrix[2 ** n - 1, 0] = 100

left_matrix_inverse = np.linalg.inv(Case3(n))

solution_matrix = np.dot(left_matrix_inverse, resultant_matrix)

solution_matrix3 = np.append(solution_matrix, [100])
x_ex = np.linspace(0, 1, num=2 ** n + 1)

x_table = np.linspace(0, L, 11)
a = 4
C = 0
D = T_L / np.cosh(a)
T = C * np.sinh(a * x_table) + D * np.cosh(a * x_table) + Ta

print('-------- Temperature vs. Position Case 3 [alpha={:.2f}]'.format(a) + ' dx = 1/4 --------')
table_a_temp1 = pd.DataFrame(solution_matrix3, columns=['Temperature (C)'], index=x_ex)
index = table_a_temp1.index
index.name = 'Position (cm)'
print(table_a_temp1)

# Plotting
plt.figure(i + 12)
plt.plot(x_ex, solution_matrix3, '-o', label='' + str(a) + ' \u03B1')
plt.plot(x_table, T, '-o', label='' + str(a) + ' \u03B1')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (C)')
plt.title('Case 3 Temperature vs. Position FDM compared to Analytical')
plt.legend(['FDM', 'Analytical'])
plt.grid(True)
# plt.show()

plt.figure(i + 13)
plt.loglog(deltx_store[1:], percent_error_t0_store3[1:])
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 3 Convergence of T_0')
plt.grid(True)
# plt.show()

plt.figure(i + 14)
plt.loglog(deltx_store[1:], percent_error_t0_store3_ex)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 3 Convergence of T_0 against T_0_extrapolated')
plt.grid(True)
# plt.show()

plt.figure(i + 15)
plt.loglog(deltx_store[1:], percent_error_qd_store3[1:])
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 3 Convergence of qdot')
plt.grid(True)
# plt.show()

plt.figure(i + 16)
plt.loglog(deltx_store[1:], percent_error_qd_store3_ex)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 3 Convergence of qdot against q_extrapolated')
plt.grid(True)
# plt.show()

# Problem 3 plots

# Case 1
plt.figure(i + 17)
plt.loglog(deltx_store, error_per_store1)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 1 Convergence of qdot Simpson')
plt.grid(True)
# plt.show()

plt.figure(i + 18)
plt.loglog(deltx_store[1:], ext_error_per_store1)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 1 Convergence of qdot against qdot_extrapolated Simpson')
plt.grid(True)
# plt.show()

# Case 2
plt.figure(i + 19)
plt.loglog(deltx_store, error_per_store2)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 2 Convergence of qdot Simpson')
plt.grid(True)
# plt.show()

plt.figure(i + 20)
plt.loglog(deltx_store[1:], ext_error_per_store2)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 2 Convergence of qdot against qdot_extrapolated Simpson')
plt.grid(True)
# plt.show()

# Case 3
plt.figure(i + 21)
plt.loglog(deltx_store, error_per_store3)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 3 Convergence of qdot Simpson')
plt.grid(True)
# plt.show()

plt.figure(i + 22)
plt.loglog(deltx_store[1:], ext_error_per_store3)
plt.xlabel('dx')
plt.ylabel('% Error')
plt.title('Case 3 Convergence of qdot against qdot_extrapolated Simpson')
plt.grid(True)
# plt.show()

q1_exact = [qL1, qL1, qL1, qL1, qL1, qL1, qL1, qL1, qL1, qL1]
q2_exact = [qL2, qL2, qL2, qL2, qL2, qL2, qL2, qL2, qL2, qL2]
q3_exact = [qL3, qL3, qL3, qL3, qL3, qL3, qL3, qL3, qL3, qL3]

# # Case 1
# print('------------- Case 1 Simpson qdot_fdm against qdot_exact Simpson-------------')
# d = {'qdot_fdm': flux_array1, 'qdot_extr': q1_exact, 'Percent Error': error_per_store1, 'Beta': simp_storebeta1_exact_q}
# df = pd.DataFrame(data=d, index=deltx_store)
# print(df)
#
# print('------------- Case 1 Simpson qdot_fdm against qdot_extr Simpson-------------')
# d = {'qdot_fdm': flux_array1[2:], 'qdot_extr': storeQe1[1:], 'Percent Error': ext_error_per_store1[1:],
#      'Beta': storeb1[1:]}
# df = pd.DataFrame(data=d, index=deltx_store[2:])
# print(df)
#
# # Case 2 Tables
# print('\n------------- Case 2 Simpson qdot_fdm against qdot_exact Simpson-------------')
# d = {'qdot_fdm': flux_array2, 'qdot_extr': q2_exact, 'Percent Error': error_per_store2, 'Beta': simp_storebeta2_exact_q}
# df = pd.DataFrame(data=d, index=deltx_store)
# print(df)
#
# print('------------- Case 2 Simpson qdot_fdm against qdot_extr Simpson-------------')
# d = {'qdot_fdm': flux_array2[2:], 'qdot_extr': storeQe2[1:], 'Percent Error': ext_error_per_store2[1:],
#      'Beta': storeb2[1:]}
# df = pd.DataFrame(data=d, index=deltx_store[2:])
# print(df)
#
# # Case 3 Tables
# print('\n------------- Case 3 Simpson qdot_fdm against qdot_exact Simpson-------------')
# d = {'qdot_fdm': flux_array2, 'qdot_extr': q3_exact, 'Percent Error': error_per_store3, 'Beta': simp_storebeta3_exact_q}
# df = pd.DataFrame(data=d, index=deltx_store)
# print(df)
#
# print('------------- Case 3 Simpson qdot_fdm against qdot_extr Simpson-------------')
# d = {'qdot_fdm': flux_array2[2:], 'qdot_extr': storeQe3[1:], 'Percent Error': ext_error_per_store3[1:],
#      'Beta': storeb3[1:]}
# df = pd.DataFrame(data=d, index=deltx_store[2:])
# print(df)