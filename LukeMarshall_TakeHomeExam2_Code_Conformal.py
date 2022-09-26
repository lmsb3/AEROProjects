# Note: This code was originally Blake Rogers code, but was sent out by Dr. Strouboulis to be slightly altered
# for the purposes of Spring 2022 AERO 430 Final Exam

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import timeit
import pandas as pd
import warnings
start_tot = timeit.default_timer()
warnings.filterwarnings("ignore")

def richardsons(Q):
    i = len(Q) - 1
    Qe = (Q[i-1]**2 - Q[i-2]*Q[i])/(2*Q[i-1]-Q[i-2]-Q[i])
    beta = (np.log(np.abs((Qe-Q[i-2]))/np.abs((Qe-Q[i-1]))))/np.log(2)
    return (Qe, beta)
def richardsons_exact(Q, exact):
    # Computes beta convergence from Richardson Extrapolation
    i = len(Q) - 1
    beta = (np.log(np.abs((exact-Q[i-1]))/np.abs((exact-Q[i]))))/np.log(2)
    return beta
def U1(y, split, K, Ubar):
    U = Ubar * np.sinh(K*np.pi*y) / np.sinh(K*np.pi*split)
    return U
def U2(y, split, K, Ubar):
    U = (100-Ubar*np.cosh(K*np.pi*(1-split))) * (np.sinh(K*np.pi*(y-split))) / (np.sinh(K*np.pi*(1-split))) + Ubar*np.cosh(K*np.pi*(y-split))
    return U
def U(split,K1,K2,kyy1,kyy2):
    U = 100/(np.sinh(K2*np.pi*(1-split))) * 1 / ((kyy1/kyy2) * (K1/K2) * (1/(np.tanh(K1*np.pi*split))) + (1 / (np.tanh(K2*np.pi*(1-split)))))
    return U
def simpson(a, b, n, input_array):
    sum = 0
    for k in range(n + 1):
        summand = input_array[k]
        if (k != 0) and (k != n):
            summand *= (2 + (2 * (k % 2)))
        sum += summand
    return ((b - a) / (3 * n)) * sum


f =  6 # Highest mesh order
mesh_order = np.linspace(1, f, f) # Generating mesh order matrix

kxx1 = 1
k1 = [0.2, 0.75, 1.75, 2.5, 6, 0.2, 12, 12]


kxx2 = 1
k2 = [0.1, 0.5, 3.5, 5, 6, 12, 12, 14]

split = np.pi/5

for j in range(0,len(k1)):
    start = timeit.default_timer()
    start = timeit.default_timer()
    store_Q_simp = []
    store_Q_richard = []
    store_B_richard = []
    store_B_exact = []
    store_relerror_simp = []
    store_relerror_rich = []
    dx_store = []
    store_exact = []
    for i in mesh_order:
        n = int(2**i)
        dx = 1/n # Solving for mesh size
        dy1 = split/(n/2)
        dy2 = (1-split)/(n/2)
        dx_store.append(dy1)
        gk = np.zeros(((n+1)**2, (n+1)**2))
        Kxx1 = kxx1
        Kyy1 = Kxx1/(k1[j]**2)
        Kxx2 = kxx2
        Kyy2 = Kxx2/k2[j]**2
        K1 = np.round(np.sqrt(Kxx1/Kyy1), 2)
        K2 = np.round(np.sqrt(Kxx2/Kyy2), 2)
        
        setU = U(split, K1, K2, Kyy1, Kyy2)
        
        # Analytical Plot
        ### ------NOTE: TO VIEW EACH SEPERATE PLOT, UNCOMMENT THIS SECTION. IT IS COMMENTED ------
        ### ---------SO THAT THE CONVERGENCE PLOTS ARE ALL ON ONE ---------------------------
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # plt.title('Analytical Temperature Distribution for dx='+str(round(dx, 5))+' and K1='+str(K1)+' K2='+str(K2))
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel(r'T(x, y)$\degree$C')
        #
        # X_linespace = np.linspace(0, 1.0, 2**int(i) + 1, endpoint = True)
        # Y_linespace = np.linspace(0, 1.0, 2**int(i) + 1, endpoint = True)
        # X_linespace, Y_linespace = np.meshgrid(X_linespace, Y_linespace)
        # us = np.where(Y_linespace > split, U2(Y_linespace, split, K2, setU)*np.sin(np.pi*X_linespace), U1(Y_linespace, split, K1, setU)*np.sin(np.pi*X_linespace))
        #
        # surf = ax.plot_surface(X_linespace, Y_linespace, us, cmap=cm.inferno, linewidth=10)
        # plt.show()
        #
        gk = np.zeros(((n+1)**2, (n+1)**2)) # Global matrix
        sol = np.zeros(((n+1)**2, 1))  # Solution matrix
        connec = np.zeros(((n+1), (n+1))) # Nodal points matrix
        element = 0
        # Filling up nodal points matrix
        for row in range(0, n+1):
            for col in range(0, n+1):
                connec[row, col] = element
                element += 1
                
        ek1 = np.zeros((4,4)) # initialize element matrix
        for diag in range(0, 4):
            ek1[diag, diag] = (dy1*Kxx1/dx) + (dx*Kyy1/dy1)
        ek1[0,1] = -(dy1/dx)*Kxx1
        ek1[0,3] = -(dx/dy1)*Kyy1
        ek1[1,0] = -(dy1/dx)*Kxx1
        ek1[1,2] = -(dx/dy1)*Kyy1
        ek1[2,1] = -(dx/dy1)*Kyy1
        ek1[2,3] = -(dy1/dx)*Kxx1
        ek1[3,0] = -(dx/dy1)*Kyy1
        ek1[3,2] = -(dy1/dx)*Kxx1
        
        ek2 = np.zeros((4,4)) # initialize element matrix
        for diag in range(0, 4):
            ek2[diag, diag] = (dy2*Kxx2/dx) + (dx*Kyy2/dy2)
        ek2[0,1] = -(dy2/dx)*Kxx2
        ek2[0,3] = -(dx/dy2)*Kyy2
        ek2[1,0] = -(dy2/dx)*Kxx2
        ek2[1,2] = -(dx/dy2)*Kyy2
        ek2[2,1] = -(dx/dy2)*Kyy2
        ek2[2,3] = -(dy2/dx)*Kxx2
        ek2[3,0] = -(dx/dy2)*Kyy2
        ek2[3,2] = -(dy2/dx)*Kxx2
        
        
        # Setting up nodes
        nodes = np.zeros((4))
        nodes[0] = 0
        nodes[1] = 1
        nodes[2] = n+2
        nodes[3] = n+1
        count = 1
        
        # Generating global matrix
        for iel in range(0, n**2):
            # Accounting for row shift
            if iel - count*n == 0 and iel > 0:
                count += 1
                nodes[0] = nodes[0] + (n+1)
                nodes[1] = nodes[1] + (n+1)
                nodes[2] = nodes[2] + (n+1)
                nodes[3] = nodes[3] + (n+1)
            # Looping through element
            for z in range(0,4):
                ig = nodes[z] + iel - (count-1)*n
                for v in range(0,4):
                    jg = nodes[v] + iel - (count-1)*n
                    if iel < int(n**2/2):
                        gk[int(ig), int(jg)] = gk[int(ig), int(jg)] + ek1[int(z), int(v)]
                    else:
                        gk[int(ig), int(jg)] = gk[int(ig), int(jg)] + ek2[int(z), int(v)]
        
        U_ij = ((2*Kxx1*dy1/dx**2)+(2*Kxx2*dy2/dx**2)+(2*Kyy1/dy1)+(2*Kyy2/dy2))
        U_i_plus_minus = -((Kxx1*dy1/dx**2)+(Kxx2*dy2/dx**2))
        U_j_minus = -2*Kyy1/dy1
        U_j_plus = -2*Kyy2/dy2
        
        for iel in range(0, n**2):
            # Accounting for row shift
            if iel - count*n == 0 and iel > 0:
                count += 1
                nodes[0] = nodes[0] + (n+1)
                nodes[1] = nodes[1] + (n+1)
                nodes[2] = nodes[2] + (n+1)
                nodes[3] = nodes[3] + (n+1)
            # Looping through element
            ig = nodes[z] + iel - (count-1)*n
            if ig > int((((n+1)**2)/2-i)) and ig < int((((n+1)**2)/2+i)):
                gk[int(ig), int(ig-(n+1))] = U_j_minus
                gk[int(ig), int(ig+(n+1))] = U_j_plus
                gk[int(ig), int(ig-1)] = U_i_plus_minus
                gk[int(ig), int(ig+1)] = U_i_plus_minus
                gk[int(ig), int(ig)] = U_ij
                
        # Penalty Method - edited from McElrath code
        ele_known_side = np.append(connec[n, :], connec[:, 0])
        ele_known_topbottom = np.append(connec[0, :], connec[:, n])
        ele_known = np.append(ele_known_topbottom, ele_known_side) # Known element nodal values
        PEN = 1e30 # Penalty value
        xi = np.linspace(0, 1, n+1) # x values along top
        for i6 in range(0, len(ele_known)):
            # Applying penalty method to known global matrix points
            gk[int(ele_known[i6]),int(ele_known[i6])] = PEN
            # Applying penalty method to solution matrix
            if i6 < n+1:
                sol[(n+1)**2-(n+1)+i6] = 100*np.sin(np.pi*xi[i6])*PEN
    
        # Solving for and rearranging temperature matrix
        U_matrix = np.linalg.solve(gk, sol)
        U_matrix = np.reshape(U_matrix, ((n+1),(n+1)))
        
        # Approximating dU/dy using forward difference - updated from Antonios code
        temp_derivative_approx_array = []
        for col in range(0, len(U_matrix)):
            col_checked = U_matrix[:,col]
            max_index = len(col_checked)-1
            second_order_one_sided_difference = (col_checked[max_index-2] - 4*col_checked[max_index-1] + 3*col_checked[max_index])/(2*dy2)
            temp_derivative_approx_array.append(second_order_one_sided_difference)
        
        # Calculating exact and FDM solutions as well as the relative errors
        exactQ = -Kyy2*2*K2*(((100 - setU*np.cosh((1-split)*K2*np.pi))*(np.cosh((1-split)*np.pi*K2))/(np.sinh((1-split)*np.pi*K2))) + setU*np.sinh((1-split)*K2*np.pi))
        # Simpson*s assumes Kyy = 1
        simpQ = -Kyy2*(dx/3) * (temp_derivative_approx_array[0] + 2*sum(temp_derivative_approx_array[:n-2:2]) + 4*sum(temp_derivative_approx_array[1:n-1:2]) + temp_derivative_approx_array[n-1])
        #simpQ = -Kyy2*simpson(0, 1, n, temp_derivative_approx_array)
        store_Q_simp.append(simpQ)
        relerror_simp = np.abs(exactQ-simpQ)/np.abs(exactQ)
        store_relerror_simp.append(relerror_simp)
        store_exact.append(exactQ)
    
        #Exact Beta calcs and Richardson Extrapolation work
        if i >= 2:
            beta = richardsons_exact(store_Q_simp, exactQ)
            store_B_exact.append(round(beta, 5))
        if i > 2:
            Qe, beta_ex = richardsons(store_Q_simp)
            store_Q_richard.append(round(Qe, 5))
            store_B_richard.append(round(beta_ex, 5))
            elerror_rich = round(np.abs(simpQ-Qe)/np.abs(Qe), 5)
            store_relerror_rich.append(elerror_rich)
        
        # 3D plot of temperature distribution - edited from Diaz code
        ### ------NOTE: TO VIEW EACH SEPERATE PLOT, UNCOMMENT THIS SECTION. IT IS COMMENTED ------
        ### ---------SO THAT THE CONVERGENCE PLOTS ARE ALL ON ONE ---------------------------
        # fig = plt.figure(1)
        # ax = fig.gca(projection='3d')
        # plt.title('Conformal FDM Temperature Distribution for dx='+str(round(dx, 5))+' and K1='+str(K1)+' K2='+str(K2))
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel(r'T(x, y)$\degree$C')
        #
        # X = np.linspace(0, 1.0, 2**int(i) + 1, endpoint = True)
        # Y = np.linspace(0, split, int(n/2)+1, endpoint = True)
        # Y = np.append(Y, np.linspace(split+dy2, 1.0, int(n/2), endpoint = True))
        # X, Y = np.meshgrid(X, Y)
        #
        # surf = ax.plot_surface(X, Y, U_matrix, cmap=cm.inferno,linewidth=10)
        # #ax.view_init(0, 0)
        # plt.show()
        
    # Relative Error Plot Exact
    plt.figure(2)
    plt.loglog(dx_store, store_relerror_simp, 'o-', label = 'K1='+str(np.round(K1,2))+' K2='+str(np.round(K2,2)))
    plt.xlabel('log(dy1)')
    plt.ylabel('log(Error)')
    plt.title('Conformal Error vs. dy1 for 2nd Order FDM against Exact')
    plt.legend(prop={'size': 8})
    plt.xlim([max(dx_store)*1.1, min(dx_store)*0.9])
    plt.grid(True)
    
    # Relative Error Plot Extrapolated
    plt.figure(3)
    plt.loglog(dx_store[2:], store_relerror_rich, 'o-',label = 'K1='+str(np.round(K1,2))+' K2='+str(np.round(K2,2)))
    plt.xlabel('log(dy1)')
    plt.ylabel('log(Error)')
    plt.title('Conformal Error vs. dy1 for 2nd Order FDM against Extrapolated')
    plt.legend(prop={'size': 8})
    plt.xlim([max(dx_store[2:])*1.1, min(dx_store[2:])*0.9])
    plt.grid(True)
    
    # Printing Results
    nan_matrix = np.full(1, '-')
    store_B_exact = np.concatenate((nan_matrix, store_B_exact))
    print('\nQ against Q_exact for K1='+str(np.round(K1,2))+' K2='+str(np.round(K2,2)))
    d = {'Q_fdm': store_Q_simp, 'Q_exact': store_exact, 'Percent Error': store_relerror_simp, 'Beta': store_B_exact}
    df = pd.DataFrame(data=d, index = dx_store)
    df.index.name = 'dy1'
    print(df)
    
    nan_matrix = ['-', '-']
    store_B_richard = np.concatenate((nan_matrix, store_B_richard))
    store_Q_richard = np.concatenate((nan_matrix, store_Q_richard))
    store_relerror_rich = np.concatenate((nan_matrix, store_relerror_rich))
    print('\nQ against Q_extrapolated for K1='+str(np.round(K1,2))+' K2='+str(np.round(K2,2)))
    d = {'Q_fdm': store_Q_simp, 'Q_extrapolated': store_Q_richard, 'Percent Error': store_relerror_rich, 'Beta': store_B_richard}
    df = pd.DataFrame(data=d, index = dx_store)
    df.index.name = 'dy1'
    print(df)
     
    stop = timeit.default_timer()
    print('Run time for K1='+str(np.round(K1,2))+' K2='+str(np.round(K2,2))+':', round(stop - start, 3), 's\n')   
    print('------------------')

plt.show()
stop_tot = timeit.default_timer()
print('Total run time:', round(stop_tot - start_tot, 3), 's')   