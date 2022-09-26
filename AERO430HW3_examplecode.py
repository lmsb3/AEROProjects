import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from math import *
import csv
from sympy import *
import sys

# Analytical Solution
def temp(x,y,k):
    u = (100 * np.sinh(k * np.pi * y) * np.sin(np.pi * x)/np.sinh(k * np.pi))
    return u
def richardson(list_estimates):
    richardson_estimate = (list_estimates[-2]**2-list_estimates[-1]*list_estimates[-3])/(2 * list_estimates[-2] - (list_estimates[-3] + list_estimates[-1]))
    return richardson_estimate
def q_analytical(k):
    return float(- (200 * k * np.cosh(k * np.pi))/np.sinh(k*np.pi))

def convergence(true_value,fdm_list,n_mesh,name_conv_data):
    # Determine the convergence at the interface:
    true_value = float(true_value)
    percent_error_against_real = []
    percent_error_against_richardson = []
    richardson_value = richardson(fdm_list)
    dx_list = []
    Beta_list_real = []
    Beta_list_richardson = []
    for n in range(len(fdm_list)):
        percent_error_against_real.append(abs(true_value - fdm_list[n]) / abs(true_value) * 100)
        percent_error_against_richardson.append(
            abs(richardson_value - fdm_list[n]) / abs(richardson_value) * 100)
        dx_list.append(1/n_mesh[n])
    for i in range(1, len(dx_list)):
        Beta_list_real.append((log(abs(true_value - fdm_list[i])) - log(abs(true_value - fdm_list[i-1])))/
                              (log(dx_list[i]) - log(dx_list[i-1])))
        Beta_list_richardson.append((log(abs(richardson_value - fdm_list[i])) - log(abs(richardson_value - fdm_list[i - 1]))) /
                              (log(dx_list[i]) - log(dx_list[i - 1])))
    # Plot Results for Analytical
    plt.plot(dx_list, percent_error_against_real)
    plt.xlabel("dx")
    plt.ylabel('% Error')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(max(dx_list), min(dx_list))
    plt.grid()
    title1 = name_conv_data + ' Error Against Analytical Solution'
    plt.title(title1)
    plt.show()
    # Plot results for Richardson
    plt.plot(dx_list, percent_error_against_richardson)
    plt.xlabel("dx")
    plt.ylabel('% Error')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(max(dx_list), min(dx_list))
    plt.grid()
    title2 = name_conv_data + ' Error Against Richardson Solution'
    plt.title(title2)
    plt.show()

    with open(title1 + '.csv', 'w') as real:
        aerowriter = csv.writer(real, lineterminator='\n')
        aerowriter.writerow(['dx(cm)', name_conv_data + ' FDM', name_conv_data + ' Analytical', name_conv_data + ' % Error', 'B'])
        for n in range(len(percent_error_against_real)):
            if n == 0:
                aerowriter.writerow(
                    [float(dx_list[n]), float(fdm_list[n]), float(true_value), float(percent_error_against_real[n]), 'NaN'])
            else:
                aerowriter.writerow([float(dx_list[n]), float(fdm_list[n]), float(true_value), float(percent_error_against_real[n]),
                                     float(Beta_list_real[n-1])])
    # Write Convergence to CSV Files
    with open(title2 + '.csv', 'w') as real:
        aerowriter = csv.writer(real, lineterminator='\n')
        aerowriter.writerow(
            ['dx(cm)', name_conv_data + ' FDM', name_conv_data + ' Richardson', name_conv_data + ' % Error', 'B'])
        for n in range(len(percent_error_against_richardson)):
            if n == 0:
                aerowriter.writerow(
                    [float(dx_list[n]), float(fdm_list[n]), float(richardson_value), float(percent_error_against_richardson[n]), 'NaN'])
            else:
                aerowriter.writerow([dx_list[n], float(fdm_list[n]), float(richardson_value), float(percent_error_against_richardson[n]),
                                         float(Beta_list_richardson[n - 1])])


def graph(k):
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    X, Y = np.meshgrid(x, y)
    Z = temp(X,Y,k)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temp(c)')
    ax.set_title('Analytical Solution: ' + str( k))
    with open('Analytical_hw_3_k_01' + '.csv', 'w') as real:
        aerowriter = csv.writer(real, lineterminator='\n')
        aerowriter.writerow(
            ['X' , 'Y', 'U(C)'])
        for n in range(len(Z)):
            if n % 2 == 0:
                aerowriter.writerow([X[n][10], Y[n][10], Z[n][10]])
    plt.show()

graph(.1)

# fdm solution
def fdm(k,num_elem):
    # Determine delta x
    delta_x = (1/(num_elem+2))
    # Create the global matrix
    x_pos = num_elem * num_elem
    y_pos = num_elem * num_elem
    global_m = np.zeros((x_pos,y_pos))
    for r in range(x_pos):
        for c in range(y_pos):
            if r == c:
                global_m[r][c] = -2 * (k**2 + 1)
            elif c == (r + 1) and (r+1) % num_elem != 0:
                global_m[r][c] = k**2
            elif c == r + num_elem:
                global_m[r][c] = 1
            elif c == r - num_elem:
                global_m[r][c] = 1
            elif c == (r-1) and r % num_elem != 0:
                global_m[r][c] = k**2
    # Create the b matrix
    b_matrix = np.zeros((x_pos,1))
    for r in range(x_pos):
        if r < num_elem:
            b_matrix[r][0] = -100 * sin((r+1)/(num_elem+1) * pi)
    u_matrix = np.linalg.solve(global_m,b_matrix)
    u_total = []
    u_top = []
    # Determine the top boundary
    x_to_plot = np.linspace(1, 0, num_elem + 2)
    y_to_plot = np.linspace(1, 0, num_elem + 2)

    for i in x_to_plot:
        u_top.append(100 * np.sin(i * np.pi))
    u_total.append(u_top)

    for i in range(num_elem):
        u_local = []
        u_local.append(0)
        for z in range(num_elem):
            u_local.append(u_matrix[i*num_elem + z][0])
        u_local.append(0)
        u_total.append(u_local)
    u_bottom = []
    for i in range(num_elem+2):
        u_bottom.append(0)
    u_total.append(u_bottom)

    # Plot the solution
    X, Y = np.meshgrid(x_to_plot, y_to_plot)

    # Determine the quantity of interest

    # Determine the slope of temperarture using second order FDM
    dt_dy = []
    for i in range(len(u_total)):
        dt_dy.append(-(u_total[2][i] - 4 * u_total[1][i] + 3 * u_total[0][i])/(2 * delta_x))

    # Apply simpsons rules
    q = 0
    for i in range(len(dt_dy)):
        if i == 0:
            q += delta_x / 3 * (dt_dy[0])
        elif i == len(dt_dy)-1:
            q += delta_x / 3 * (dt_dy[i])
        elif i % 2 != 0:
            q += delta_x / 3 * 2 * (dt_dy[i])
        elif i % 2 == 0:
            q += delta_x / 3 * 4 * (dt_dy[i])
        else:
            print('Error')

    return X, Y, u_total, q, delta_x



# Setupplot
fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y , u_total,q,dx  = fdm(.1,15)
ax.contour3D(X, Y, u_total, 50, cmap='coolwarm')

# After plotting
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temp(c)')
ax.set_title('FDM for k = .1 ')
# Enter the data into a CSV file
with open('FDM_hw_3_k_01' + '.csv', 'w') as real:
    aerowriter = csv.writer(real, lineterminator='\n')
    aerowriter.writerow(
        ['X', 'Y', 'U(C)'])
    for n in range(len(u_total)):
        aerowriter.writerow([X[n][10], Y[n][10], u_total[n][10]])

plt.show()

q_list = []
num_mesh_list = []
for r in range(2,40,2):
    q_value, dx = fdm(.1,r)[3], fdm(.1,r)[4]
    q_list.append(q_value)
    num_mesh_list.append(1/dx)

convergence(q_analytical(.1),q_list,num_mesh_list,"Heat")

# Find values for differing k
k_list = [.1,.25,.5,.75,1,2,6,10]


q_list_differ_k = []
num_mesh_differ_k = []
for k in k_list:
    q_local = []
    num_mesh_local = []
    for r in range(2, 30, 2):
        q_local.append(fdm(k, r)[3])
        num_mesh_local.append(1/fdm(k, r)[4])
    q_list_differ_k.append(q_local)
    num_mesh_differ_k.append(num_mesh_local)


def convergence_real(true_value,fdm_list,n_mesh,k):
    # Determine the convergence at the interface:
    true_value = float(true_value)
    percent_error_against_real = []
    percent_error_against_richardson = []
    richardson_value = richardson(fdm_list)
    dx_list = []
    Beta_list_real = []
    Beta_list_richardson = []
    for n in range(len(fdm_list)):
        percent_error_against_real.append(abs(true_value - fdm_list[n]) / abs(true_value) * 100)
        percent_error_against_richardson.append(
            abs(richardson_value - fdm_list[n]) / abs(richardson_value) * 100)
        dx_list.append(1/n_mesh[n])
    for i in range(1, len(dx_list)):
        Beta_list_real.append((log(abs(true_value - fdm_list[i])) - log(abs(true_value - fdm_list[i-1])))/
                              (log(dx_list[i]) - log(dx_list[i-1])))
        Beta_list_richardson.append((log(abs(richardson_value - fdm_list[i])) - log(abs(richardson_value - fdm_list[i - 1]))) /
                              (log(dx_list[i]) - log(dx_list[i - 1])))
    if k == 6:
        with open('k_6_against_real' + '.csv', 'w') as real:
            aerowriter = csv.writer(real, lineterminator='\n')
            aerowriter.writerow(
                ['dx(cm)', 'heat ' + ' FDM', 'heat ' + ' Analytical', 'heat ' + ' % Error', 'B'])
            for n in range(len(percent_error_against_real)):
                if n == 0:
                    aerowriter.writerow(
                        [float(dx_list[n]), float(fdm_list[n]), float(true_value), float(percent_error_against_real[n]),
                         'NaN'])
                else:
                    aerowriter.writerow(
                        [float(dx_list[n]), float(fdm_list[n]), float(true_value), float(percent_error_against_real[n]),
                         float(Beta_list_real[n - 1])])
    # Plot Results for Analytical
    plt.plot(dx_list, percent_error_against_real,'-o',  label = 'K: '+str( k))
    plt.xlabel("dx")
    plt.ylabel('% Error')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(max(dx_list), min(dx_list))
    plt.grid()


# Richardson Convergence
def convergence_richardson(true_value,fdm_list,n_mesh,k):
    # Determine the convergence at the interface:
    true_value = float(true_value)
    percent_error_against_real = []
    percent_error_against_richardson = []
    richardson_value = richardson(fdm_list)
    dx_list = []
    Beta_list_real = []
    Beta_list_richardson = []
    for n in range(len(fdm_list)):
        percent_error_against_real.append(abs(true_value - fdm_list[n]) / abs(true_value) * 100)
        percent_error_against_richardson.append(
            abs(richardson_value - fdm_list[n]) / abs(richardson_value) * 100)
        dx_list.append(1/n_mesh[n])
    for i in range(1, len(dx_list)):
        Beta_list_real.append((log(abs(true_value - fdm_list[i])) - log(abs(true_value - fdm_list[i-1])))/
                              (log(dx_list[i]) - log(dx_list[i-1])))
        Beta_list_richardson.append((log(abs(richardson_value - fdm_list[i])) - log(abs(richardson_value - fdm_list[i - 1]))) /
                              (log(dx_list[i]) - log(dx_list[i - 1])))
    if k == 6:
        # Write Convergence to CSV Files
        with open('k_6_richardson' + '.csv', 'w') as real:
            aerowriter = csv.writer(real, lineterminator='\n')
            aerowriter.writerow(
                ['dx(cm)', 'heat ' + ' FDM', 'heat ' + ' Richardson', 'heat ' + ' % Error', 'B'])
            for n in range(len(percent_error_against_richardson)):
                if n == 0:
                    aerowriter.writerow(
                        [float(dx_list[n]), float(fdm_list[n]), float(richardson_value),
                         float(percent_error_against_richardson[n]), 'NaN'])
                else:
                    aerowriter.writerow([dx_list[n], float(fdm_list[n]), float(richardson_value),
                                         float(percent_error_against_richardson[n]),
                                         float(Beta_list_richardson[n - 1])])

    # Plot results for Richardson
    plt.plot(dx_list, percent_error_against_richardson, '-o',  label = 'K: '+str( k))
    plt.xlabel("dx")
    plt.ylabel('% Error')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(max(dx_list), min(dx_list))
    plt.grid()



# Define the real q list

i = 0

for k in k_list:
    convergence_real(q_analytical(k),q_list_differ_k[i],num_mesh_differ_k[i],k)
    i += 1
plt.title('Heat Transfer Against Analytical')
plt.legend()
plt.show()


i = 0

for k in k_list:
    convergence_richardson(q_analytical(k),q_list_differ_k[i],num_mesh_differ_k[i],k)
    i += 1
plt.title('Heat Transfer Against Richardson')
plt.legend()
plt.show()