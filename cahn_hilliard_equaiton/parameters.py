import numpy as np
import math
import numpy.fft as fft

# two demension x and y, the interval of x is[-1,1],the same as y demension
x_length = 2
y_length = 2

#parameter used in pde to be solved
kappa = 1e-3
j = complex(0, 1)

# to use fft, devide the interval into 128 pieces.
x_nodenum = 256
y_nodenum = 256

#time limitation and time step control
total_time = 1
time_step_upperbound = 1e-3
time_step_lowerbound = 1e-9

#set origin time step to start the calculation
delta_t_adaptive = 1e-7
t_list = np.arange(0, total_time, delta_t_adaptive)
calculate_steps = int(total_time/delta_t_adaptive)

#controlled error setting
controlled_error = 1e-7

#ascher222 and ascher111 coef in butcher table
gamma = (2-math.sqrt(2))/2
delta = 1 - 1/(2*gamma)

#A(the parameter to control the stability of the euqation)
A = 0.5
adaptive_expand_rate = 40

#计算网格划分
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
y = y_length/y_nodenum*np.arange(-y_nodenum/2, y_nodenum/2, 1)
x_grid, y_grid = np.meshgrid(x, y)
kx = fft.fftfreq(x_nodenum, x_length/(2*math.pi*x_nodenum))
ky = fft.fftfreq(y_nodenum, y_length/(2*math.pi*y_nodenum))
kx_grid, ky_grid = np.meshgrid(kx, ky)