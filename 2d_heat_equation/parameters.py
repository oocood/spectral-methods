import numpy as np
import math
import numpy.fft as fft

#网格划分
x_length = 2
y_length = 2
x_nodenum = 256
y_nodenum = 256
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
y = y_length/y_nodenum*np.arange(-y_nodenum/2, y_nodenum/2, 1)
X,Y = np.meshgrid(x, y)
kx = fft.fftfreq(x_nodenum, x_length / (2*math.pi*x_nodenum))
ky = fft.fftfreq(y_nodenum, y_length / (2*math.pi*x_nodenum))
kx_grid, ky_grid = np.meshgrid(kx, ky)

#初始值设置
u0 = np.exp(-20*(X**2+Y**2))
u_hat = fft.fft2(u0)

#数值结果存储
file_folder = './output/improved_euler/'

#求解步长
delta_t = 2e-4
total_time = 0.25
calculation_steps = int(total_time/delta_t)
steps_per_saved = 5