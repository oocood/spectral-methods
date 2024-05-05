import numpy as np
import math
import numpy.fft as fft

#网格划分
x_length = 2
x_nodenum = 256
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
k = fft.fftfreq(x_nodenum, x_length / (2 * math.pi * x_nodenum))

#初始条件
initial_path = './output/initial_value.txt'
#u0 = np.random.uniform(-0.1, 0.1, size=(1, x_nodenum))
u0 = np.loadtxt(initial_path).reshape(-1, x_nodenum)
u0 = np.squeeze(u0)
u_hat = fft.fft(u0)

#方程参数
lambda1 = 0.01

#总时间和求解时间步长
total_time = 100
delta_t = 1e-4
calculation_steps = int(total_time/delta_t)
steps_per_saved = 5000

#数值结果存储路径
path = './output/imex'+str(delta_t)+'.txt'