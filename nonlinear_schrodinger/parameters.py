import numpy as np
import numpy.fft as fft
import math

#变量设定
x_length = 20
x_nodenum = 256
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
k = fft.fftfreq(x_nodenum, x_length / (2 * math.pi * x_nodenum))
total_time = 5
delta_t = 1e-6 
j = complex(0,1)
idt = j*delta_t #i*delta_t
ik2dt = j*k**2*delta_t #i*k**2*delta_t

#存储路径
path1 = './output/forward_euler.txt'
path2 = './output/imex.txt'

#初始值
u0 = 1/np.cosh(x)
u_hat = fft.fft(u0)

#计算部分
u_hat1 = u_hat #显式欧拉
u_hat2 = u_hat #一阶imex
u1 = u0
u2 = u0
calculate_steps = int(total_time/delta_t)
steps_per_saved = 10000