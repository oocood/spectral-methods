import numpy as np
import numpy.fft as fft
import math
import time
import parameters as p

#变量设定
x_length = p.x_length
x_nodenum = p.x_nodenum
x = x_length/x_nodenum*np.arange(-x_nodenum/2, x_nodenum/2, 1)
k = fft.fftfreq(x_nodenum, x_length / (2 * math.pi * x_nodenum))
total_time = p.total_time
delta_t = p.delta_t
j = complex(0,1)
ik = j*k
ik3 = (j*k)**3

#存储路径
path1 = './output/forward_euler.txt'
path2 = './output/imex.txt'
#初始值
u0 = 1/(np.cosh(x))**2
u_hat = fft.fft(u0)

#计算部分
u_hat1 = u_hat #显式欧拉
u_hat2 = u_hat #一阶imex
u1 = u0
u2 = u0
calculate_steps = int(total_time/delta_t)

start_time = time.time()
#with open(path1, 'w') as f1, open(path2, 'w') as f2:
with open(path2, 'w') as f2:
    np.savetxt(f2, u2, fmt = '%f')
    #np.savetxt(f1, u1, fmt = '%f')
    for i in range(1,calculate_steps, 1):
        #u_hat1 = u_hat1 - delta_t*(ik*u_hat1 + 12*fft.fft(u1*fft.ifft(ik*u_hat1))+ik3*u_hat1)
        u_hat2 = (u_hat2 - delta_t*(ik*u_hat2 + 12*fft.fft(u2*fft.ifft(ik*u_hat2))))/(1+delta_t*ik3)
        #u1 = fft.ifft(u_hat1).real
        u2 = fft.ifft(u_hat2).real
        if(i%1==0):
            #np.savetxt(f1, u1, fmt = '%f')
            np.savetxt(f2, u2, fmt = '%f')
end_time = time.time()
print('all the calculation has done time cost is', end_time-start_time, ' s')