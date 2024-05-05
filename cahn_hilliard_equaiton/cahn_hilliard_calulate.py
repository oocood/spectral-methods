import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import math
import parameters as p
import os
import time
import re

#从存储变量的parameters文件中读取参数并进行本地赋值
x_node = p.x_nodenum
y_node = p.y_nodenum
x_length = p.x_length
y_length = p.y_length
kappa = p.kappa
kx_grid = p.kx_grid
ky_grid = p.ky_grid

#中间变量
A_kappa_k4 = p.A*p.kappa*(kx_grid**2 + ky_grid**2)**2
j = complex(0, 1)
ik = j*(kx_grid + ky_grid)
ik2 = j*(kx_grid**2 + ky_grid**2)
ikx = j*kx_grid
iky = j*ky_grid
ikx_k2 = j*kx_grid*(kx_grid**2 + ky_grid**2)
iky_k2 = j*ky_grid*(kx_grid**2 + ky_grid**2)

#初始条件
u0 = np.random.uniform(0.45, 0.55, size = (x_node, y_node)) #随机初始条件
'''
half = int(p.x_nodenum/2)
u0 = np.zeros((p.x_nodenum, p.y_nodenum))
u0[:,0:half] = 0.15
u0[:, half:p.x_nodenum] = 0.85
'''
print(u0)
u_hat = fft.fft2(u0)
delta_t = p.delta_t_adaptive #迭代初始时间步长
total_time = p.total_time #总时间步长

#存储路径
u_path = './output1/u_result/'
dt_path = './output1/dt_adap.txt'

#自适应步长进行计算
def ascher222(delta_t, u_hat): #两部显式，两步隐式，二阶精度
    dt_gamma = delta_t*p.gamma
    u = fft.ifft2(u_hat)
    f_c1 = fft.fft2(20*u**2*(u-1) + 20*u*(u-1)**2 + np.log(u) - np.log(1-u))
    #righthand_term1 = ik*fft.fft2(u*(1-u)*fft.ifft2(f_c1 + p.kappa*ik3*u_hat))
    righthand_term1 = ikx*fft.fft2(u*(1-u)*fft.ifft2(ikx*f_c1 + p.kappa*ikx_k2*u_hat)) + \
    iky*fft.fft2(u*(1-u)*fft.ifft2(iky*f_c1 + p.kappa*iky_k2*u_hat))
    k1_hat = A_kappa_k4*u_hat + righthand_term1
    k1 = -A_kappa_k4*(u_hat + dt_gamma*k1_hat)/(1+A_kappa_k4*dt_gamma)
    u1_hat = u_hat + dt_gamma*k1 + dt_gamma*k1_hat
    u1 = fft.ifft2(u1_hat)
    #f_c2 = ik*fft.fft2(20*u1**2*(u1-1) + 20*u1*(u1-1)**2 + np.log(u1) - np.log(1-u1))
    f_c2 = fft.fft2(20*u1**2*(u1-1) + 20*u1*(u1-1)**2 + np.log(u1) - np.log(1-u1))
    righthand_term2 = ikx*fft.fft2(u1*(1-u1)*fft.ifft2(ikx*f_c2 + p.kappa*ikx_k2*u1_hat)) + \
    iky*fft.fft2(u1*(1-u1)*fft.ifft2(iky*f_c2 + p.kappa*iky_k2*u1_hat))
    k2_hat = A_kappa_k4*u1_hat + righthand_term2 
    k2 = -A_kappa_k4*(u_hat + delta_t*(1-p.gamma)*k1 + delta_t*p.delta*k1_hat + delta_t*(1-p.delta)*k2_hat)/(1 + A_kappa_k4*dt_gamma)
    u_hat = u_hat + delta_t*(1 - p.gamma)*k1 + delta_t*p.gamma*k2 + delta_t*p.delta*k1_hat + delta_t*(1-p.delta)*k2_hat
    return fft.ifft2(u_hat).real

def forward_backward_euler(delta_t, u_hat): 
    u = fft.ifft2(u_hat)
    f_c1 = fft.fft2(20*u**2*(u-1) + 20*u*(u-1)**2 + np.log(u) - np.log(1-u))
    righthand_term1 = ikx*fft.fft2(u*(1-u)*fft.ifft2(ikx*f_c1 + p.kappa*ikx_k2*u_hat)) + \
    iky*fft.fft2(u*(1-u)*fft.ifft2(iky*f_c1 + p.kappa*iky_k2*u_hat))
    k1_hat = A_kappa_k4*u_hat + righthand_term1
    k1 = -A_kappa_k4*(u_hat + delta_t*k1_hat)/(1 + A_kappa_k4*delta_t)
    u_hat = u_hat + delta_t*k1 + delta_t*k1_hat
    return fft.ifft2(u_hat).real

#判断是否接受这次迭代（判断条件是：二阶精度算法和一阶精度算法结果的差是否再给定的控制误差以内）
def accepted_iteration(delta_t, u_hat):
    flag = 1
    #u = np.array([0.0]*len(u_hat), dtype='float32')
    while(1):
        u1 = forward_backward_euler(delta_t, u_hat)
        u2 = ascher222(delta_t, u_hat)
        flag = flag + 1
        estimated_error = np.linalg.norm((u1-u2), ord = 2)/(x_node*y_node)
        #print('in this iteration the estimated error is:', estimated_error)
        if estimated_error<p.controlled_error:
            delta_t = 0.95*delta_t*(estimated_error/p.controlled_error)**(-0.5)
            u = u2
            break
        else:
            delta_t = 0.95*delta_t*(estimated_error/p.controlled_error)**(-0.5)
        if flag>=100:
            print('calculation error, the iteration has come over limit number\n')
            u = -1.0 
            break
    return [u, delta_t]


#main 函数部分
start_time = time.time()
estimated_iteration_num = int(total_time/delta_t)
u = u0
t_eval = 0.0
iteration_num_to_stored_num_rate = 10
origin_u_path = os.path.join(u_path, '0.txt')
with open(origin_u_path, 'w') as f1:
    np.savetxt(f1, u, fmt = '%f')
f1.close()
with open(dt_path, 'w') as f2:
    np.savetxt(f2, [t_eval], fmt = '%.15f')
    for i in range(1, estimated_iteration_num, 1):
        [u, delta_t] = accepted_iteration(delta_t, u_hat)
        t_eval = t_eval + delta_t
        u_hat = fft.fft2(u)
        if(i%iteration_num_to_stored_num_rate==0):
            print('the', i, 'iteration has finished, present time step is', delta_t)
            path = str(i)+'.txt'
            path = os.path.join(u_path, path)
            with open(path, 'w') as f1:
                np.savetxt(f1, u, fmt = '%f')
            np.savetxt(f2, [t_eval], fmt = '%.15f')
            f1.close()
        if(t_eval>total_time): break
end_time1 = time.time()
print("the calculation part has finished already", 'time cost is', end_time1-start_time, ' s')
