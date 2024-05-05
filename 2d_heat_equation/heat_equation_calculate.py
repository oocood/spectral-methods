import numpy as np
import numpy.fft as fft
import math
import parameters as p
import os
import time

u0 = p.u0
u_hat = p.u_hat
kx_grid = p.kx_grid
ky_grid = p.ky_grid
j = complex(0, 1)
delta_t = p.delta_t #求解步长
half_ik2 = 0.5*delta_t*(kx_grid**2+ky_grid**2) #存储常用中间变量
ik2_dt = delta_t*(kx_grid**2+ky_grid**2)
calculation_steps = p.calculation_steps #总求解步长
file_folder = p.file_folder #数值结果存储的文件目录(梯形公式)
#file_folder = './output/forward_euler/' #（向前欧拉）
steps_per_saved = p.steps_per_saved #每个多少步存一步
path0 = '0.txt'
path0 = os.path.join(file_folder, path0)
start_time = time.time()
with open(path0, 'w') as f0: #存初始分布
    np.savetxt(f0, u0, fmt = '%f')
f0.close()

for i in range(1, calculation_steps, 1):
    u_hat = (1-half_ik2)/(1+half_ik2)*u_hat
    u = fft.ifft2(u_hat).real
    if(i%steps_per_saved==0):
        path = str(round(i*delta_t, 3))+'.txt'
        path = os.path.join(file_folder, path)
        with open(path, 'w') as f:
            np.savetxt(f, u, fmt = '%f')
        f.close()
end_time = time.time()
print('all the calculation has done', 'time cost is', end_time - start_time, ' s')

'''
for i in range(1, calculation_steps, 1):
    u_hat = (1-ik2_dt)*u_hat
    u = fft.ifft2(u_hat).real
    if(i%steps_per_saved==0):
        path = str(round(i*delta_t, 3))+'.txt'
        path = os.path.join(file_folder, path)
        with open(path, 'w') as f:
            np.savetxt(f, u, fmt = '%f')
        f.close()
end_time = time.time()
print('all the calculation has been done', 'time cost is', end_time - start_time, 's')
'''