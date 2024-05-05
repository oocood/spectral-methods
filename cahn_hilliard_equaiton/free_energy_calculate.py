import numpy as np
import parameters as p
import os
import numpy.fft as fft
import time
import re

#自由能密度函数
def f_c(c):
    res = 10*c**2*(c-1)**2 + c*np.log(c) + (1-c)*np.log(1-c)
    return res

start_time = time.time() #计算初始时刻

#从parameters读取变量
j = p.j
kx_grid = p.kx_grid
ky_grid = p.ky_grid
delta_x = p.x_length/p.x_nodenum
delta_y = p.y_length/p.y_nodenum
kappa = p.kappa/2  #计算时候用的kappa参数是计算自由能参数时表示flory-huggins系数的两倍

#数据读取路径
dt_path = './output1/dt_adap.txt'
fig_path = './output1/fig/'
free_energy_path = './output1/free_energy/free_energy.txt'
facial_energy_path = './output1/free_energy/facial_energy.txt'

# 数据读取和自由能计算
file_folder = './output1/u_result'
files = os.listdir(file_folder)
sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group())) #数据跟去前面的迭代次数排序
print(sorted_files)
file_num = len(files)
with open(facial_energy_path, 'w') as f3, open(free_energy_path, 'w') as f4:
    for i in range(file_num):
        data_path = os.path.join(file_folder, sorted_files[i])
        data = np.loadtxt(data_path)
        #print(data.shape, i) #检查数组大小是否异常
        data_hat = fft.fft2(data)
        du_dx = fft.ifft2(j*kx_grid*data_hat).real
        du_dy = fft.ifft2(j*ky_grid*data_hat).real
        facial_energy = np.sum(kappa*(du_dx**2 + du_dy**2))*delta_x**2
        free_energy = np.sum(f_c(data))*delta_x**2 + facial_energy
        np.savetxt(f3, [facial_energy], fmt = '%f')
        np.savetxt(f4, [free_energy], fmt = '%f')
f3.close()
f4.close()
end_time = time.time()
print("the free energy calculation part has finished", 'time cost is', end_time-start_time, ' s')
