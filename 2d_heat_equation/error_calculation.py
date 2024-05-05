import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import parameters as p
import os
import re

def sort_key(name):
    return int(re.search(r'(\d+)', name).group(1))

x_nodenum = p.x_nodenum
y_nodenum = p.y_nodenum
delta_t = p.delta_t
steps_per_saved = p.steps_per_saved
t_list = np.arange(0, p.total_time, steps_per_saved*delta_t)
file_folder = p.file_folder
#file_folder = './output/forward_euler/'
kx_grid = p.kx_grid
ky_grid = p.ky_grid
u0 = p.u0
#t_list = np.arange(0, p.total_time, p.delta_t*p.steps_per_saved)
file_list = os.listdir(file_folder)
file_list = sorted(file_list, key=sort_key)
error = 0.0 #存储误差的变量
for i in range(1, len(t_list), 1):
    #print(t_list[i], file_list[i-1])
    path = os.path.join(file_folder, file_list[i-1])
    data = np.loadtxt(path)
    analytical_sol = fft.ifft2(fft.fft2(u0)*np.exp(-(kx_grid**2+ky_grid**2)*t_list[i])).real
    error = error + np.max(abs(data-analytical_sol))
print('the error under time step', delta_t, 'is', error)
