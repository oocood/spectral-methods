import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import parameters as p
import os
from mpl_toolkits.mplot3d import Axes3D

x_nodenum = p.x_nodenum
y_nodenum = p.y_nodenum
#file_folder = p.file_folder
file_folder = './output/forward_euler/'
kx_grid = p.kx_grid
ky_grid = p.ky_grid
u0 = p.u0
#t_list = np.arange(0, p.total_time, p.delta_t*p.steps_per_saved)
file_list = os.listdir(file_folder)
time_cap = '0.05'
fig_name = time_cap + '.txt'
path = os.path.join(file_folder, fig_name)
data = np.loadtxt(path).reshape(-1, y_nodenum) #数值解
time_slice = 0.05
analytical_sol = fft.ifft2(fft.fft2(u0)*np.exp(-(kx_grid**2+ky_grid**2)*time_slice)).real #解析解
print(data.shape)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
#ax.plot_surface(p.X, p.Y, data, cmap='rainbow')
#ax.plot_surface(p.X, p.Y, analytical_sol, cmap= 'rainbow')
ax.plot_surface(p.X, p.Y, abs(analytical_sol-data), cmap = 'rainbow')
#plt.title('Temperature distribution')
plt.title('Error distribution')
plt.title
plt.xlabel('x axis')
plt.ylabel('y axis')
#plt.savefig('./output/numerical_png/'+time_cap+'_forward_euler.png', dpi = 300)
#plt.savefig('./output/analytical_png/'+str(time_slice)+'.png', dpi = 300)
plt.savefig('./output/error_png/'+str(time_slice)+'_forward_euler.png', dpi = 300)