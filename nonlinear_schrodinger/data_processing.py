import numpy as np
import matplotlib.pyplot as plt
import parameters as p
from mpl_toolkits.mplot3d import Axes3D

file_path = p.path2
data = np.loadtxt(file_path).reshape(-1, p.x_nodenum)
t = np.arange(0, p.total_time, p.delta_t*p.steps_per_saved)
x = p.x
X,T = np.meshgrid(x, t)
print(T.shape, X.shape, data.shape)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection ='3d')
ax.plot_surface(T, X, data, cmap = 'rainbow')
#ax.plot_surface(T, X, abs((np.exp(p.j*T/2)/np.cosh(X))), cmap = 'rainbow')
plt.title('IMEX numerical result')
plt.xlabel('t axis')
plt.ylabel('x axis')
#ax.view_init(elev = 60, azim = 0)
#plt.savefig('./output/nschrodinger_forward_euler.png', dpi = 300)
plt.savefig('./output/nschrodinger_imex.png', dpi = 300)
#plt.savefig('./output/nschrodinger_analytical.png', dpi = 300)
#plt.show()