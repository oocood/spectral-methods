import numpy as np
import matplotlib.pyplot as plt
import parameters as p
from mpl_toolkits.mplot3d import Axes3D

file_path = p.path1
data = np.loadtxt(file_path).reshape(-1, p.x_nodenum)
t = np.arange(0, 1, p.delta_t*10000)
x = p.x
X,T = np.meshgrid(x, t)
print(T.shape, X.shape, data.shape)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection ='3d')
ax.plot_surface(T, X, data, cmap = 'rainbow')
#ax.plot_surface(T, X, 1/(np.cosh(X-5*T))**2, cmap = 'rainbow')
plt.title('IMEX numerical result')
plt.xlabel('t axis')
plt.ylabel('x axis')
ax.view_init(elev = 60, azim = 0)
plt.savefig('./output/kdv_imex.png', dpi = 300)
#plt.show()