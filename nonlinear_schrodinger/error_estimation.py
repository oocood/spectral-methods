import numpy as np
import parameters as p
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data1 = np.loadtxt(p.path1).reshape(-1, p.x_nodenum)
data2 = np.loadtxt(p.path2).reshape(-1, p.x_nodenum)

x = p.x
t = np.arange(0, p.total_time, p.delta_t*p.steps_per_saved)
X,T = np.meshgrid(x, t)
analytical_sol = np.abs(np.exp(p.j*T/2)/np.cosh(X))

#error1 = np.sum(abs(data1-analytical_sol))/np.sum(abs(analytical_sol))
#error2 = np.sum(abs(data2-analytical_sol))/np.sum(abs(analytical_sol))
error1 = np.linalg.norm(abs(data1-analytical_sol), 2)/np.linalg.norm(abs(analytical_sol), 2)
error2 = np.linalg.norm(abs(data2-analytical_sol), 2)/np.linalg.norm(abs(analytical_sol), 2)
print(error1, error2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.plot_surface(T, X, np.abs(data1-analytical_sol), cmap = 'rainbow')
plt.title('error distribution')
plt.xlabel('t axis')
plt.ylabel('x axis')
plt.savefig('./output/error_distribution.png', dpi = 300)