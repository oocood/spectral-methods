import numpy as np
import parameters as p

data1 = np.loadtxt(p.path1).reshape(-1, p.x_nodenum)
data2 = np.loadtxt(p.path2).reshape(-1, p.x_nodenum)

x = p.x
t = np.arange(0, p.total_time, p.delta_t)
X,T = np.meshgrid(x, t)
analytical_sol = 1/(np.cosh(X-5*T))**2

#error1 = np.sum(abs(data1-analytical_sol))/np.sum(abs(analytical_sol))
#error2 = np.sum(abs(data2-analytical_sol))/np.sum(abs(analytical_sol))
error1 = np.linalg.norm(abs(data1-analytical_sol), 2)/np.linalg.norm(abs(analytical_sol), 2)
error2 = np.linalg.norm(abs(data2-analytical_sol), 2)/np.linalg.norm(abs(analytical_sol), 2)
print(error1, error2)