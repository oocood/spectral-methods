import matplotlib.pyplot as plt
import numpy as np
import parameters as p

x_nodenum = p.x_nodenum
x = p.x
path = p.path
data = np.loadtxt(path).reshape(-1, x_nodenum)
slices = data.shape[0] #将行数调出来以便于绘制变化的曲线
plt.imshow(data,cmap='viridis')
plt.colorbar()
plt.title('color map of Allen-Cahn equation')
plt.xlabel('x axis')
plt.ylabel('t axis')
plt.savefig('./output/imshow'+str(p.total_time)+'.png', dpi = 300)
plt.close()
'''for i in range(slices):
    if(i%10==0):
        plt.plot(x, data[i,:], '-')
plt.title('concentration curve change with time')
plt.xlabel('x axis')
plt.ylabel('u value')
plt.savefig('./output/figure'+str(p.total_time)+'.png', dpi = 300)
'''