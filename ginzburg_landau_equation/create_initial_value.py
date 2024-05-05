import numpy as np

x_nodenum = 256
u0 = np.random.uniform(-0.1, 0.1, size = (1, x_nodenum))
u0 = np.squeeze(u0)
save_path = './output/initial_value.txt'
with open(save_path, 'w') as f:
    np.savetxt(f, u0, fmt = '%f')
f.close()
print('create initial value successfully')