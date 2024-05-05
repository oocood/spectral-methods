import numpy as np
import matplotlib.pyplot as plt
import os
import re
import parameters as p


facial_energy_path = './output1/free_energy/facial_energy.txt'
free_energy_path = './output1/free_energy/free_energy.txt'
dt_path = './output1/dt_adap.txt' #累计时间存储路径

facial_energy_data = np.loadtxt(facial_energy_path)
free_energy_data = np.loadtxt(free_energy_path)
time_data = np.loadtxt(dt_path)

print(facial_energy_data.shape, free_energy_data.shape, time_data.shape)

plt.plot(time_data[1:-1], facial_energy_data[1:-1], '-', color = 'black')
plt.title('Facial energy changing with time')
plt.xlabel('t')
plt.ylabel('Facial energy')
x_value = 0.00465352
plt.axvline(x = x_value, color = 'blue', linestyle = '--')
plt.text(x_value+0.02, 0, 'x = {}'.format(x_value), rotation=0, verticalalignment = 'bottom')
plt.savefig('./output1/fig/facial_energy.png', dpi = 300)
plt.close()
plt.plot(time_data[1:-1], free_energy_data[1:-1], '-', color = 'black')
plt.title('Free energy changing with time')
plt.xlabel('t')
plt.ylabel('Free energy')
plt.savefig('./output1/fig/free_energy.png', dpi = 300)

'''u_result_path = './output/u_result'
u_files = os.listdir(u_result_path)
u_files = sorted(u_files, key=lambda x: int(re.search(r'\d+', x).group()))
print(u_files)
files_num = len(u_files)
for i in range(files_num):
    if(i%10==0):
        path = os.path.join(u_result_path, u_files[i])
        u_data = np.loadtxt(path)
        plt.imshow(u_data)
        plt.title('t='+str(round(time_data[i],5))+',concentration distribution')
        plt.savefig('./output/fig/check_fig'+str(int(i/10))+'.png', dpi=300)
        plt.close()

for i in range(files_num):
    if(i%10==0):
        path = os.path.join(u_result_path, u_files[i])
        u_data = np.loadtxt(path)
        plt.plot(range(p.x_nodenum), np.squeeze(u_data[0,:]), '-', color = 'black')
        plt.title('t='+str(round(time_data[i],5))+',concentration distribution')
        plt.savefig('./output/fig/check_fig_cross'+str(int(i/10))+'.png', dpi=300)
        plt.close()
'''