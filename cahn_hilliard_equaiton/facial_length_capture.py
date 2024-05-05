import numpy as np
import matplotlib.pyplot as plt
import os
import re

def capture_face(u):
    flag = 0 #用来记录在界面之中的点数
    row = u.shape[0]
    column = u.shape[1]
    for i in range(row):
        for j in range(column):
            if(u[i][j]<0.8 and u[i][j]>0.7):
                flag = flag+1
    return flag

dt_path = './output1/dt_adap.txt'
time_data = np.loadtxt(dt_path)
u_file_folder = './output1/u_result/'
u_files = os.listdir(u_file_folder)
u_files = sorted(u_files, key=lambda x: int(re.search(r'\d+', x).group())) #数据跟去前面的迭代次数排序

#计算界面长度随时间的变化
facial_num = np.zeros((1,len(time_data)))
facial_num = np.squeeze(facial_num)
for i in range(len(time_data)):
    path = os.path.join(u_file_folder, u_files[i])
    u_data = np.loadtxt(path)
    facial_num[i] = capture_face(u_data)
plt.plot(time_data, facial_num, '-', color = 'black')
plt.title('Interface Length changing with time')
plt.xlabel('t')
plt.ylabel('Facial Length')
x_value = 0.00392454
plt.axvline(x = x_value, color = 'blue', linestyle = '--')
plt.text(x_value+0.02, 0, 'x = {}'.format(x_value), rotation=0, verticalalignment = 'bottom')
plt.savefig('./output1/fig/facial_length.png', dpi = 300)