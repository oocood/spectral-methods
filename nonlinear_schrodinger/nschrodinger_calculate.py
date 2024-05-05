import numpy as np
import math
import numpy.fft as fft
import time
import parameters as p

#从parameters调用计算参数
x = p.x
k = p.k
u1 = p.u1
u2 = p.u2
idt = p.idt
ik2dt = p.ik2dt
u_hat1 = p.u_hat1
u_hat2 = p.u_hat2

#主体计算
start_time = time.time()
with open(p.path1, 'w') as f1:
#with open(p.path1, 'w') as f1, open(p.path2, 'w') as f2:
    np.savetxt(f1, u1, fmt = '%f')
    #np.savetxt(f2, u2, fmt = '%f')
    for i in range(1, p.calculate_steps, 1):
        u_hat1 = u_hat1 - (ik2dt)/2*u_hat1 + idt*fft.fft(abs(u1)**2*u1)
        #u_hat2 = (u_hat2+idt*fft.fft(abs(u2)**2*u2))/(1+(ik2dt)/2)
        u1 = fft.ifft(u_hat1)
        #u2 = fft.ifft(u_hat2)
        if(i%p.steps_per_saved==0):
            np.savetxt(f1, abs(u1), fmt = '%f')
            #np.savetxt(f2, abs(u2), fmt = '%f')
end_time = time.time()
print('all the calculation has done','time cost is', end_time - start_time, ' s')