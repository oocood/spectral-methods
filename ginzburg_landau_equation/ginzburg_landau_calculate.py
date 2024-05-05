import numpy as np
import numpy.fft as fft
import math
import time
import parameters as p
import time

x = p.x
k = p.k
u0 = p.u0
u = u0
u_hat = p.u_hat
calculation_steps = p.calculation_steps
steps_per_saved = p.steps_per_saved
path = p.path
lambda1 = p.lambda1
delta_t = p.delta_t

#常用中间变量
tmp = 1/(1+delta_t*lambda1**2*k**2)

start_time = time.time()
with open(path, 'w') as f:
    np.savetxt(f, u0, fmt = '%f')
    for i in range(1, calculation_steps, 1):
        u_hat = tmp*(u_hat-delta_t*fft.fft(u**3-u))
        u = fft.ifft(u_hat).real
        if(i%steps_per_saved==0):
            np.savetxt(f, u, fmt = '%f')
end_time = time.time()
print('all the calculation has done','time cost is',end_time-start_time, ' s')
