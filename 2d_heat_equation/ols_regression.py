import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import parameters as p

x_length = p.x_length
dt = [5*1e-6, 2.5*1e-6, 1.25*1e-6, 1e-6]
error = [0.002387140705, 0.001249015841, 0.0006816998779, 0.000569481784831]
#dt = [1e-3, 5e-4, 2.5e-4, 2e-4, 1.25e-4]
#error = [0.00671658438837, 0.0017587536037364, 0.0005273205531784, 0.0003791497039689895, 0.0002209195575371553]

#create linear regression model
dt = np.log10(dt).reshape(-1, 1)
error = np.log10(error)
model = LinearRegression()
model.fit(dt, error)

#get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

#create regression line
x_line = np.array([dt[0]+0.2,dt[-1]-0.2])
y_line = model.predict(x_line)

#scatter origin point
plt.scatter(dt, error, color = 'blue', label = 'data points')

#plot regression line
plt.plot(x_line, y_line, color = 'grey', label='regression line')

plt.xlabel('dx')
plt.ylabel('dt')
plt.title('dt vs dx in log-log scale')
plt.legend(loc = 'best')
plt.text(dt[1]-0.4, error[1],f'slope = {slope:.2f}')
plt.savefig('./output/error_png/error_convergence1.png', dpi = 300)
plt.show()
