from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt', delimiter= "\t")
xdata = data[:,1]
ydata = data[:,0]
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    print(k)
    return (y)

p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')

x = np.linspace(-10, 350, 1000)
y = sigmoid(x, *popt)

plt.plot(xdata, ydata, 'o', label='data')
plt.plot(x,y, label='fit')
# plt.ylim(0, 1.3)
plt.legend(loc='best')
plt.show()