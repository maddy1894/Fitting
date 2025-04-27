import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Original data with flipped x and y
data = np.loadtxt('data.txt', delimiter= "\t", skiprows=1)
y = data[:,0]
x = data[:,1]

# Define sigmoid function
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k*(x - x0))) + b

# Initial guess
p0 = [max(y)-min(y), np.median(x), 0.01, min(y)]

# Set bounds to force smoothness (small k)
bounds = ([0, min(x), 0.001, 0], [1000, max(x)*2, 0.1, 100])

# Fit the curve
popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds)

# Generate smooth x for plotting
x_fit = np.linspace(min(x), max(x), 500)
y_fit = sigmoid(x_fit, *popt)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_fit, y_fit, '-', color='red', label='Smooth Sigmoid Fit')
plt.title('Smooth Sigmoid Fit to Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
