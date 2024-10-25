import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as spc
import pandas as pd

# Funktion für den Fit
def fit_f(x, a, mu, sigma, b):
    return a * np.exp (-1/2 * ((x-mu)/sigma)**2)+b

# Daten einlesen
df = pd.read_excel('Test.xlsx', names=['xdata', 'ydata'])

# x- und y-Werte
xpoints = df['xdata'].to_numpy()
ypoints = df['ydata'].to_numpy()


p0 = [5e-9, 18.226, 0.2, 1e-10]

# Fit mit curve_fit
params, covariance = curve_fit(fit_f, xpoints, ypoints, p0=p0)

errors = np.sqrt(np.diag(covariance))

yerrors = np.array([1e-10]* len(xpoints))

# Plotten des Fits
xaxis = np.linspace(min(xpoints), max(xpoints), 1000)
plt.plot(xaxis, fit_f(xaxis, *params), color='red')
plt.errorbar(xpoints, ypoints, yerr= yerrors, marker = '.', linestyle = 'none', label = 'Datenpunkte')
plt.scatter(xpoints, ypoints, label='Daten', s=2, color = 'red')	
plt.title('Chromatogramm')
plt.xlabel('q/z')
plt.ylabel('Partialdruck')
plt.legend()
plt.show()
plt.imshow(covariance, cmap='viridis', interpolation='none', aspect='auto')

# Anpassung der Plot-Achsen
plt.colorbar(label='Covariance Value')
plt.title('Covariance Matrix Heatmap')
plt.xticks(range(len(params)), ['a', 'b', 'c', 'd'])
plt.yticks(range(len(params)), ['a', 'b', 'c', 'd'])
plt.show()

#Berechnung des X^2
chi_sqrd = np.sum(np.square(ypoints-fit_f(xpoints, *params))/np.square(yerrors))
print(chi_sqrd)
red_chi_sqrd = chi_sqrd/(len(xpoints) - len(params))
print(red_chi_sqrd)