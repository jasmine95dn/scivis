import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

xx, yy = np.meshgrid(x, y, sparse=True)

# function f(x,y)
z = np.sin(6*np.cos(np.sqrt(xx**2+yy**2)) + 5*np.arctan2(yy,xx))

# plot
fig = plt.contourf(x,y,z, cmap=plt.get_cmap('gnuplot2') )

plt.colorbar(fig)
plt.title('Problem 6')

plt.savefig('problem6.pdf')

plt.show()