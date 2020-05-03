import numpy as np
import matplotlib.pyplot as plt

with open("Data1_1.txt") as data:
	data = np.loadtxt(data)

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(data)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(data)
plt.yscale('log')
plt.title('Log')
plt.grid(True)

plt.show()