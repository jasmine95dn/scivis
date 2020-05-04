from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('Data1_2.txt')

# find the index of the maximum
maxindex = np.argmax(data[:,2])
# coordinates of the maximum
max_coord = data[maxindex]

# plot the datas
ax = plt.axes(projection='3d')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
ax.scatter3D(data[:,0], data[:,1], data[:,2])
ax.set_title('Three-Dimensional Data', weight='bold')

# annotate the maximum
x,y,z = tuple(max_coord)
ax.text(x,y,z, f'maximum ({x},{y},{z})', c='r')

plt.savefig('problem5.pdf')

plt.show()

