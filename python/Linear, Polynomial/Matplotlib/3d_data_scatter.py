import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
from matplotlib import cm
import random

font = {
		'family': 'serif',
		'color': 'darkblue',
		'weight': 'normal',
		'size': 16
	}
font1 = {
		'family': 'serif',
		'color': 'darkblue',
		'weight': 'normal',
		'size': 13
	}

feats = np.array([
	2.6, 8.0, 2.9, 1.3, 3.5, 5.7, 8.4, 6.3, 6.6, 6.9, 5.6, 9.9, 1.5, 2.3, 
	8.0, 7.6, 5.1, 1.03, 0.61, 7.7, 8.2, 9.9, 6.6, 1.2, 7.4, 8.1, 4.6, 2.2, 5.6, 
	4.1, 6.05, 4.76, 8.9, 0.93, 8.6, 4.05, 9.02, 9.7, 6.9, 4.8
	])

out = np.array([
	70.7, 61.5, 77.5, 78.2, 66.1, 60.6, 58.9, 61.9, 55.7, 61.1, 59.0, 53.4, 
	71.6, 78.2, 59.7, 53.3, 61.3, 80.5, 86.8, 66.4, 55.7, 59.3, 57.0, 75.5, 
	65.4, 50.8, 75.2, 71.1, 66.8, 76.5, 66.1, 64.3, 62.6, 83.2, 49.1, 74.6, 
	54.0, 51.1, 61.1, 69.5
	])

box = [(0, 1, 46),
		(10.5, 1, 46),
		(10.5, 1, 90),
		(0, 1, 90)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

hexcol = lambda: random.randint(0,255)

for i in range(len(feats)):
	x = np.ones(10) * feats[i]
	z = np.ones(10) * out[i]
	y = np.arange(0,10)
	x, y = np.meshgrid(x,y)

	rand_col = '#%02X%02X%02X' % (hexcol(), hexcol(), hexcol())

	line = ax.plot(x,y,zs=z, color=rand_col)
	plt.setp(line, linewidth=1.0)
	# ax.scatter(x[0,0], y[1,0], z[0])

# plane = mp3d.art3d.Poly3DCollection([box], alpha=0.25, linewidth=1)
# plane.set_facecolor((0,0,1, 0.25))
# ax.add_collection3d(plane)

# ax.set_yticklabels([])

plt.xlabel('Time (Minutes)', fontdict=font1)
plt.ylabel(r'$x_0$', fontdict=font1)
ax.set_zlabel('Temperature (Celcius)', fontdict=font1)
plt.title('Temperature of a Potato\n(data is not real)', fontdict=font)

plt.show()