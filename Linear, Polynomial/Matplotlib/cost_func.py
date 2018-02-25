import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = np.arange(60, 101, 40/20)
ys = np.arange(-6, 0,6/20)

X, Y = np.meshgrid(xs, ys)

zs = []

for i in range(20):
	z = []
	for j in range(21):
		val = xs[j] +ys[i]*feats
		cost = ((val-out)**2)/(2*feats.size)
		z.append(np.sum(cost))
	zs.append(z)

zs = np.array(zs)
print(X.shape)
print(Y.shape)
print(zs.shape)

ax.plot_surface(X, Y, zs, cmap=cm.coolwarm)

plt.xlabel(r'Parameter $\theta_0$', fontdict=font1)
plt.ylabel(r'Parameter $\theta_1$', fontdict=font1)
plt.title(r'Average Cost: $C(\theta)$', fontdict=font)

plt.show()