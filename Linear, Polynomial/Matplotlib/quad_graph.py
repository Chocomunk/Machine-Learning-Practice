import matplotlib.pyplot as plt
import numpy as np

def func(inp):
	return np.sin(inp)/3 + ((inp-30)**2)/20

def deriv(inp):
	return np.cos(inp)/3 + (inp-30)/10

font = {
		'family': 'serif',
		'color': 'darkblue',
		'weight': 'normal',
		'size': 16
	}

x = np.arange(0,61)
y = func(x)

fig, ax = plt.subplots(1)
	
graph = ax.plot(x,y)
plt.setp(graph, linewidth=1.0)

val = 5
count = 0

while val>0 and val<61 and count < 20:
	slope = deriv(val)

	if count % 5 == 0:
		outp = func(val)

		x1 = np.arange(val-10, val+11	)
		y1 = outp + slope*(x1-val)

		line = ax.plot(x1,y1)
		plt.setp(line, linewidth=1.0)
		ax.scatter(val, outp)

	val -= slope
	count += 1

# plt.axis('off')
plt.grid(True)
ax.set_yticklabels([])
ax.set_xticklabels([])

plt.xlabel(r'Parameters $(\theta)$', fontdict=font)
plt.ylabel(r'Cost $C(\theta)$', fontdict=font)
plt.title('Least Squares Cost Function', fontdict=font)

plt.show()