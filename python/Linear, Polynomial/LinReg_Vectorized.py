import numpy as np
import matplotlib.pyplot as plt
import random

class LinReg:
	def __init__(self, features, output, learning_factor, growth_limit, feature_max=None, order=1):
		feat = []
		if not feature_max is None:
			for i in range(len(features)):
				feat.append([a/features_max[i] for a in features[i]])
		else:
			feat = list(features)
		for i in range(2,order+1):
			for j in range(len(features)):
				feat.append([a**i for a in feat[j]])
		self.feats = np.matrix([[1 for i in output]] + feat)

		self.out = np.matrix(output)
		self.learn_fact = learning_factor**order
		self.limit = growth_limit
		self.param = np.asmatrix(np.zeros(len(self.feats)))
		self.accuracy = 0
		self.iterations = 0

		count = 0
		met_lim = False
		while not met_lim:
			shift = self.learn_fact * self.derivlin()
			shift_v = abs(np.linalg.norm(shift))
			met_lim = met_lim or (shift_v < abs(self.limit))
			# met_lim = met_lim or (count >= clim)
			if count%1000 is 0:
				print("{0} and {1}".format(self.param, shift))
			if met_lim:
				print("Shift of {0} passed limit of {1} at: {2}".format(shift_v, abs(self.limit), count))
				self.accuracy = shift_v
				self.iterations = count
			else:
				count += 1
				try:
					self.param = self.param - shift
				except:
					print("{0} - {1}".format(self.param, shift))

	def derivlin(self):
		h = self.param * self.feats
		val = (h - out)*self.feats.getT()
		return val/self.out.shape[1]

def generate_data(line, length, variance, start, stop):
	func = np.matrix(line).getT()
	x = []
	y = []
	for i in range(length):
		t = start + random.random() * (stop-start)
		dat = t + random.random() * variance
		y.append((np.matrix([1,dat]) * func)[0,0])
		x.append(t)
	return [x],y

def graph_data(inp, outp, func, start, stop):
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
		'size': 12
	}

	length = len(inp[0])
	x = np.arange(start, stop+1)
	y = np.matrix(np.vstack((np.ones(stop-start+1), x))).getT() * func

	plt.figure(figsize=(8.0,6.0))

	line = plt.plot(x, y)
	plt.setp(line, linewidth=1)
	plt.text(5, 80, '$y={1:2f}x + {0:2f}$'.format(func[0,0], func[1,0]), fontdict=font1, 
			bbox = {'facecolor': 'yellow', 'alpha': 0.5, 'pad': 5})
	# plt.text(-1.5, 50, 'Iterations: {0}\nShift: {1}'.format(iterations, accuracy), fontdict=font1,
			# bbox = {'facecolor': 'pink', 'alpha': 0.5, 'pad': 5})
	plt.scatter(inp, outp, s=20)

	plt.xlabel('Time (Minutes)', fontdict=font)
	plt.ylabel('Temperature (Celcius)', fontdict=font)
	plt.title('Temperature of a Potato\n(data is not real)', fontdict=font)
	plt.grid(linestyle='dotted')

	plt.show()
	plt.savefig('fig.png')

# feats = [[0,1,2,3]
		 # ]
# out = [1,4,7,10]
line_func = [90,-3]
# feats, out = generate_data(line_func, 40, 5, 0, 10)
# print(feats)
# print(out)

feats = [
	[2.6, 8.0, 2.9, 1.3, 3.5, 5.7, 8.4, 6.3, 6.6, 6.9, 5.6, 9.9, 1.5, 2.3, 
	8.0, 7.6, 5.1, 1.03, 0.61, 7.7, 8.2, 9.9, 6.6, 1.2, 7.4, 8.1, 4.6, 2.2, 5.6, 
	4.1, 6.05, 4.76, 8.9, 0.93, 8.6, 4.05, 9.02, 9.7, 6.9, 4.8]
]
out = [
	70.7, 61.5, 77.5, 78.2, 66.1, 60.6, 58.9, 61.9, 55.7, 61.1, 59.0, 53.4, 
	71.6, 78.2, 59.7, 53.3, 61.3, 80.5, 86.8, 66.4, 55.7, 59.3, 57.0, 75.5, 
	65.4, 50.8, 75.2, 71.1, 66.8, 76.5, 66.1, 64.3, 62.6, 83.2, 49.1, 74.6, 
	54.0, 51.1, 61.1, 69.5]

learn_fact = 0.01
limit = 1e-10
clim = 10000

line = LinReg(feats, out, learn_fact, limit, order=1)
print(*["\n"+str(i) for i in line.param])

accuracy = line.accuracy
iterations = line.iterations

graph_data(feats, out, line.param.getT(), 0, 10)