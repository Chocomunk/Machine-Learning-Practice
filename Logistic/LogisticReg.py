import numpy as np

class LogisticReg:
    def __init__(self, features, output, learning_factor, growth_limit, features_max=None, order=1):
        self.feats = []
        if not features_max is None:
        	for i in range(len(features)):
        		self.feats.append([a/features_max[i] for a in features[i]])
        else:
            self.feats = self.feats + list(features)
        for i in range(2,order+1):
            for j in range(len(features)):
                self.feats.append([a**i for a in self.feats[j]])
        self.feats = [[1 for i in output]] + self.feats

        self.out = output
        self.learn_fact = learning_factor**order
        self.limit = growth_limit
        self.param = [0 for i in self.feats]

        count = 0
        met_lim = False
        while not met_lim:
            count += 1
            for j in range(len(self.param)):
                shift = learn_fact * self.derivlin(j)
                self.param[j] = self.param[j] - shift
                met_lim = met_lim or (abs(shift) < abs(self.limit))
                if count%10000 is 0:
                    print("{0} and {1}".format(self.param, shift))
                if met_lim:
                    print("Shift of {0} passed limit of {1} at: {2}".format(abs(shift), abs(self.limit), count))

    def derivlin(self, param_i):
        s = 0
        for i in range(len(self.out)):
            inp = self.get_feat_i(i)
            h = 1/(1+np.exp(-self.sumfunc(inp)))
            s += (h - self.out[i])*inp[param_i]/len(self.out)
        return s

    def sumfunc(self, inp):
        s = 0
        for i in range(len(self.param)):
            s += self.param[i]*inp[i]
        return s

    def get_feat_i(self, index):
        ls = []
        for f in self.feats:
            try:
                ls.append(f[index])
            except IndexError:
                print("PROBLEM WITH: {}, in {}".format(i, f))
        return ls



feats = [[2,4,8],
         ]
out = [3,9,27]
learn_fact = 0.001
limit = 0.0000000000000001

reg = LogisticReg(feats, out, learn_fact, limit, order=2)
print(*["\n"+str(i) for i in reg.param])
