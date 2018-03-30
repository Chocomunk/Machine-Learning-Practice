import numpy as np

class LogisticReg:
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

        count = 0
        met_lim = False
        while not met_lim:
            count += 1
            shift = self.learn_fact * self.derivlin()
            try:
                self.param = self.param - shift
            except:
                print("{0} - {1}".format(self.param, shift))
            shift_v = abs(np.linalg.norm(shift))
            met_lim = met_lim or (shift_v < abs(self.limit))
            # if count%100 is 0:
                # print("{0} and {1}".format(self.param, shift))
            if met_lim:
                print("Shift of {0} passed limit of {1} at: {2}".format(shift_v, abs(self.limit), count))

    def derivlin(self):
        h = 1/(1+np.exp(-self.param * self.feats))
        val = (h - out)*self.feats.getT()
        return val/len(self.out)

feats = [[0,3,6],
         ]
out = [-1,5,11]
learn_fact = 0.02
limit = 0.00000000000001

reg = LogisticReg(feats, out, learn_fact, limit, order=2)
print(*["\n"+str(i) for i in reg.param])
