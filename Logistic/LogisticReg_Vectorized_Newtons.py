import numpy as np

class LogisticReg:
    def __init__(self, features, output, growth_limit = 1e-15, feature_max=None, order=1, regularizeC=0):
        feat = []
        if not feature_max is None:
            for i in range(len(features)):
                feat.append([a/feature_max[i] for a in features[i]])
        else:
            feat = list(features)
        for i in range(2,order+1):
            for j in range(len(features)):
                feat.append([a**i for a in feat[j]])
        self.feats = np.matrix([[1 for i in output]] + feat).getT()

        self.out = np.matrix(output)
        self.limit = growth_limit
        self.lmbda = regularizeC
        self.param = np.asmatrix(np.zeros(len(self.feats.getT())))

        count = 0
        met_lim = False
        while not met_lim:
            count += 1
            shift = self.calcshift()
            try:
                self.param = self.param - shift
            except:
                print("{0} - {1}".format(self.param, shift))
            shift_v = abs(np.linalg.norm(shift))
            met_lim = met_lim or (shift_v < abs(self.limit))
            if count%100 is 0:
                print("{0} and {1}".format(self.param, shift))
            if met_lim:
                print("Shift of {0} passed limit of {1} at: {2}".format(shift_v, abs(self.limit), count))

    def calcshift(self):
        return (np.linalg.inv(self.hessian())*self.derivlin().getT()).getT()

    def hessian(self):
        return ((self.h()*(1-self.h()).getT())[0,0]*self.feats).getT()*self.feats/self.out.shape[1]

    def derivlin(self):
        val = (self.h() - self.out)*self.feats + (self.lmbda/self.out.shape[1])*self.param
        return val/self.out.shape[1]

    def h(self):
        return (1/(1+np.exp(-self.param * self.feats.getT())))

feats = [[2,4,8,16,20,25,30,32],
         ]
out = [0,0,0,1,0,1,1,1]
limit = 1e-15

reg = LogisticReg(feats, out, limit)
print(*["\n"+str(i) for i in reg.param])
