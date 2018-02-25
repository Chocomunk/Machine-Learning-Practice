import numpy as np

class LinReg:
    def __init__(self, features, output, order=1):
        feat = list(features)
        for i in range(2,order+1):
            for j in range(len(features)):
                feat.append([a**i for a in feat[j]])
        self.feats = np.matrix([[1 for i in output]] + feat)
        self.out = np.matrix(output)
        self.param = np.linalg.inv(self.feats * self.feats.getT()) * self.feats * self.out.getT()

#corn, beans, wheat, barley, oats, field hay, tobacco, field peas, cattle, sheep
# hogs,
feats = [[50,50,10,0],
         [50,20,10,0],
         [50,20,50,0],
         [0,10,10,0],
         [0,10,10,0],
         [0,20,0,50],
         [0,0,20,50],
         [0,0,0,0],
         [0,0,0,10],
         [10,10,20,0],
         [0,10,30,50]
         ]
out = [300,400,250,550]

line = LinReg(feats, out, order=1)
print(*["\n"+str(i) for i in line.param])
