import numpy as np
import math
from time import time

class NeuralNetwork:
    '''Represents a Neural Network

    Attributes:
        b (list of numpy.matrix):           List of plus values for each calculation in the network
        weights (list of numpy.matrix):     List of weights for each calculation in the network
        feats (numpy.matrix):               Shuffled matrix of given input
        out (numpy.matrix):                 Suhffled matrix of given output
        alpha (float):                      Learning rate of the algorithm
        limit (float):                      Growth limit of the algorithm
        lmbda (float):                      Regularization constant of the algorithm
    '''
    def __init__(self, features, output, network, learning_rate, limit=1e-4, regularize=0, reg_type=0):
        '''Creates a NeuralNetwork object
        
        Args:
            features (list of list of num):     Input to the algorithm, structured (rxc) as (fetures x training data)
            output (list of list of num):       Output corresponding to given input, structured (rxc) as (outputs x training data)
            network (list of int):              List of integers representing each layer and the number of nodes in each layer
                                                    - Must have at least 2 layers for input and output
            learning_rate (int):                Learning rate of the algorithm

        Kwargs:
            limit (float):          Growth limit of the shift of the cost function
            regularize (float):     Regularization constant, lambda
            reg_type (int):         Type of regression to compute (stochastic is recommended): 
                                        - 0 for stochastic
                                        - 1 for batch
        '''
        if len(features) != network[0] or len(network)<2:
            print("Bad network: {}. Too short, or features do not match".format(network))
            return
        self.b = []
        self.weights = []
        for i in range(1, len(network)):
            self.weights.append(np.asmatrix((0.01/math.sqrt(network[i-1]))*np.random.randn(network[i-1],network[i])))
            self.b.append(np.asmatrix(np.zeros(shape=(1,network[i]))))
        self.feats, self.out = self.compileData(features, output)
        self.alpha = learning_rate
        self.limit = limit
        self.lmbda = regularize

        if reg_type:
            self.grad_batch()
        else:
            self.grad_stochastic()

    def eval(self, input: list):
        '''Computes the result of the network with the given input

        Args:
            input (list of num):    Input to evaluate
                                        - Must have same number of features as internal algorithm
                                        - Evaluates a single set of data, therefore 1-d

        Returns:
            List of numpy.matrix representing the network with given input
        '''
        if len(input) != self.feats.shape[1]:
            print("Input size of {0} does not match overall feature size {1}".format(len(input), self.feats.shape[1]))
            return
        return self.calcNetwork(np.matrix(input))

    def update(self, inp: list, out: list):
        '''Updates the network with new inputs and outputs

        Recalculates network after updating input

        Args:
            inp (list of num or list of list of num):       New input values
                                                                - Must have same number of features as internal algorithm
                                                                - Can have varying size of training data
            out (list of num or list of list of num):       New output value
                                                                - Must have same number of outputs as internal algorithm
                                                                - Can have varying size of training data
        '''
        i = np.matrix(inp).getT()
        o = np.matrix(out).getT()
        invalid = False
        if i.shape[1] != self.feats.shape[1]:
            print("Input size of {0} does not match overall feature size {1}".format(i.shape[1], self.feats.shape[1]))
            invalid = True
        if o.shape[1] != self.out.shape[1]:
            print("Output size of {0} does not match overall output size {1}".format(o.shape[1], self.out.shape[1]))
            invalid = True
        if o.shape[0] != i.shape[0]:
            print("Additional traning set sizes do not match for (input, output): ({0}, {1})".format(i.shape[0], o.shape[0]))
            invalid = True
        if invalid:
            return
        self.feats = np.concatenate(i, self.feats)
        self.out = np.concatenate(o, self.out)
        self.grad_stochastic()

    def compileData(self, feats, outs):
        '''Shuffles features of outputs in the same order

        Args:
            feats (list of list of num):        Features to shuffle
            outs (list of list of num):         Outputs to shuffle

        Returns:
            Tuple of shuffled input and output values: (input, output)
        '''
        x = np.matrix(feats).getT()
        y = np.matrix(outs).getT()
        if x.shape[0] != y.shape[0]:
            print("Mismatched training set size for (input, output): ({0}, {1})".format(x.shape[0], y.shape[0]))
        data = np.concatenate((x, y), axis=1)
        np.random.shuffle(data)
        return (data[:,0:x.shape[1]],data[:,data.shape[1]-y.shape[1]:data.shape[1]])

    def grad_batch(self):
        '''Calculates network with Batch Gradient Descent
        Prints updates to shift over time
        Prints resultant weights and plus values
        '''
        t0 = time()
        count = 0
        met_lim = False
        while not met_lim:
            count += 1
            network, derivs = self.backProp(self.feats, self.out)
            avg = 0
            for i in range(len(self.b)):
                self.weights[i] = (self.weights[i]) - self.alpha * (network[i].getT()*derivs[i+1] / self.feats.shape[0] + (self.lmbda*self.weights if self.lmbda!=0 else 0))
                self.b[i] = self.b[i] - self.alpha * np.sum(derivs[i+1], axis=0) / self.feats.shape[0]
                avg += np.linalg.norm(np.sum(derivs[i+1], axis=0) / self.feats.shape[0])
            shift = avg/len(self.b)
            if count % 10000 == 0:
                print("Iteration {0} with shift {1}".format(count, shift))
            met_lim = met_lim or shift < self.limit
        t1 = time()
        print("Shift of {0} passed limit {1} at count: {2}, with a time of: {5} seconds\n\n\tWeights:{3}\n\tPlus Values:{4}".format(shift,self.limit,count,self.weights,self.b, t1-t0))

    def grad_stochastic(self):
        '''Calculates network with Stochastic Gradient Descent
        Prints updates to shift over time
        Prints resultant weight and plus values
        '''
        t0 = time()
        count = 0
        met_lim = False
        while not met_lim:
            j = count % self.feats.shape[0]
            count += 1
            network,derivs = self.backProp(self.feats[j,:], self.out[j,:])
            avg = 0
            for i in range(len(self.b)):
                self.weights[i] = self.weights[i] - self.alpha * (network[i].getT()*derivs[i+1] + (self.lmbda*self.weights if self.lmbda!=0 else 0))
                self.b[i] = self.b[i] - self.alpha * derivs[i+1]
                avg += np.linalg.norm(derivs[i+1])
            shift = avg/len(self.b)
            if count/self.feats.shape[0] % 10000 == 0:
                print("Iteration {0} with shift {1}".format(count//self.feats.shape[0], shift))
            met_lim = met_lim or shift < self.limit
        t1 = time()
        print("Shift of {0} passed limit {1} at count: {2}, and iteration: {5}, with a time of: {6} seconds\n\n\tWeights:{3}\n\tPlus Values:{4}".format(shift,self.limit,count,self.weights,self.b,count//self.feats.shape[0], t1-t0))

    def grad_newtons(self):
        '''Calculates network with Newton's method applied to gradient descent
        Prints updates to shift over time
        Prints resultant weight and plus values
        '''
        t0 = time()
        count = 0
        met_lim = False
        while not met_lim:
            count += 1
            network, derivs = self.backProp(self.feats, self.out)
            avg = 0
            for i in range(len(self.b)):
                dw = (network[i].getT()*derivs[i+1] / self.feats.shape[0] + (self.lmbda*self.weights if self.lmbda!=0 else 0))
                db = np.sum(derivs[i+1], axis=0) / self.feats.shape[0]
                
                self.weights[i] = (self.weights[i]) - self.alpha * 
                self.b[i] = self.b[i] - self.alpha * np.sum(derivs[i+1], axis=0) / self.feats.shape[0]
                avg += np.linalg.norm(np.sum(derivs[i+1], axis=0) / self.feats.shape[0])
            shift = avg/len(self.b)
            if count % 10000 == 0:
                print("Iteration {0} with shift {1}".format(count, shift))
            met_lim = met_lim or shift < self.limit
        t1 = time()
        print("Shift of {0} passed limit {1} at count: {2}, with a time of: {5} seconds\n\n\tWeights:{3}\n\tPlus Values:{4}".format(shift,self.limit,count,self.weights,self.b, t1-t0))

    def backProp(self, input, out):
        '''Computes some steps of the Backpropogation algorithm

        Forward passes to calculate the network, and backpropogates to
        compute derivative values for every node

        Args:
            input (numpy.matrix):       Matrix of input values
            out (numpy.matrix):         Matrix of output values

        Returns:
            Tuple of network and derivatives for each node: (network, derivatives)
        '''
        network = self.calcNetwork(input)
        derivs = self.networkDeriv(network, out)
        return (network,derivs)

    def calcNetwork(self, input):
        '''Calculates the network given an input

        Calculation is based of the weights and plus values

        Args:
            input (numpy.matrix):   Matrix of input values

        Returns:
            The network as a list of matrices
        '''
        nodes = [input]
        for i in range(len(self.b)):
            nodes.append(self.sigmoid(nodes[len(nodes)-1],self.weights[i],self.b[i]))
        return nodes

    def networkDeriv(self, network, out):
        '''Calculates derivatives of the nodes of a network

        Args:
            network (list of numpy.matrix):     Network created from forward pass
            out (numpy.matrix)                  Output values of training set

        Return:
            A list of matrices representing the derivative of every node
        '''
        init_d = np.multiply(-(out-network[len(network)-1]), self.deriv_s(network, len(network)-1))
        derivs = [init_d]
        for i in range(len(network)-2, -1, -1):
            derivs.append(np.multiply(derivs[len(derivs)-1] * self.weights[i].getT(), self.deriv_s(network, i)))
        return list(reversed(derivs)) 

    def sigmoid(self, x,w,b):
        '''Computes a sigmoid function (logistic)

        Returns:
            Result of sigmoid operation
        '''
        return (1/(1+np.exp(-(x*w + b))))

    def deriv_s(self, network, index):
        '''Computes derivative of sigmoid function

        Returns:
            Result of derivative
        '''
        return np.multiply(network[index], 1-network[index])

feats = [[50,50,10,0,50,0,0,50],
         [50,20,10,0,50,0,0,50],
         [50,20,50,0,0,0,0,50],
         [0,10,10,0,0,0,0,0],
         [0,10,10,0,50,0,0,0],
         [0,20,0,50,0,0,0,0],
         [0,0,20,50,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,10,10,160,45,0],
         [10,10,20,0,0,0,0,10],
         [0,10,30,50,0,0,0,0]
         ]
out = [[300,400,250,550,-250,-2100,-185,-225]]
rate = 0.2
limit = 1e-4

reg = NeuralNetwork(feats, out, [3,4], rate, limit, reg_type=0)
