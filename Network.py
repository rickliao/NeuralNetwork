import numpy as np
from random import random, uniform

# class Node:
#     # weights is a list of numbers
#     def __init__(self, weights, theta, activationFunction):
#         self.weights = weights
#         self.theta = theta
#         self.activationFunction = activationFunction
#
#     # Each input must correspond to each weight in order
#     def calcOutput(self, input):
#         mul = [i * w for i,w in zip(input, self.weights)]
#         self.output =  self.activationFunction(mul + self.theta)

class Layer:
    def __init__(self):
        self.prevLayer = None
        self.nextLayer = None
        self.numNode = 0
        self.weights = [] # 2d list
        self.theta = []
        self.activationFunction = []
        self.output = []
        self.delta = []

    def addRandomSigmoidNode(self, isInputNode=False):
        if not isInputNode:
            weightForNode = []
            for i in range(self.prevLayer.numNode):
                weightForNode.append(uniform(-0.5, 0.5))
            self.weights.append(weightForNode)
            self.theta.append(random())
            self.activationFunction.append(sigmoid)
        self.numNode = self.numNode + 1

    def addNode(self, weights, theta, activationFunction):
        self.weights.append(weights)
        self.theta.append(theta)
        self.activationFunction.append(activationFunction)
        self.numNode = self.numNode + 1

    def calcOutputs(self):
        if self.prevLayer is not None:
            for i in range(len(self.weights)):
                out = self.activationFunction[i](np.dot(self.prevLayer.output, self.weights[i]) + self.theta[i])
                self.output.append(out)

    # target is a list with same length as output; only needed for output layer
    def calcDeltasAndAdjust(self, target=None):
        eta = 0.1
        if self.nextLayer is None:
            for i in range(len(self.output)):
                self.delta.append(self.output[i] * (1 - self.output[i]) * (self.output[i] - target[i]))
                # adjust weight and bias
                for j in range(len(self.weights[i])):
                    self.weights[i][j] = self.weights[i][j] - eta * self.delta[-1] * self.prevLayer.output[j]
                    self.theta[i] = self.theta[i] - eta * self.delta[-1]
        else:
            for i in range(len(self.output)):
                sum = 0
                for j in range(len(self.nextLayer.delta)):
                    sum = sum + self.nextLayer.delta[j] * self.nextLayer.weights[j][i]
                out = self.output[i] * (1 - self.output[i]) * sum
                self.delta.append(out)
                # adjust weight and bias
                for j in range(len(self.weights[i])):
                    self.weights[i][j] = self.weights[i][j] - eta * self.delta[-1] * self.prevLayer.output[j]
                    self.theta[i] = self.theta[i] - eta * self.delta[-1]

    def flushOutputAndDelta(self):
        self.output.clear()
        self.delta.clear()

class Network:
    def __init__(self):
        self.layers = []

    def addLayer(self):
        newLayer = Layer()
        if len(self.layers) == 0:
            self.layers.append(newLayer)
        else :
            newLayer.prevLayer = self.layers[-1]
            self.layers[-1].nextLayer = newLayer
            self.layers.append(newLayer)

    # input is a list with same length as number of node in first layer
    # target is a list with same length as number of node in last layer
    def runOneIteration(self, input, target):
        self.layers[0].output = input
        for i in range(1, len(self.layers)):
            self.layers[i].flushOutputAndDelta()
            self.layers[i].calcOutputs()
        for i in range(len(self.layers)-1, -1, -1):
            if i == len(self.layers)-1:
                self.layers[i].calcDeltasAndAdjust(target)
            elif i != 0:
                self.layers[i].calcDeltasAndAdjust()

    def calcOutput(self, input):
        self.layers[0].output = input
        for i in range(1, len(self.layers)):
            self.layers[i].flushOutputAndDelta()
            self.layers[i].calcOutputs()
        return self.layers[-1].output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))