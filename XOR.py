import Network

if __name__ == "__main__":
    network = Network.Network()
    network.addLayer()
    network.addLayer()
    network.addLayer()

    network.layers[0].addRandomSigmoidNode(True)
    network.layers[0].addRandomSigmoidNode(True)
    network.layers[1].addRandomSigmoidNode()
    network.layers[1].addRandomSigmoidNode()
    network.layers[1].addRandomSigmoidNode()
    network.layers[1].addRandomSigmoidNode()
    network.layers[2].addRandomSigmoidNode()

    input = [[0,0], [0,1], [1,0], [1,1]]
    target = [[0], [1], [1], [0]]

    for c in range(10000):
        for i in range(len(input)):
            network.runOneIteration(input[i], target[i])

    print(network.calcOutput(input[0]))
    print(network.calcOutput(input[1]))
    print(network.calcOutput(input[2]))
    print(network.calcOutput(input[3]))

    print("Weights: ", network.layers[1].weights)
    print("Theta: ", network.layers[1].theta)
    print("Output: ", network.layers[1].output)

    print("Weights: ", network.layers[-1].weights)
    print("Theta: ", network.layers[-1].theta)
    print("Output: ", network.layers[-1].output)