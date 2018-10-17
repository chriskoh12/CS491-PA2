import numpy as np
def perceptron_train(X,Y):
    convergence = False
    W = [[],[]] #create list for weights and biases
    for i in range (len(X[0])):
        W[0].append(0)
    W[1].append(0)
    print(W)
    while (convergence == False):
        convergence = True
        for i in range(len(X)):
            if ((calcActivation(W, X[i])) * Y[i][0]) < 1:  #if weights and biases need to be updated
                convergence = False
                for j in range(len(X[0])):
                    W[0][j] = W[0][j] + Y[i][0] * X[i][j]
                W[1][0] = W[1][0] + Y[i][0]
                print("New weights and biases for sample {} is {}" .format(i + 1, W))
    return W

def calcActivation(W, X): #calculates activation of a sample not including mult by label
    activation = 0
    for i in range(len(X)):
        activation += X[i] * W[0][i]
    activation += W[1][0]
    return activation




X = np.array([[0,0], [1,1],[0,1],[2,2],[1,0],[1,2]])
Y = np.array([[-1],[1],[-1],[1],[-1],[1]])
#X = np.array ([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
#Y = np.array ([[1], [1], [-1], [1], [-1], [-1], [-1]])
W = perceptron_train(X,Y)
print("Final weights and biases are ", W)
#test_acc = perceptron_test (X, Y, W[0], W[1])