import pickle
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
with open("train.pcl", 'rb') as f:
    data = pickle.load(f)
# X = data[['price', 'item_seq_number', 'image_top_1']]
# выходные данные
# y = data[['deal_probability']]
new_data = data.loc[data['deal_probability'] > 0]
new_data = new_data[0:4]
X = new_data[['price', 'item_seq_number', 'image_top_1']]
# выходные данные
y = new_data[['deal_probability']]

# # Each row is a training example, each column is a feature  [X1, X2, X3]
# X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
def sigmoid(x):
    sigm = 1 / (1 + np.exp(-x))
    return sigm
def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:

    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("Total error: {}".format(np.sum(np.abs(y - NN.feedforward()))))
        print("\n")
    if i == 1000:
        p = NN.feedforward()
        kek = 5

    NN.train(X, y)
kek = 5

