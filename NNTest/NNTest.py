import pickle
import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

with open("train.pcl", 'rb') as f:
    data = pickle.load(f)
# X = data[['price', 'item_seq_number', 'image_top_1']]
# выходные данные
# y = data[['deal_probability']]
data = data.loc[data['deal_probability'] > 0]
#data = data[0:10]
X = data[['price', 'item_seq_number', 'image_top_1']]
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
# выходные данные
y = data[['deal_probability']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Each row is a training example, each column is a feature  [X1, X2, X3]
# X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
def sigmoid(x):
    sigm = 1 / (1 + np.exp(-x))
    return sigm
def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:

    def __init__(self, X):
        self.input      = X
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)                 
        #self.y          = y
        #self.output     = np.zeros(self.y.shape)
        self.output = np.zeros(len(X))

    def changeX(self, X):
        self.input = X

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self,y):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, y):
        self.output = self.feedforward()
        self.backprop(y)

    def predict(self):
        out = self.feedforward()
        return out



NN = NeuralNetwork(X_train)
for i in range(1200):  # trains the NN 1,000 times
    NN.train(y_train)


#print("Input : \n" + str(X_train))
print("Actual Output: \n" + str(y_train))
print("Predicted Output: \n" + str(NN.feedforward()))
print("Loss: \n" + str(np.mean(np.square(y_train - NN.feedforward()))))  # mean sum squared loss
print("Total error: {}".format(np.sum(np.abs(y_train - NN.feedforward()))))
print("\n")
NN.changeX(X_test)
print("Actual Output: \n" + str(y_test))
print("Predicted Output: \n" + str(NN.feedforward()))
print("Loss: \n" + str(np.mean(np.square(y_test - NN.feedforward()))))  # mean sum squared loss
print("Total error: {}".format(np.sum(np.abs(y_test - NN.feedforward()))))
kek = 5

