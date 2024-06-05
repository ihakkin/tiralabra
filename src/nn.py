import numpy as np
import pandas as pd
import random

def relu(z):
    return np.maximum(z, 0)

def relu_prime(z):
    return z > 0

def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.random.randn(hidden_size, 1) * 0.1
        self.w2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.random.randn(output_size, 1) * 0.1

    def forward_propagation(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    def backward_propagation(self, x_batch, y_batch, a1, a2, z1, learning_rate, num_classes):
        m = x_batch.shape[1]
        one_hot_y = np.eye(num_classes)[y_batch].T

        delta2 = a2 - one_hot_y
        nabla_w2 = np.dot(delta2, a1.T) / m
        nabla_b2 = np.sum(delta2, axis=1, keepdims=True) / m

        delta1 = np.dot(self.w2.T, delta2) * relu_prime(z1)
        nabla_w1 = np.dot(delta1, x_batch.T) / m
        nabla_b1 = np.sum(delta1, axis=1, keepdims=True) / m

        self.w1 -= learning_rate * nabla_w1
        self.b1 -= learning_rate * nabla_b1
        self.w2 -= learning_rate * nabla_w2
        self.b2 -= learning_rate * nabla_b2

    def train(self, x_train, y_train, x_test, y_test, learning_rate, epochs, batch_size, num_classes):
        training_data = list(zip(x_train.T, y_train))
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                x_batch, y_batch = zip(*mini_batch)
                x_batch = np.array(x_batch).T
                y_batch = np.array(y_batch)
                z1, a1, z2, a2 = self.forward_propagation(x_batch)
                self.backward_propagation(x_batch, y_batch, a1, a2, z1, learning_rate, num_classes)
            test_accuracy = self.evaluate(x_test, y_test)
            print(f"Epoch {epoch + 1}: Test accuracy {test_accuracy:.4f}")

    def evaluate(self, x_test, y_test):
        _, _, _, a2 = self.forward_propagation(x_test)
        predictions = np.argmax(a2, axis=0)
        return np.mean(predictions == y_test)

    def save_parameters(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load_parameters(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']

def preprocess_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    x_train = (train_data.iloc[:, 1:].values / 255).T
    y_train = train_data.iloc[:, 0].values

    x_test = (test_data.iloc[:, 1:].values / 255).T
    y_test = test_data.iloc[:, 0].values

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = preprocess_data('../data/mnist_train.csv', '../data/mnist_test.csv')

    input_size = x_train.shape[0]
    hidden_size = 30
    output_size = 10
    learning_rate = 0.5
    epochs = 10
    batch_size = 32
    num_classes = 10

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    nn.train(x_train, y_train, x_test, y_test, learning_rate, epochs, batch_size, num_classes)
     
    nn.save_parameters('nn_parameters.npz')

    nn.load_parameters('nn_parameters.npz')

    test_accuracy = nn.evaluate(x_test, y_test)
    print(f"Test Accuracy after loading parameters: {test_accuracy:.4f}")