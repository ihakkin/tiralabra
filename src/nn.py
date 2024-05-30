import pandas as pd
import numpy as np

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

def one_hot(y):
    one_hot_y = np.zeros((y.max() + 1, y.size))
    one_hot_y[y, np.arange(y.size)] = 1 
    return one_hot_y



    #def save_parameters(self, filename):
       # np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    #def load_parameters(self, filename):
        #data = np.load(filename)
        #self.w1 = data['w1']
        ##self.b1 = data['b1']
        ##self.w2 = data['w2']
        #self.b2 = data['b2']    
    
    def forward_propagation(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2
    
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambd=0.01, dropout_rate=0.5):
        self.w1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.random.randn(hidden_size, 1) * 0.1
        self.w2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.random.randn(output_size, 1) * 0.1
        self.lambd = lambd  # Regularisointiparametri
        self.dropout_rate = dropout_rate  # Dropout-parametri

    def save_parameters(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load_parameters(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']  
    
    def forward_propagation(self, x, train=True):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = relu(z1)
        
        if train:
            self.dropout_mask = (np.random.rand(*a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            a1 *= self.dropout_mask
        else:
            a1 *= (1 - self.dropout_rate)
        
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2
    
    def backward_propagation(self, x_train, y_train, a1, a2, z1, learning_rate):
        m = x_train.shape[1]
        one_hot_y = one_hot(y_train)
        
        # Output layer
        delta2 = a2 - one_hot_y
        nabla_w2 = (np.dot(delta2, a1.T) + self.lambd * self.w2) / m
        nabla_b2 = np.sum(delta2, axis=1, keepdims=True) / m

        # Hidden layer
        delta1 = np.dot(self.w2.T, delta2) * relu_prime(z1)
        delta1 *= self.dropout_mask  # Apply dropout mask
        nabla_w1 = (np.dot(delta1, x_train.T) + self.lambd * self.w1) / m
        nabla_b1 = np.sum(delta1, axis=1, keepdims=True) / m

        # Update parameters
        self.w1 -= learning_rate * nabla_w1
        self.b1 -= learning_rate * nabla_b1
        self.w2 -= learning_rate * nabla_w2
        self.b2 -= learning_rate * nabla_b2

    def train(self, x_train, y_train, x_test, y_test, learning_rate, epochs):
        for i in range(epochs):
            z1, a1, z2, a2 = self.forward_propagation(x_train, train=True)
            self.backward_propagation(x_train, y_train, a1, a2, z1, learning_rate)

            #if (i + 1) % 10 == 0:
            test_accuracy = self.evaluate(x_test, y_test)
            print(f"Epoch {i + 1}: Test Accuracy = {test_accuracy:.4f}")

    def evaluate(self, x, y):
        _, _, _, a2 = self.forward_propagation(x, train=False)
        predictions = np.argmax(a2, axis=0)
        accuracy = np.mean(predictions == y)
        return accuracy

if __name__ == "__main__":
    train_data = pd.read_csv('/home/ihakkine/koulu/nn/data/mnist_train.csv')
    test_data = pd.read_csv('/home/ihakkine/koulu/nn/data/mnist_test.csv')

    x_train = (train_data.iloc[:, 1:].values / 255).T
    y_train = train_data.iloc[:, 0].values

    x_test = (test_data.iloc[:, 1:].values / 255).T
    y_test = test_data.iloc[:, 0].values

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    input_size = x_train.shape[0]
    hidden_size = 30
    output_size = 10
    learning_rate = 0.4
    epochs = 100
    lambd = 0.01  # Regularisointiparametri
    dropout_rate = 0.5  # Dropout-parametri

    nn = NeuralNetwork(input_size, hidden_size, output_size, lambd=lambd, dropout_rate=dropout_rate)
    nn.train(x_train, y_train, x_test, y_test, learning_rate, epochs)

# Tallenna painot
    nn.save_parameters('nn_parameters.npz')

    # Lataa painot
    #nn.load_parameter('nn_parameters.npz')

    # Testaa ladattuja painoja
    #test_accuracy = nn.evaluate(x_test, y_test)
    #print(f"Test Accuracy after loading parameters: {test_accuracy:.4f}")