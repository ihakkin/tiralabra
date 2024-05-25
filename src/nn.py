import pandas as pd
import numpy as np


def relu(z):
    return np.maximum(z,0)

def relu_prime(z):
    return z > 0

def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=0)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def init_network(size):
    w1 = np.random.rand(10,size) *0.1 
    b1 = np.random.rand(10,1) *0.1
    w2 = np.random.rand(10,10) *0.1
    b2 = np.random.rand(10,1) *0.1
    return w1,b1,w2,b2

def forward_propagation(x,w1,b1,w2,b2):
    z1 = np.dot(w1,x) + b1 #10xm
    A1 = relu(z1) #10xm
    z2 = np.dot(w2,A1) + b2 #10xm
    A2 = softmax(z2) #10xm
    return z1, A1, z2, A2

def one_hot(y):
    one_hot_y = np.zeros((y.max()+1,y.size))
    one_hot_y[y,np.arange(y.size)] = 1 
    return one_hot_y


def backward_propagation(x_train, y_train, w2, A1, A2, z1, m):
    one_hot_y = one_hot(y_train)
    
    # Output layer
    delta2 = A2 - one_hot_y  #10xm
    nabla_w2 = np.dot(delta2, A1.T) / m #10x10
    nabla_b2 = np.sum(delta2, axis=1, keepdims=True) / m #10x1

    #print(f"ad2: {ad2.shape}")
    #print(f"delta2: {delta2.shape}")
    #print(f"nablaw2: {nabla_w2.shape}")
    #print(f"nablab2: {nabla_b2.shape}")


    # Hidden layer
    delta1 = np.dot(w2.T, delta2) * relu_prime(z1) #10xm
    nabla_w1 = np.dot(delta1, x_train.T) / m #10x784
    nabla_b1 = np.sum(delta1, axis=1, keepdims=True) / m #10x1
    #print(f"nablab1: {nabla_b1.shape}")
    #print("Gradient w1 max, min:", np.max(nabla_w1), np.min(nabla_w1)) 
    #print("Gradient w2 max, min:", np.max(nabla_w2), np.min(nabla_w2))
    #print("gradients(delta1):", np.mean(delta1))
    return (nabla_w1, nabla_b1, nabla_w2, nabla_b2)


def update(learning_rate, w1, b1, w2, b2, nabla_w1, nabla_b1, nabla_w2, nabla_b2):
    w1 -= learning_rate * nabla_w1
    b1 -= learning_rate * nabla_b1
    w2 -= learning_rate * nabla_w2
    b2 -= learning_rate * nabla_b2
    return w1, b1, w2, b2


def gradient_descent(x_train, y_train, x_test, y_test, learning_rate, epochs):
    size, m = x_train.shape
    w1, b1, w2, b2 = init_network(size)
    for i in range(epochs):
        z1, a1, z2, a2 = forward_propagation(x_train, w1, b1, w2, b2)
        nabla_w1, nabla_b1, nabla_w2, nabla_b2 = backward_propagation(x_train, y_train, w2, a1, a2, z1, m)
        w1, b1, w2, b2 = update(learning_rate, w1, b1, w2, b2, nabla_w1, nabla_b1, nabla_w2, nabla_b2)   

        if (i+1) % 10 == 0: 
            test_accuracy = evaluate(x_test, y_test, w1, b1, w2, b2)
            print(f"Epochs {i+1}: Test Accuracy = {test_accuracy:.4f}")

    return w1, b1, w2, b2


def evaluate(x, y, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(x, w1, b1, w2, b2)
    predictions = np.argmax(a2,0)
    accuracy = np.mean(predictions == y)
    return accuracy




if __name__ == "__main__":

    train_data = pd.read_csv('/home/ihakkine/koulu/nn//data/mnist_train.csv')
    test_data = pd.read_csv('/home/ihakkine/koulu/nn/data/mnist_test.csv')

    x_train = (train_data.iloc[:, 1:].values / 255).T # pikseliarvojen skaalaus
    y_train = train_data.iloc[:, 0].values

    x_test = (test_data.iloc[:, 1:].values / 255).T
    y_test = test_data.iloc[:, 0].values
    print("x_train shape:", x_train.shape) 
    print("x_test shape:", x_test.shape) 

    w1, b1, w2, b2 = gradient_descent(x_train, y_train, x_test, y_test, learning_rate=0.1, epochs=300)

   