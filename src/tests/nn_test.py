import unittest
import pandas as pd
import numpy as np
from src.nn import NeuralNetwork, one_hot



class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.input_size = 784
        self.hidden_size = 10
        self.output_size = 10
        self.learning_rate = 0.1
        self.epochs = 1000
        self.lambd = 0.01

        self.nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, lambd=self.lambd)

        train_data = pd.read_csv('/home/ihakkine/koulu/nn/data/mnist_train.csv')
        sample_data = train_data.sample(n=10, random_state=42)
        self.x_train = (sample_data.iloc[:, 1:].values / 255).T
        self.y_train = sample_data.iloc[:, 0].values


    def compute_loss(self, a2, y):
        m = y.shape[0]
        one_hot_y = one_hot(y)
        log_probs = -np.log(a2[one_hot_y == 1])
        loss = np.sum(log_probs) / m
        return loss


    def test_overfitting(self):
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train, self.learning_rate, self.epochs)
        
        _, _, _, a2 = self.nn.forward_propagation(self.x_train)
        train_loss = self.compute_loss(a2, self.y_train)
        train_accuracy = self.nn.evaluate(self.x_train, self.y_train)
        print(f"Training accuracy: {train_accuracy}")
        print(f"Training loss: {train_loss}")
    
        self.assertTrue(train_accuracy == 1.0 or train_loss < 1e-6)


    def test_weight_changes(self):
        initial_w1 = self.nn.w1.copy()
        initial_w2 = self.nn.w2.copy()

        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train, self.learning_rate, epochs=1)

        print("Delta w1:", self.nn.w1 - initial_w1)
        print("Delta w2:", self.nn.w2 - initial_w2)

        self.assertTrue(np.any(self.nn.w1 - initial_w1 != 0))
        self.assertTrue(np.any(self.nn.w2 - initial_w2 != 0))
   

    def test_bias_changes(self):
        initial_b1 = self.nn.b1.copy()
        initial_b2 = self.nn.b2.copy()

        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train, self.learning_rate, epochs=1) 

        print("Delta b1:", self.nn.b1 - initial_b1)
        print("Delta b2:", self.nn.b2 - initial_b2)

        self.assertTrue(np.any(self.nn.b1 - initial_b1 != 0))
        self.assertTrue(np.any(self.nn.b2 - initial_b2 != 0))
