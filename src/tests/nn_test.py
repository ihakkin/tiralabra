import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from src.nn import NeuralNetwork, preprocess_data, softmax, sigmoid, sigmoid_prime


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.input_size = 784
        self.hidden_size = 30
        self.output_size = 10
        self.learning_rate = 0.1
        self.epochs = 100
        self.batch_size = 10

        self.nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, self.learning_rate, self.epochs, self.batch_size)

        script_dir = Path(__file__).resolve().parent
        data_path = script_dir / '../../data/mnist_train.csv'

        train_data = pd.read_csv(data_path)
        sample_data = train_data.sample(n=10, random_state=42)
        self.x_train = (sample_data.iloc[:, 1:].values / 255).T
        self.y_train = sample_data.iloc[:, 0].values

    def test_overfitting(self):
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train) 
        train_accuracy = self.nn.evaluate(self.x_train, self.y_train)
        self.assertTrue(train_accuracy == 1.0)

    def test_weight_changes(self):
        initial_w1 = self.nn.w1.copy()
        initial_w2 = self.nn.w2.copy()
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train)
        self.assertTrue(np.any(self.nn.w1 - initial_w1 != 0))
        self.assertTrue(np.any(self.nn.w2 - initial_w2 != 0))

    def test_bias_changes(self):
        initial_b1 = self.nn.b1.copy()
        initial_b2 = self.nn.b2.copy()
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train)
        self.assertTrue(np.any(self.nn.b1 - initial_b1 != 0))
        self.assertTrue(np.any(self.nn.b2 - initial_b2 != 0))

    def test_shuffled_minibatch(self):
        _, _, initial_output = self.nn.forward_propagation(self.x_train)
        shuffled_indices = np.random.permutation(self.x_train.shape[1])
        x_shuffled = self.x_train[:, shuffled_indices]
        _, _, shuffled_output = self.nn.forward_propagation(x_shuffled)
        for i in range(self.x_train.shape[1]):
            initial_index = shuffled_indices[i]
            self.assertTrue(np.allclose(shuffled_output[:, i], initial_output[:, initial_index], atol=1e-6))
            
    def test_invalid_input_shape(self):
        invalid_input = np.random.randn(self.input_size + 1, 1)
        self.assertRaises(ValueError, self.nn.forward_propagation, invalid_input)

    def test_initialization(self):
        self.assertEqual(self.nn.input_size, self.input_size)
        self.assertEqual(self.nn.hidden_size, self.hidden_size)
        self.assertEqual(self.nn.output_size, self.output_size)
        self.assertEqual(self.nn.learning_rate, self.learning_rate)
        self.assertEqual(self.nn.epochs, self.epochs)
        self.assertEqual(self.nn.batch_size, self.batch_size)
        self.assertEqual(self.nn.w1.shape, (self.hidden_size, self.input_size))
        self.assertEqual(self.nn.b1.shape, (self.hidden_size, 1))
        self.assertEqual(self.nn.w2.shape, (self.output_size, self.hidden_size))
        self.assertEqual(self.nn.b2.shape, (self.output_size, 1))
        self.assertIsNone(self.nn.test_accuracy)

    def test_save_load_parameters(self):
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train)
        self.nn.save_parameters('test_params.npz')
        nn2 = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, self.learning_rate, self.epochs, self.batch_size)
        nn2.load_parameters('test_params.npz')
        self.assertTrue(np.array_equal(self.nn.w1, nn2.w1))
        self.assertTrue(np.array_equal(self.nn.b1, nn2.b1))
        self.assertTrue(np.array_equal(self.nn.w2, nn2.w2))
        self.assertTrue(np.array_equal(self.nn.b2, nn2.b2))
        self.assertEqual(self.nn.test_accuracy, nn2.test_accuracy)
        Path('test_params.npz').unlink()

    def test_load_parameters_file_not_found(self):
        self.assertRaises(FileNotFoundError, self.nn.load_parameters, 'ghost.npz')
    
    def test_preprocess_data(self):
        script_dir = Path(__file__).resolve().parent
        train_data_path = script_dir / '../../data/mnist_train.csv'
        test_data_path = script_dir / '../../data/mnist_test.csv'

        x_train, y_train, x_test, y_test = preprocess_data(train_data_path, test_data_path)
        self.assertEqual(x_train.shape[0], 784)
        self.assertEqual(x_test.shape[0], 784)
        self.assertEqual(len(y_train), x_train.shape[1])
        self.assertEqual(len(y_test), x_test.shape[1])

   
    def test_softmax(self):
        z = np.array([1, 2, 3])
        result = softmax(z)
        self.assertTrue(np.allclose(result, np.exp(z) / np.sum(np.exp(z))))

    def test_sigmoid(self):
        z = np.array([0, 2, -2])
        result = sigmoid(z)
        expected = 1 / (1 + np.exp(-z))
        self.assertTrue(np.allclose(result, expected))

    def test_sigmoid_prime(self):
        z = np.array([0, 2, -2])
        result = sigmoid_prime(z)
        expected = sigmoid(z) * (1 - sigmoid(z))
        self.assertTrue(np.allclose(result, expected))
