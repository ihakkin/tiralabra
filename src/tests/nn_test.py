import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from src.nn import NeuralNetwork, preprocess_data, softmax, sigmoid, sigmoid_prime


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        hyperparameters = {
            'hidden_size': 30,
            'learning_rate': 0.1,
            'epochs': 100,
            'batch_size': 10
        }

        self.nn = NeuralNetwork(hyperparameters)

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
        initial_w1 = self.nn.parameters['w1'].copy()
        initial_w2 = self.nn.parameters['w2'].copy()
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train)
        self.assertTrue(np.any(self.nn.parameters['w1'] - initial_w1 != 0))
        self.assertTrue(np.any(self.nn.parameters['w2'] - initial_w2 != 0))

    def test_bias_changes(self):
        initial_b1 = self.nn.parameters['b1'].copy()
        initial_b2 = self.nn.parameters['b2'].copy()
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train)
        self.assertTrue(np.any(self.nn.parameters['b1'] - initial_b1 != 0))
        self.assertTrue(np.any(self.nn.parameters['b2'] - initial_b2 != 0))

    def test_shuffled_minibatch(self):
        activations_initial = self.nn.forward_propagation(self.x_train)
        shuffled_indices = np.random.permutation(self.x_train.shape[1])
        x_shuffled = self.x_train[:, shuffled_indices]
        activations_shuffled = self.nn.forward_propagation(x_shuffled)
        for i in range(self.x_train.shape[1]):
            initial_index = shuffled_indices[i]
            self.assertTrue(np.allclose(activations_shuffled['a2'][:, i], activations_initial['a2'][:, initial_index], atol=1e-6))

    def test_invalid_input_shape(self):
        invalid_input = np.random.randn(785, 1)
        self.assertRaises(ValueError, self.nn.forward_propagation, invalid_input)

    def test_initialization(self):
        self.assertEqual(self.nn.hyperparameters['hidden_size'], 30)
        self.assertEqual(self.nn.hyperparameters['learning_rate'], 0.1)
        self.assertEqual(self.nn.hyperparameters['epochs'], 100)
        self.assertEqual(self.nn.hyperparameters['batch_size'], 10)
        self.assertEqual(self.nn.parameters['w1'].shape, (30, 784))
        self.assertEqual(self.nn.parameters['b1'].shape, (30, 1))
        self.assertEqual(self.nn.parameters['w2'].shape, (10, 30))
        self.assertEqual(self.nn.parameters['b2'].shape, (10, 1))
        self.assertIsNone(self.nn.test_accuracy)

    def test_save_load_parameters(self):
        self.nn.train(self.x_train, self.y_train, self.x_train, self.y_train)
        self.nn.save_parameters('test_params.npz')
        nn2 = NeuralNetwork(self.nn.hyperparameters)
        nn2.load_parameters('test_params.npz')
        self.assertTrue(np.array_equal(self.nn.parameters['w1'], nn2.parameters['w1']))
        self.assertTrue(np.array_equal(self.nn.parameters['b1'], nn2.parameters['b1']))
        self.assertTrue(np.array_equal(self.nn.parameters['w2'], nn2.parameters['w2']))
        self.assertTrue(np.array_equal(self.nn.parameters['b2'], nn2.parameters['b2']))
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

