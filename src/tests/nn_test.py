import unittest
from src.nn import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        print("Set up goes here")

    def test_hello_world(self):
        self.assertEqual("Hello world", "Hello world")