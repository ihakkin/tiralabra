from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import random
from nn import NeuralNetwork

app = Flask(__name__, template_folder='templates')
nn = NeuralNetwork(input_size=784, hidden_size=30, output_size=10)

nn.load_parameters('/home/ihakkine/koulu/nn/src/nn_parameters.npz')

def load_data(sample_size=1000):
    test_data = pd.read_csv('/home/ihakkine/koulu/nn/data/mnist_test.csv')  #korjaa
    sampled_data = test_data.sample(n=sample_size)
    x_test = (sampled_data.iloc[:, 1:].values / 255).T
    y_test = sampled_data.iloc[:, 0].values
    return x_test, y_test

x_test, y_test = load_data(sample_size=1000)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    index = random.randint(0, x_test.shape[1] - 1)
    input_image = x_test[:, index].reshape(784, 1)
    _, _, _, prediction = nn.forward_propagation(input_image)
    predicted_class = np.argmax(prediction, axis=0)
    true_label = y_test[index]
    result = {'true_label': int(true_label), 'predicted_class': int(predicted_class[0])}
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)