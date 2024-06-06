from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nn import NeuralNetwork

app = Flask(__name__, template_folder='templates')
nn = NeuralNetwork(input_size=784, hidden_size=30, output_size=10)

nn.load_parameters('../src/nn_parameters.npz')
test_accuracy = nn.test_accuracy *100

def load_data(sample_size=1000):
    test_data = pd.read_csv('../data/mnist_test.csv') 
    sampled_data = test_data.sample(n=sample_size)
    x_test = (sampled_data.iloc[:, 1:].values / 255).T
    y_test = sampled_data.iloc[:, 0].values
    return x_test, y_test

x_test, y_test = load_data(sample_size=1000)

def save_mnist_image(image_array, image_path):
    plt.figure(figsize=(2,2))
    plt.imshow(image_array.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None, test_accuracy=test_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    index = random.randint(0, x_test.shape[1] - 1)
    input_image = x_test[:, index].reshape(784, 1)
    _, _, _, prediction = nn.forward_propagation(input_image)
    predicted_class = np.argmax(prediction, axis=0)
    true_label = y_test[index]
    result = {'true_label': int(true_label), 'predicted_class': int(predicted_class[0])}
    image_path = 'static/mnist_image.png'
    save_mnist_image(input_image, image_path)
    
    return render_template('index.html', result=result, test_accuracy=test_accuracy, image_path=image_path)

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)