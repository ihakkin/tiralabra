"""
Flask-sovellus, joka käyttää neuroverkkoa MNIST-numeroiden ennustamiseen.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template
from nn import NeuralNetwork

matplotlib.use('Agg')

app = Flask(__name__, template_folder='templates')
nn = NeuralNetwork(
    input_size=784, hidden_size=30, output_size=10,
    learning_rate=0.5, epochs=10, batch_size=32
)

nn.load_parameters('../src/nn_parameters.npz')
test_accuracy = nn.test_accuracy * 100

def load_data(sample_size=1000):
    """
    Lataa ja esikäsittelee testidatan.

    Args:
        sample_size: Näytteen koko, kuinka monta esimerkkiä ladataan.

    Returns:
        Tuple, joka sisältää esikäsitellyt testisyötteet ja -arvot.
    """
    test_data = pd.read_csv('../data/mnist_test.csv')
    sampled_data = test_data.sample(n=sample_size)
    test_x = (sampled_data.iloc[:, 1:].values / 255).T
    test_y = sampled_data.iloc[:, 0].values
    return test_x, test_y

x_test_data, y_test_data = load_data(sample_size=1000)

def save_mnist_image(image_array, image_path):
    """
    Tallentaa MNIST-kuvan tiedostoon.

    Args:
        image_array: Kuvadata, joka tallennetaan.
        image_path: Polku, johon kuva tallennetaan.
    """
    plt.figure(figsize=(2, 2))
    plt.imshow(image_array.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@app.route('/', methods=['GET'])
def index():
    """
    Renderöi etusivun.

    Returns:
        Renderöity HTML-sivu.
    """
    return render_template('index.html', result=None, test_accuracy=f"{test_accuracy:.2f}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Ennustaa satunnaisen testikuvan luokan.

    Returns:
        Renderöity HTML-sivu ennustetun tuloksen kanssa.
    """
    random_index = random.randint(0, x_test_data.shape[1] - 1)
    input_image = x_test_data[:, random_index].reshape(784, 1)
    _, _, prediction = nn.forward_propagation(input_image)
    predicted_class = np.argmax(prediction, axis=0)
    true_label = y_test_data[random_index]
    result = {
        'true_label': int(true_label),
        'predicted_class': int(predicted_class[0])
    }
    image_path = 'static/mnist_image.png'
    save_mnist_image(input_image, image_path)
    return render_template(
        'index.html', result=result,
        test_accuracy=f"{test_accuracy:.2f}", image_path=image_path
    )

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
