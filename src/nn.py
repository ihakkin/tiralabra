"""
Neuroverkko
"""

import random
import numpy as np
import pandas as pd

def softmax(z):
    """
    Output-kerroksen aktivaatiofunktio. Antaa jokaiselle ennustettavalle luokalle
    arvon väliltä [0, 1]. Suurimman arvon saama luokka on neuroverkon ennuste syötteen arvoksi.

    Args:
        z: Array, jolle softmax lasketaan. Neuronin output ennen aktivaatiota.

    Returns:
        Array, jossa laskettu output-kerroksen aktivaatiot. Luokkien todennäköisyysjakauma.
    """
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=0)

def sigmoid(z):
    """
    Piilokerroksen aktivaatiofunktio.

    Args:
        z: Array, jolle sigmoid lasketaan

    Returns:
        Array, jossa laskettu piilokerroksen aktivaatio.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Sigmoid-funktion derivaatta. Käytetään vastavirta-algoritmissa.

    Args:
        z: Array, jolle sigmoid-derivaatta lasketaan

    Returns:
        Sigmoid-funktion derivaatan arvo.
    """
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    """
    Neuroverkko-luokka.

    Attributes:
        parameters: Sanakirja, joka sisältää verkon painot ja biasit.
        hyperparameters: Sanakirja, joka sisältää säädettävät hyperparametrit (input_size, hidden_size,
                output_size, learning_rate, epochs, batch_size).
        test_accuracy: testiaineistosta tehtävän luokittelun tarkkuus.
    """

    def __init__(self, hyperparameters):
        """
        Alustaa neuroverkon hyperparametrit ja kutsuu parametrien alustamisen.

        Args:
            hyperparameters: Sanakirja, joka sisältää säädettävät hyperparametrit.
            hidden_size (int): Piilokerroksen neuronien määrä.
                learning_rate (float): Kuinka suuria askelia otetaan gradientin vastavektorin suuntaan.
                epochs (int): Kuinka monta kertaa koulutusdata käydään läpi.
                batch_size (int): Minibatchien koko.

        Attributes:
            parameters (dict): Sanakirja, joka sisältää verkon painot ja biasit.
            test_accuracy (float): Testiaineistosta tehtävän luokittelun tarkkuus, aluksi None.
        """
        self.hyperparameters = hyperparameters
        self.test_accuracy = None
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Alustaa neuroverkon kerrokset sekä painot ja biasit.

        Returns:
            parameters: Sanakirja, joka sisältää painot ja biasit.
        """
        input_size = 784
        hidden_size = self.hyperparameters['hidden_size']
        output_size = 10

        # Alustetaan painot ja biasit satunnaisesti normaalijakauman arvoilla (keskiarvo 0, keskihajonta 1)
        # Arvot skaalattu 0.1:llä, mikä vähentää suurten lukujen vaikutusta ja vakauttaa oppimista
        parameters = {
            'w1': np.random.randn(hidden_size, input_size) * 0.1,
            'b1': np.random.randn(hidden_size, 1) * 0.1,
            'w2': np.random.randn(output_size, hidden_size) * 0.1,
            'b2': np.random.randn(output_size, 1) * 0.1,
        }
        return parameters

    def forward_propagation(self, x):
        """
        Suorittaa eteenpäin suuntautuvan laskennan.

        Args:
            x: Syöte arrayna.

        Returns:
            Sanakirja, joka sisältää piilokerroksen ja output-kerroksen painotetut summat
            ja aktivaatiofunktiot.
        """
        # Piilokerroksen painomatriisin ja syötteen välinen pistetulo, johon lisätään piilokerroksen bias.
        z1 = np.dot(self.parameters['w1'], x) + self.parameters['b1']
         # Lasketaan aktivaatiofunktio piilokerroksen neuroneille.
        a1 = sigmoid(z1)
        # Ulostulokerroksen painomatriisin ja piilokerroksen outputin välinen pistetulo,
        # johon lisätään ulostulokerroksen bias.
        z2 = np.dot(self.parameters['w2'], a1) + self.parameters['b2']
        # Lasketaan aktivaatiofunktio ulostulokerroksen neuroneille.
        a2 = softmax(z2)
        return {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    def backward_propagation(self, x_batch, y_batch, activations):
        """
        Vastavirta-algoritmi sekä painojen ja biasien päivitys.

        Args:
            x_batch: Array, jonka jokainen sarake on yksi koulutusdatan sample.
                    Pilkottu mini-batcheiksi.
            y_batch: Array, jonka arvot kertovat mikä luku kussakin mini-batchin samplessa on.
            activations: Sanakirja, joka sisältää aktivaatiofunktiot (a1, a2) ja z-arvot (z1, z2).
        """
        m = x_batch.shape[1]
        # Muutetaan y_batch one hot -muotoon, jossa luokat esitetään binaarisena.
        one_hot_y = np.eye(10)[y_batch].T

        delta2 = activations['a2'] - one_hot_y # Virhe output-kerroksessa (cross entropy -virhefunktion gradientti)
        nabla_w2 = np.dot(delta2, activations['a1'].T) / m # Output-kerroksen painojen gradientti
        nabla_b2 = np.sum(delta2, axis=1, keepdims=True) / m # Output-kerroksen biasien gradientti

        delta1 = np.dot(self.parameters['w2'].T, delta2) * sigmoid_prime(activations['z1']) # Piilokerroksen virhe
        nabla_w1 = np.dot(delta1, x_batch.T) / m # Piilokerroksen painojen gradientti
        nabla_b1 = np.sum(delta1, axis=1, keepdims=True) / m # Piilokerroksen biasien gradientti

        # Painojen ja biasien päivitys laskettujen gradienttien perusteella
        self.parameters['w1'] -= self.hyperparameters['learning_rate'] * nabla_w1
        self.parameters['b1'] -= self.hyperparameters['learning_rate'] * nabla_b1
        self.parameters['w2'] -= self.hyperparameters['learning_rate'] * nabla_w2
        self.parameters['b2'] -= self.hyperparameters['learning_rate'] * nabla_b2

    def train(self, x_train, y_train, x_test, y_test):
        """
        Kouluttaa neuroverkon.

        Args:
            x_train: array, jossa koko koulutusdatan syötteet.
            y_train: array, jossa koko koulutusdatan todelliset arvot.
            x_test: array, jossa testiaineiston syötteet.
            y_test: array, jossa testiaineiston todelliset arvot.
        """
        training_data = list(zip(x_train.T, y_train))
        for epoch in range(self.hyperparameters['epochs']):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.hyperparameters['batch_size']]
                for k in range(0, len(training_data), self.hyperparameters['batch_size'])]
            for mini_batch in mini_batches:
                x_batch, y_batch = zip(*mini_batch)
                x_batch = np.array(x_batch).T
                y_batch = np.array(y_batch)
                activations = self.forward_propagation(x_batch)
                self.backward_propagation(x_batch, y_batch, activations)
            self.test_accuracy = self.evaluate(x_test, y_test)
            print(f"Epoch {epoch + 1}: Test accuracy {self.test_accuracy:.4f}")

    def evaluate(self, x_test, y_test):
        """
        Arvioi neuroverkon tarkkuuden testiaineistolla.

        Args:
            x_test: array, jossa testiaineiston syötteet. Jokainen sarake vastaa yhtä samplea.
            y_test: array, jossa testiaineiston todelliset arvot.

        Returns:
            Testiaineiston tarkkuus.
        """
        activations = self.forward_propagation(x_test)
        predictions = np.argmax(activations['a2'], axis=0)
        return np.mean(predictions == y_test)

    def save_parameters(self, filename):
        """
        Tallentaa neuroverkon painot, biasit ja testitarkkuuden tiedostoon käyttöliittymää varten.

        Args:
            filename: Tiedoston nimi, johon parametrit tallennetaan.
        """
        np.savez(filename, **self.parameters, test_accuracy=self.test_accuracy)

    def load_parameters(self, filename):
        """
        Lataa neuroverkon painot, biasit ja testitarkkuuden tiedostosta käyttöliittymää varten.

        Args:
            filename: Tiedoston nimi, josta parametrit ladataan.
        """
        data = np.load(filename)
        self.parameters = {
        'w1': data['w1'],
        'b1': data['b1'],
        'w2': data['w2'],
        'b2': data['b2']
        }
        self.test_accuracy = data['test_accuracy']

def preprocess_data(train_file, test_file):
    """
    Esikäsittelee koulutus- ja testidatan.

    Args:
        train_file: Polku koulutusdatan tiedostoon.
        test_file: Polku testidatan tiedostoon.

    Returns:
        Tuple, joka sisältää esikäsitellyt koulutus- ja testisyötteet ja -arvot.
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    x_train = (train_data.iloc[:, 1:].values / 255).T
    y_train = train_data.iloc[:, 0].values

    x_test = (test_data.iloc[:, 1:].values / 255).T
    y_test = test_data.iloc[:, 0].values

    return x_train, y_train, x_test, y_test

def main(hidden_size, learning_rate, epochs, batch_size):
    """
    Pääfunktio, joka lataa datan, kouluttaa neuroverkon ja tallentaa parametrit.

    Args:
        hidden_size: Piilokerroksen neuronien määrä.
        learning_rate: Oppimisnopeus.
        epochs: Kuinka monta kertaa koulutusdata käydään läpi.
        batch_size: Minibatchien koko.
    """
    x_train, y_train, x_test, y_test = preprocess_data(
        '../data/mnist_train.csv', '../data/mnist_test.csv')

    hyperparameters = {
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size
    }

    nn = NeuralNetwork(hyperparameters)
    nn.train(x_train, y_train, x_test, y_test)

    test_accuracy = nn.evaluate(x_test, y_test)
    nn.save_parameters('nn_parameters.npz')
    print(f"Saved test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main(hidden_size=200, learning_rate=0.5, epochs=20, batch_size=32)
