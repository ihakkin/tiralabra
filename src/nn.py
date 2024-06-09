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
        Array, jossa laskettu piilokerroksen aktivaatiot.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Sigmoid-funktion derivaatta. Käytetään vastavirta-algoritmissa.

    Args:
        z: Array, jolle sigmoid-derivaatta lasketaan

    Returns:
        Sigmoid-funktion derivaatan arvo
    """
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    """
    Neuroverkko-luokka.

    Attributes:
        w1: array, jossa piilokerroksen painot
        b1: array, jossa piilokerroksen biasit
        w2: array, jossa ulostulokerroksen painot
        b2: array, jossa ulostulokerroksen biasit
        test_accuracy: testiaineiston tarkkuus (float tai None)

    Methods:
        forward_propagation: Suorittaa eteenpäin suuntautuvan laskennan
        backward_propagation: Suorittaa vastavirta-algoritmin ja parametrien päivityksen.
        train: Kouluttaa neuroverkon.
        evaluate: Arvioi neuroverkon tarkkuuden
        save_parameters: Tallentaa neuroverkon parametrit tiedostoon.
        load_parameters: Lataa neuroverkon parametrit tiedostosta.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs, batch_size):
        """
        Alustaa neuroverkon.

        Args:
            input_size: Syötekerroksen neuronien määrä.
            hidden_size: Piilokerroksen neuronien määrä.
            output_size: Output-kerroksen neuronien määrä.
            learning_rate: Oppimisnopeus.  Oppimisnopeus. Vaikuttaa siihen kuinka suuria askelia otetaan gradientin vastavektorin suuntaan.
            epochs: Kuinka monta kertaa koulutusdata käydään läpi
            batch_size: Minibatchien koko,  eli kuinka monta samplea käsitellään kerrallaan.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Alustetaan painot ja biasit satunnaisesti
        # Skaalattu 0.1:llä vähentämään suurten lukujen vaikutusta
        self.w1 = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.b1 = np.random.randn(self.hidden_size, 1) * 0.1
        self.w2 = np.random.randn(self.output_size, self.hidden_size) * 0.1
        self.b2 = np.random.randn(self.output_size, 1) * 0.1
        self.test_accuracy = None

    def forward_propagation(self, x):
        """
        Suorittaa eteenpäin suuntautuvan laskennan

        Args:
            x: Syöte arrayna

        Returns:
            z1, a1, a2: Piilokerroksen ja output-kerroksen painotetut summat ja aktivaatiofunktiot.
        """
        z1 = np.dot(self.w1, x) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = softmax(z2)
        return z1, a1, a2

    def backward_propagation(self, x_batch, y_batch, a1, a2, z1):
        """
        Vastavirta-algoritmi sekä painojen ja biasien päivitys.

        Args:
            x_batch: array, jonka jokainen sarake on yksi koulutusdatan sample. Pilkottu mini-batcheiksi.
            y_batch: array, jonka arvot kertovat mikä luku kussakin mini-batchin samplessa on.
            a1: Piilokerroksen aktivaatio
            a2: Output-kerroksen aktivaatio, eli mallin ennustama arvo, jota verrataan todelliseen arvoon.
            z1: Piilokerroksen painotettu summa ennen aktivaatiofunktiota.
        """
        m = x_batch.shape[1]
        one_hot_y = np.eye(self.output_size)[y_batch].T

        delta2 = a2 - one_hot_y # Virhe output-kerroksessa (cross entropy -virhefunktion gradientti)
        nabla_w2 = np.dot(delta2, a1.T) / m # Output-kerroksen painojen gradientti
        nabla_b2 = np.sum(delta2, axis=1, keepdims=True) / m # Output-kerroksen biasien gradientti

        delta1 = np.dot(self.w2.T, delta2) * sigmoid_prime(z1) # Piilokerroksen virhe
        nabla_w1 = np.dot(delta1, x_batch.T) / m # Piilokerroksen painojen gradientti
        nabla_b1 = np.sum(delta1, axis=1, keepdims=True) / m # Piilokerroksen biasien gradientti

        # Painojen ja biasien päivitys laskettujen gradienttien perusteella
        self.w1 -= self.learning_rate * nabla_w1
        self.b1 -= self.learning_rate * nabla_b1
        self.w2 -= self.learning_rate * nabla_w2
        self.b2 -= self.learning_rate * nabla_b2

    def train(self, x_train, y_train, x_test, y_test):
        """
        Kouluttaa neuroverkon

        Args:
            x_train: array, jossa koko koulutusdatan syötteet. Jokainen sarake vastaa yhtä samplea.
            y_train: array, jossa koko koulutusdatan todelliset arvot.
            x_test: array, jossa testiaineiston syötteet. Jokainen sarake vastaa yhtä samplea.
            y_test: array, jossa testiaineiston todelliset arvot.
        """
        training_data = list(zip(x_train.T, y_train))
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + self.batch_size] for k in range(0, len(training_data), self.batch_size)]
            for mini_batch in mini_batches:
                x_batch, y_batch = zip(*mini_batch)
                x_batch = np.array(x_batch).T
                y_batch = np.array(y_batch)
                z1, a1, a2 = self.forward_propagation(x_batch)
                self.backward_propagation(x_batch, y_batch, a1, a2, z1)
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
        _, _, a2 = self.forward_propagation(x_test)
        predictions = np.argmax(a2, axis=0)
        return np.mean(predictions == y_test)

    def save_parameters(self, filename):
        """
        Tallentaa neuroverkon painot, biasit ja testitarkkuuden tiedostoon käyttöliittymää varten

        Args:
            filename: Tiedoston nimi, johon parametrit tallennetaan.
            test_accuracy: Testiaineiston tarkkuus
        """
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, test_accuracy=self.test_accuracy)

    def load_parameters(self, filename):
        """
        Lataa neuroverkon painot, biasit ja testitarkkuuden tiedostosta käyttöliittymää varten

        Args:
            filename: Tiedoston nimi, josta parametrit ladataan.
        """
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.test_accuracy = data['test_accuracy']

def preprocess_data(train_file, test_file):
    """
    Esikäsittelee koulutus- ja testidatan

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
        hidden_size: Piilokerroksen neuronien määrä
        learning_rate: Oppimisnopeus
        epochs: Kuinka monta kertaa koulutusdata käydään läpi
        batch_size: Minibatchien koko
    """
    x_train, y_train, x_test, y_test = preprocess_data('../data/mnist_train.csv', '../data/mnist_test.csv')

    input_size = x_train.shape[0]
    output_size = 10

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, epochs, batch_size)
    nn.train(x_train, y_train, x_test, y_test)

    test_accuracy = nn.evaluate(x_test, y_test)
    nn.save_parameters('nn_parameters.npz')
    print(f"Saved test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main(hidden_size=30, learning_rate=0.5, epochs=10, batch_size=32)
    