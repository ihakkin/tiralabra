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
        w1, b1, w2, b2: Painot ja biasit.
        hidden_size, learning_rate, epochs, batch_size: Säädettävät hyperparametrit.
        test_accuracy: Testiaineistosta tehtävän luokittelun tarkkuus.
    """

    def __init__(self, hyperparameters):
        """
        Alustaa neuroverkon hyperparametrit ja kutsuu parametrien alustamisen.

        Args:
            hyperparameters: Sanakirja, joka sisältää säädettävät hyperparametrit.
                hidden_size (int): Piilokerroksen neuronien määrä.
                learning_rate (float): Oppimisnopeus vaikuttaa siihen, kuinka suuria askelia
                                       otetaan gradientin vastavektorin suuntaan.
                epochs (int): Kuinka monta kertaa koulutusdata käydään läpi.
                batch_size (int): Minibatchien koko.

        Attributes:
            w1, b1, w2, b2: Painot ja biasit.
            test_accuracy (float): Testiaineistosta tehtävän luokittelun tarkkuus, aluksi None.
        """
        self.hidden_size = hyperparameters['hidden_size']
        self.learning_rate = hyperparameters['learning_rate']
        self.epochs = hyperparameters['epochs']
        self.batch_size = hyperparameters['batch_size']
        self.test_accuracy = None
        self.w1, self.b1, self.w2, self.b2 = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Alustaa neuroverkon kerrokset sekä parametrit.
        Painot ja biasit alustetaan satunnaisesti normaalijakauman arvoilla (keskiarvo 0,
        keskihajonta 1). Arvot skaalattu 0.1:llä, mikä vähentää suurten lukujen vaikutusta ja
        vakauttaa oppimista

        Returns:
            w1, b1, w2, b2: Painot ja biasit.
        """
        input_size = 784
        hidden_size = self.hidden_size
        output_size = 10

        w1 = np.random.randn(hidden_size, input_size) * 0.1
        b1 = np.random.randn(hidden_size, 1) * 0.1
        w2 = np.random.randn(output_size, hidden_size) * 0.1
        b2 = np.random.randn(output_size, 1) * 0.1
        return w1, b1, w2, b2

    def forward_propagation(self, x):
        """
        Suorittaa eteenpäin suuntautuvan laskennan neuroverkossa. Ottaa syötteen input-kerroksessa
        ja laskee kerroksittain lineaarikombinaatiot ja lähettää ne seuraavaan kerrokseen
        aktivaation jälkeen. Tuloksia käytetään vastavirta-algoritmissa ja verkon tarkkuuden
        evaluaatiossa.

        Args:
            x: Syöte arrayna.

         Returns:
            a1, a2: Kerrosten aktivoidut outputit.
            z1, z2: Kerrosten outputit ennen aktivointia.
        """
        # Piilokerroksen painomatriisin ja syötteen pistetulo, johon lisätään piilokerroksen bias.
        z1 = np.dot(self.w1, x) + self.b1
        # Lasketaan aktivaatiofunktio piilokerroksen neuroneille.
        a1 = sigmoid(z1)
        # Ulostulokerroksen painomatriisin ja piilokerroksen outputin välinen pistetulo,
        # johon lisätään ulostulokerroksen bias.
        z2 = np.dot(self.w2, a1) + self.b2
        # Lasketaan aktivaatiofunktio ulostulokerroksen neuroneille.
        a2 = softmax(z2)
        return a1, a2, z1

    def backward_propagation(self, x_batch, y_batch, a1, a2, z1):
        """
        Vastavirta-algoritmi minimoi virhettä eli eroa verkon tekemän ennusteen ja todellisten luokkien
        välillä. Tavoitteena on löytää minimointia vastaavat painot ja biasit. Algoritmi laskee verkon
        virheen osittaisderivaatat suhteessa painoihin ja biaseihin. Gradienttimenetelmässä 
        osittaisderivaatoista koostuvan gradientin perusteella päivitetään uudet parametrien arvot.

        Args:
            x_batch: Array, jonka jokainen sarake on yksi koulutusdatan sample.
                    Pilkottu mini-batcheiksi.
            y_batch: Array, jonka arvot kertovat mikä luku kussakin mini-batchin samplessa on.
            a1, a2: Kerrosten aktivoidut outputit.
            z1: Piilokerroksen output ennen aktivointia.
        """
        m = x_batch.shape[1]
        # Muutetaan y_batch one hot -muotoon, jossa luokat esitetään binaarisena.
        one_hot_y = np.eye(10)[y_batch].T

        # Lasketaan virhe output-kerroksessa, eli ero ennusteen ja todellisen luokan välillä.
        # Hukkafunktiona cross entropy
        delta2 = a2 - one_hot_y

        # Lasketaan gradientit ulostulokerroksen painoille (nabla_w2) ja biaseille (nabla_b2).
        nabla_w2 = np.dot(delta2, a1.T) / m
        nabla_b2 = np.sum(delta2, axis=1, keepdims=True) / m

        # Lasketaan virhe piilokerroksessa (delta1) käyttäen ulostulokerroksen virhettä ja
        # piilokerroksen aktivaatioiden derivaattaa (sigmoid_prime).
        delta1 = np.dot(self.w2.T, delta2) * sigmoid_prime(z1)

        # Lasketaan gradientit piilokerroksen painoille (nabla_w1) ja biaseille (nabla_b1).
        nabla_w1 = np.dot(delta1, x_batch.T) / m
        nabla_b1 = np.sum(delta1, axis=1, keepdims=True) / m

        # Päivitetään painot ja biasit laskettujen gradienttien perusteella.
        self.w1 -= self.learning_rate * nabla_w1
        self.b1 -= self.learning_rate * nabla_b1
        self.w2 -= self.learning_rate * nabla_w2
        self.b2 -= self.learning_rate * nabla_b2

    def train(self, x_train, y_train, x_test, y_test):
        """
        Kouluttaa neuroverkon. Koulutusdata jaetaan satunnaistettuihin alijoukkoihin, "mini-batch",
        jotka viedään verkon läpi eteenpäin suuntautuvassa laskennassa (forward propagation). Laskettuja
        arvoja käytetään taaksepäin suuntautuvassa vastavirta-algoritmissa, joka minimoi ennusteen virhettä.
        Parametrit päivitetään jokaisen mini-batchin ajon jälkeen, eli neuroverkko korjaa itseään.
        Lopuksi kutsutaan arviointifunktiota testidatalla.

        Args:
            x_train: array, jossa koko koulutusdatan syötteet. Jokainen sarake vastaa yhtä syötettä.
            y_train: array, jossa koko koulutusdatan todelliset arvot.
            x_test: array, jossa testiaineiston syötteet. Jokainen sarake vastaa yhtä syötettä.
            y_test: array, jossa testiaineiston todelliset arvot.
        """
        training_data = list(zip(x_train.T, y_train))
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.batch_size]
                for k in range(0, len(training_data), self.batch_size)]
            for mini_batch in mini_batches:
                x_batch, y_batch = zip(*mini_batch)
                x_batch = np.array(x_batch).T
                y_batch = np.array(y_batch)
                a1, a2, z1 = self.forward_propagation(x_batch)
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
        _, a2, _ = self.forward_propagation(x_test)
        predictions = np.argmax(a2, axis=0)
        return np.mean(predictions == y_test)

    def save_parameters(self, filename):
        """
        Tallentaa neuroverkon painot, biasit ja testitarkkuuden tiedostoon käyttöliittymää varten.

        Args:
            filename: Tiedoston nimi, johon parametrit tallennetaan.
        """
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 test_accuracy=self.test_accuracy, hidden_size=self.hidden_size,
                 learning_rate=self.learning_rate, epochs=self.epochs, batch_size=self.batch_size)

    def load_parameters(self, filename):
        """
        Lataa neuroverkon painot, biasit ja testitarkkuuden tiedostosta käyttöliittymää varten.

        Args:
            filename: Tiedoston nimi, josta parametrit ladataan.
        """
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.test_accuracy = data['test_accuracy']
        self.hidden_size = int(data['hidden_size'])
        self.learning_rate = float(data['learning_rate'])
        self.epochs = int(data['epochs'])
        self.batch_size = int(data['batch_size'])

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
    main(hidden_size=30, learning_rate=0.5, epochs=10, batch_size=32)
