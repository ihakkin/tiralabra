# Tiralabra: Neuroverkko

Harjoitustyön aiheena on MNIST-tietokannan käsinkirjoitettujen numeroiden tunnistus neuroverkon avulla. Projektilla on yksinkertainen graafinen käyttöliittymä, jossa käyttäjä voi testata neuroverkon luokittelun onnistumista 1000 näytteen randomisoidulla testidatasetillä. Neuroverkon luokittelutarkkuus on tällä hetkellä n. 97%.

## Asennusohjeet

### 1. Kloonaa repositorio ja siirry sen juurikansioon

```bash
git clone https://github.com/ihakkin/tiralabra.git
cd tiralabra
mkdir data
```
### 2. Lataa MNIST-data 

[lataa](https://www.dropbox.com/scl/fo/jzbrj0pikmnos8ww93pje/AKGDpmv7gVMBiRk6zgSR0A8?rlkey=nc55rx8s1ypfe8885i4yhks0l&st=2l4ytkz4&dl=0) mnist_train.csv ja mnist_test.csv kansioon tiralabra/data


### 3. Asenna poetry jos sitä ei vielä ole asennettu

[ohjeita](https://algolabra-hy.github.io/poetry)


### 4. Asenna riippuvuudet 

```bash
poetry install
```

### 5. Aktivoi virtuaaliympäristö

```bash
poetry shell
```

### 6. Kouluta verkko tai testaa suoraan käyttöliittymää aiemmin tallennetuilla parametreilla

```bash
cd src
```
#### Kouluta ja tallenna uudet parametrit sekä senhetkinen tarkkuus: 
```bash
python3 nn.py
```

#### Testaa käyttöliittymää
```bash
python3 app.py
```

### 7. Aja testit

```bash
coverage run -m pytest
```