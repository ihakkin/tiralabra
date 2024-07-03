# Toteutusdokumentti

## Ohjelman yleisrakenne

Neuroverkko luokittelee MNIST-tietokannan käsinkirjoitettuja numeroita. MNIST-tietokannassa on 60000 näytettä neuroverkon koulutukseen ja 10000 sen testaamiseen. MNIST-datan näytteet ovat 28x28 pikselin kokoisia harmaasävykuvia. Neuroverkko on toteutettu Pythonilla ja se käyttää numpy-kirjastoa matriisilaskentaan. NeuralNetwork-luokka sisältää verkon alustamisen, eteenpäin suuntautuvan laskennan ja vastavirta-algoritmin, kouluttamisen, luokittelutarkkuuden arvioinnin sekä arvojen tallennuksen käyttöliittymää varten.

Ensin alustetaan neuroverkon kerrokset sekä säädettävät hyperparametrit: piilokerroksen neuronien määrä, oppimisnopeus, epochien määrä ja minibatchien koko. Painot ja vakiotermit alustetaan satunnaisesti normaalijakauman arvoilla, jotka on skaalattu pienentämään suurten lukujen vaikutusta ja vakauttamaan oppimista. 

Eteenpäin suuntautuvassa laskennassa (forward propagation) syötetään näytedata neuroverkkon input-kerrokseen, jossa se kerrotaan painomatriisilla, summataan arvot ja lisätään summaan vakiotermi. Summa syötetään aktivaatiofunktioon ja siitä edelleen syötteeksi seuraavalle kerrokselle, jossa toistetaan sama prosessi. Piilokerroksen aktivaatiofunktiona käytetään sigmoidia ja output-kerroksessa softmax-funktiota, joka muuntaa summan todennäköisyysjakaumaksi eri luokkien välillä.

Vastavirta-algoritmissa (backpropagation) lasketaan virhe, eli ero ennustetun ja todellisen luokan välillä. Hukkafunktiona käytetään ristientropiaa, joka sopii hyvin sekä luokitteluongelmaan että yhdessä softmax-funktion kanssa käytettäväksi. Hukkafunktiota minimoidaan laskemalla gradientteja, jotka ilmaisevat miten painojen ja vakiotermien arvoja tulee muuttaa. Tämä tapahtuu ketjusäännön avulla, joka laskee osittaisderivaatat kerros kerrokselta taaksepäin lähtien output-kerroksesta. Lopuksi painot ja vakiotermit päivitetään vähentämällä niistä oppimisnopeuden ja gradientin tulo.

Neuroverkko koulutuksessa käytetään mini-batch gradient decent -menetelmää. Koulutusdata jaetaan pienempiin osiin (mini-batch), jotka syötetään neuroverkkoon matriiseina. Mini-batchin koko on säädettävä hyperparametri. Matriisi kuljetetaan enuroverkon läpi (forward porpagation), jonka jälkeen hukkafunktiota minimoidaan ja parametrit päivitetään (backpropagation). Tämä prosessi toistetaan jokaiselle mini-batchille. Kun kaikki mini-batchit on käsitelty, niin yksi epookki on valmis. Epookin lopuksi lasketaan neuroverkon luokittelutarkkuus testidatalla. Koulutusta jatketaan useiden epookkien ajan, jolloin luokittelutarkkuus paranee. Testidata ei ole ollut mukana koulutusprosessissa, joten se mittaa objektiivisesti neuroverkon yleistämiskykyä. 

Koulutuksen jälkeen hyperparametrit ja verkon parametrit tallennetaan erilliseen tiedostoon käyttöliittymää varten. Käyttöliittymässä on 1000 kuvan satunnaisesti valittu osio testidatasta.

Datan esikäsittely tapahtuu NeuralNetwork luokan ulkopuolella. Näytteiden pikseliarvot normalisoidaan ja kuvat muunnetaan 784x1 vektoreiksi.



## Aikavaativuudet sekä suorituskyky
Neuroverkon forward pass koostuu sarjasta matriisien kertolaskuja ja epälineaarisia aktivaatiofunktioita. Matriisin kertolaskun aikavaativuus on O(nmp), kun kerrottavat matriisit ovat dimensioiltaan n x m ja m x p. Jos yksinkertaistetaan asetelmaa siten, että käsitellään neliömatriiseja, niin aikavaativuus on O(n³). Aktivaatiofunktion laskeminen elementti elementiltä on O(n) operaatio. Jos oletetaan, että joka kerroksessa on yhtä monta neuronia ja kerrosten määrä on yhtä suuri kuin neuronien määrä kerroksessa, niin forward passin aikavaativuus on O(n⁴). Kurssilla toteutettu verkko on kovin pieni, ja kerrosten lukumäärä vain murto-osa neuronien lukumäärästä, joten ajattelen, että forward passin teoreettista aikavaativuutta kuvaa O(n⁴) sijaan paremmin O(nmpL), jossa L on kerrosten lukumäärä. Koska kerrosten määrä on niin pieni, niin voisi myös asettaa L vakioksi, ja ajatella aikavaativuuden O(n³). Matriisien kertolaskut on toteutettu optimoidulla NumPyn dot -funktiolla, jonka käytännön suorituskyky on erittäin hyvä. Käyttöliittymässä numeroiden luokittelu tapahtuu jo koulutetun verkon avulla forward proragation -funktiota kutsumalla.

Vastavirta-algoritmissa teoreettinen aikavaativuus on samankaltainen. Neuroverkon koulutuksessa train-funktiossa on kaksi sisäkkäistä silmukkaa. Sisemmässä silmukassa aikavaativuus on riippuvainen mini-batchien koosta ja niiden määrästä per epookki. Yhden mini-batchin aikavaativuus on O(batch_size x nmp) ja mini-batchien määrä per epookki on len(training_data)/batch_size. Nämä yhdistämällä saadaan yhden epookin aikavaativuus O(len(training_data) x  nmp). Ulompi silmukka iteroi epookkeja, joten kokonaisaikavaativuus koulutukselle on tulkintani mukaan O(epochs x len (training_data) x nmp). Koulutuksen aikavaativuus on siis lineaarisesti riippuvainen verkon arkkitehtuurin lisäksi epookkien määrästä ja koulutusdatan koosta. Kouluttaminen onkin huomattavasti hitaampaa kuin pelkkä luokittelu.



## Puutteet ja parannusehdotukset

Toteutuksessa voisi miettiä yleistystä muihin datasetteihin ja luokitteluongelmiin. Muokkasin yhdessä vaiheessa toteutusta kirjastomaisemmaksi, jolloin esim. verkon kerrosten määrää ja kokoa, sekä aktivaatiofunktioita olisi helpompi tarvittaessa vaihtaa. Päädyin kuitenkin tähän yksinkertaisempaan ratkaisuun, sillä se sopi tähän kyseiseen luokitteluongelmaan ja on mielestäni helppolukuinen.
Jos rakentaisin nyt projektin alusta, niin ottaisin käyttöön validointijoukon ja panostaisin hyperparametrien optimoinnin automatisointiin. Verkon oppimisen ja ylisovittamisen tutkiminen oli mielenkiintoista. Cost-funktion arvon voisi myös laskea erikseen, jotta sitä voisi hyödyntää ylisovittamisen tutkimisessa. Nyt tutkin jonkin verran regularisointia, mutta olisi myös mielenkiintoista implementoida learning rate decay ja early stopping -mekanismeja.


## Laajojen kielimallien käyttö

Käytin työskentelyn tukena ChatGPT:a. Käytin sitä projektin aluksi apuna vastavirta-algoritmin matematiikan ymmärtämiseen ja uusien käsitteiden avaamiseen. Tässä täytyi tietenkin noudattaa tiettyä varovaisuutta ja kriittisyyttä, mutta koen että siitä oli paljon apua oppimiseen. Muuta apua hain erilaisten ongelmatilanteiden ratkomiseen esim. virtuaaliympäristön pystyttämisessä, import-ongelmissa ja ongelmissa testikattavuusraportin tietojen hakemisessa. Näissä kielimallin tarjoamat ratkaisut toimivat vaihtelevalla menestyksellä, mutta auttoivat ainakin alkuun etten jäänyt jumiin ongelmien kanssa.

## Viitteet

 https://tim.jyu.fi/view/143092#VrMhIvLNy2PU

 http://neuralnetworksanddeeplearning.com/

 https://www.parasdahal.com/softmax-crossentropy

 https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

 https://lunalux.io/introduction-to-neural-networks/computational-complexity-of-neural-networks/
