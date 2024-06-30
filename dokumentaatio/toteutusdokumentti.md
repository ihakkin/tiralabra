# Toteutusdokumentti

## Ohjelman yleisrakenne

Neuroverkko luokittelee MNIST-tietokannan käsinkirjoitettuja numeroita. MNIST-tietokannassa on 60000 näytettä neuroverkon koulutukseen ja 10000 sen testaamiseen. MNIST-datan näytteet ovat 28x28 pikselin kokoisia harmaasävykuvia. Neuroverkko on toteutettu Pythonilla ja se käyttää numpy-kirjastoa matriisilaskentaan. NeuralNetwork-luokka sisältää verkon alustamisen, eteenpäin suuntautuvan laskennan ja vastavirta-algoritmin, kouluttamisen, luokittelutarkkuuden arvioinnin sekä arvojen tallennuksen käyttöliittymää varten.

Neuroverkko alustetaan määrittelemällä säädettävät hyperparametrit: piilokerroksen neuronien määrä, oppimisnopeus, epochien määrä ja minibatchien koko. Painot ja vakiotermit alustetaan satunnaisesti normaalijakauman arvoilla, jotka on skaalattu pienentämään suurten lukujen vaikutusta ja vakauttamaan oppimista. Halusin erotella hyperparametrit, joita säätämällä verkon toimintaa voi optimoida tässä nimenomaisessa luokitteluongelmassa.

Eteenpäin suuntautuvassa laskennassa (forward propagation) syötetään näytedata neuroverkkon input-kerrokseen, jossa se kerrotaan painomatriisilla, summataan arvot ja lisätään summaan vakiotermi. Summa syötetään aktivaatiofunktioon ja siitä edelleen syötteeksi seuraavalle kerrokselle, jossa toistetaan sama prosessi. Piilokerroksen aktivaatiofunktiona käytetään sigmoidia ja output-kerroksessa softmax-funktiota, joka muuntaa summan todennäköisyysjakaumaksi eri luokkien välillä.

Vastavirta-algoritmissa (backpropagation) lasketaan virhe, eli ero ennustetun ja todellisen luokan välillä. Hukkafunktiona käytetään ristientropiaa, joka sopii hyvin sekä luokitteluongelmaan että yhdessä softmax-funktion kanssa käytettäväksi. Hukkafunktiota minimoidaan laskemalla gradientteja, jotka ilmaisevat miten painojen ja vakiotermien arvoja tulee muuttaa. Tämä tapahtuu ketjusäännön avulla, joka laskee osittaisderivaatat kerros kerrokselta taaksepäin lähtien output-kerroksesta. Lopuksi painot ja vakiotermit päivitetään vähentämällä niistä oppimisnopeuden ja gradientin tulo.

Neuroverkko koulutuksessa käytetään mini-batch gradient decent -menetelmää. 60000:n näytteen koulutusdata jaetaan pienempiin osiin (mini-batch), jotka syötetään neuroverkkoon matriiseina. Mini-batchin koko on säädettävä hyperparametri. Matriisi kuljetetaan enuroverkon läpi (forward porpagation), jonka jälkeen hukkafunktiota minimoidaan ja parametrit päivitetään (backpropagation). Tämä prosessi toistetaan jokaiselle mini-batchille. Kun kaikki mini-batchit on käsitelty, niin yksi epookki on valmis. Epookin lopuksi lasketaan neuroverkon luokittelutarkkuus testidatalla. Koulutusta jatketaan useiden epookkien ajan, jolloin luokittelutarkkuus paranee. Testidata ei ole ollut mukana koulutusprosessissa, joten se tarjoaa objektiivisen mittarin neuroverkon yleistämiskyvylle. 

Hyperparametrit ja verkon parametrit tallennetaan koulutuksen jälkeen erilliseen tiedostoon käyttöliittymää varten. Käyttöliittymässä on 1000 kuvan satunnaisesti valittu osio testidatasta.

Datan esikäsittely tapahtuu Neuralnetwork luokan ulkopuolella. Näytteiden pikseliarvot normalisoidaan ja kuvat muunnetaan 784x1 vektoreiksi.



## Aikavaativuudet sekä suorituskyky



## Puutteet ja parannusehdotukset

Toteutuksessa voisi miettiä yleistystä muihin datasetteihin ja luokitteluongelmiin. Muokkasin yhdessä vaiheessa toteutusta kirjastomaisemmaksi, jolloin esim. verkon kerrosten määrää ja kokoa, sekä aktivaatiofunktioita olisi helpompi tarvittaessa vaihtaa. Päädyin kuitenkin tähän yksinkertaisempaan ratkaisuun, sillä se sopi tähän kyseiseen luokitteluongelmaan ja on mielestäni helppolukuinen.
Jos rakentaisin nyt projektin alusta, niin ottaisin käyttöön validointijoukon ja panostaisin hyperparametrien optimointiin. Verkon oppimisen ja ylisovittamisen tutkiminen oli mielenkiintoista. Cost-funktion arvon voisi myös laskea erikseen, jotta sitä voisi hyödyntää ylisovittamisen tutkimisessa. 


## Laajojen kielimallien käyttö

Käytin työskentelyn tukena ChatGPT:a. Käytin sitä projektin aluksi apuna vastavirta-algoritmin matematiikan ymmärtämiseen ja uusien käsitteiden avaamiseen. Tässä täytyi tietenkin noudattaa tiettyä varovaisuutta ja kriittisyyttä, mutta koen että siitä oli paljon apua oppimiseen. Muuta apua hain erilaisten ongelmatilanteiden ratkomiseen esim. virtuaaliympäristön pystyttämisessä, import-ongelmissa ja ongelmissa testikattavuusraportin tietojen hakemisessa. Näissä kielimallin tarjoamat ratkaisut toimivat vaihtelevalla menestyksellä, mutta auttoivat ainakin alkuun etten jäänyt jumiin ongelmien kanssa.

## Viitteet

 https://tim.jyu.fi/view/143092#VrMhIvLNy2PU

 http://neuralnetworksanddeeplearning.com/

 https://www.parasdahal.com/softmax-crossentropy

 https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

 https://lunalux.io/introduction-to-neural-networks/computational-complexity-of-neural-networks/
