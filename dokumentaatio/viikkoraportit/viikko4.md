ma: Testien miettimistä ja testiluento.  4h

ti: Päivitin neuroverkon rakennetta. Löysin virheen relu-funktiosta. Lisäsin mini-batchit ja stochastic gradient descentin. En saanut aiemman gradien descent -mallin tarkkuutta nousemaan yli 93% optimointiyrityksistä huolimatta. Mini-batchien ja SGD:n myötä tarkkuus nousi n. 96 prosenttiin. Uusi toteutus on odotetusti hitaampi, mutta toisaalta epocheja ei tarvitse niin paljon kuin ennen. Kokeilin eri aktivaatiofunktioita. Näyttäisi että paras tarkkuus on leaky relulla ja sigmoidilla. Sigmoidin kanssa tarkkuus kasvaa tasaisemmin koulutuksessa. Leaky relulla tarkkuus vähän sahaa. Kokeilin myös erilaisia parametrien alustuksia. 6h

to: Päivitin testejä. Lisäsin testin, jossa katsotaan vaikuttaako mini-batchin samplejen järjestyksen sekoittaminen yksittäisten samplejen ennustukseen. 3h

pe: Lisää testejä. Testikattavuus nyt 85% ja kattaa kaikki funktiot, paitsi aktivaatiofunktiot, joiden testejä en ole vielä tehnyt. Docstringien ja koodin kommenttien kirjoittamista. 5h

la: Tein järkevästi isoja muutoksia koodiin ennen vertaisarviointiin palautusta. Yritän saada nopeasti kaiken toimimaan. Testidokumentti puuttuu vielä, mutta priorisoin nyt toimivaa versiota.