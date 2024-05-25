Viikko 2

Ma: 3h Teoriaa
Ti: 5h Kirjan (http://neuralnetworksanddeeplearning.com) algoritmin läpikäyntiä
Ke: 3h Ongelmia virtuaaliympäristön ja umporttauksen kanssa. Päädyin pystyttämään uudelleen projektin.
To: 4h Toteutin kirjan algoritmin pohjalta matriisitoteutuksen, jossa algoritmi saa inputvektorin sijaan mini-batch -matriisin. Matriisitoteutus kyllä nopeutti algoritmia huomattavasti, mutta lopulta muutoksia kirjan algoritmiin tuli yllättävän vähän, joten tajusin että pelkkä muokkaaminen matriisipohjaiseksi ei riitä, vaan pitää toteuttaa koko takaisinvirtausalgoritmi itse.
Pe: 5h Pseudokoodin etsintää ja takaisinvirtausalgoritmin kirjoitusta. 
La: 10h Yritin tehdä yksinkertaisen toteutuksen, joka on jaettu pieniin funktioihin. Luokkaan kapselointi olisi ehkä ollut järkevämpää, kuten kirjan esimerkissä. Verkon koko on nyt hyvin pieni: input-layer, pieni hidden layer ja output-layer. Aktivaatiofunktioina on nyt relu ja outputissa softmax, ja virheen kvantifioinnissa cross entropy. Tarkkuus nousee lähelle 90% kun epocheja on tarpeeksi, mutta lähtee todella matalalta ja nousee jokseenkin hitaasti. Pitää perehtyä vielä optimointiin tarkemmin, nyt halusin saada edes jokseenkin toimivaa. Ehkä SGD auttaisi.
Testejä en päässyt vieläkään aloittamaan. Tuntuu että tässä aiheessa testien tekeminen heti alussa samaa vauhtia metodien kanssa on haastavaa, koska aihe itsessään on haastava. 