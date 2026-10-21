# MGB-2 : transcriptions alignées du split train (XML utf8)

Source : jeu de données `QCRI/mgb2` (Hugging Face), composante D (alignement léger),
2214 émissions Al Jazeera, fichiers `xml/utf8/*.xml` extraits de `train.tar.gz`.

Licence QCRI-ALT : usage de recherche non commercial, non transférable, données à protéger
contre toute divulgation. Ce dossier ne doit être partagé qu'avec les personnes ayant
elles-mêmes accepté la licence MGB-2. Toute œuvre dérivée doit reprendre la mention de la licence.

Format : `<segment starttime endtime who WMER PMER AWD>` contenant des `<element type="word">`.
Extraction : `python -m src.fetch_train_xml` (projet nlp-project).
