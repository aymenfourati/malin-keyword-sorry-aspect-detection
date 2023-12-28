
# Ce module est utlisé pour classifier l'aspect d'excuse dans une phrase donnée.
# Il prend une phrase en entrée et renvoie 1 si l'aspect d'excuse est présent dans la phrase et 0 sinon.
# On definit deux methodes pour la detection de l'aspect d'excuse:

# 1) Model based: Nous utilisons un modèle LSTM entrainé sur un dataset de phrases annotées manuellement.
# les details de l'entrainement du modèle sont disponibles dans le notebook LSTM_apology_detection.ipynb

# 2) Rule based: Nous utilisons une liste de mots clés pour détecter l'aspect d'excuse dans une phrase donnée.
# Il s'avere que cette methode est plus efficace que la methode model based puisque les mots utilisé pour exprimer l'aspect de l'excuse sont très limités.
# Cette methode ce base sur la recherche forme de base des mots clés dans la phrase.
# Pour ce faire, nous implementons le stemming pour identidier les formes de base des mots dans la phrase et dans la liste des mots clés.




# Nous utilisons la librairie keras pour charger le modèle et la librairie pandas pour le preprocessing des données.
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd


# Nous chargeons le modèle LSTM entrainé.
loaded_model = load_model('/application/application/sorry_aspect/LSTM_apology_detection.keras')



APOLOGY_THRESHOLD = 0.72

# Nous utilisons la librairie nltk pour le preprocessing de la même manière que dans le module keyword_detection/keyword_detection.py
# Pour eviter de créer un couplage fort entre les deux modules, nous avons copié le code du module keyword_detection/keyword_detection.py dans ce module.
from nltk.tokenize import casual
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def preprocess_text(text, language='english'):


    tokenizer = casual.TweetTokenizer() 
    tokens = tokenizer.tokenize(text)


    tokens = [token.lower() for token in tokens if token.isalpha()]


    stop_words = set(stopwords.words(language))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = SnowballStemmer(language)
    tokens = [stemmer.stem(token) for token in tokens]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# Rule based method
def common_check(sentence):
    apology_words = ["sorry", "apology", "regret", "forgive", "excuse"]

    for word in apology_words:
        if preprocess_text(word) in preprocess_text(sentence):
            return 1
    return 0


# Model based method
def predict(input):

    # Nous utilisons le modèle LSTM entrainé pour prédire l'aspect d'excuse dans la phrase donnée.
    data = {"response":[f"{input}"]}
    test = pd.DataFrame(data)

    # Les valeurs de max_words et max_len doivent être les mêmes que celles utilisées lors de l'entrainement du modèle.
    max_words = 341
    max_len = 41

    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(test["response"])
    sequences = tok.texts_to_sequences(test["response"])
    txts = sequence.pad_sequences(sequences, maxlen=max_len, padding='post')
    preds = loaded_model.predict(txts)
    preds_=[ 1 if APOLOGY_THRESHOLD<j else 0 for i,j in preds ] 
    return preds_[0]


