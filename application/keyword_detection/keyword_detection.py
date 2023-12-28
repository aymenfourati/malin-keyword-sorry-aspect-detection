
# Ce module est utilisé pour détecter les mots clés dans une phrase donnée.
# Il prend une liste de mots clés et une phrase en entrée et renvoie une liste de mots clés qui sont présents dans la phrase.
# Nous utilisons la similarité cosinus pour calculer la similarité entre la phrase et les mots clés.
# pour le preprocessing, nous utilisons la tokenization, la supression des liaisons, la suppression de la ponctuation, la suppression des stop words et le stemming.

# Nous utilisons la libraire scikit-learn pour la vectorization avec TF-IDF et le calcul de la similarité cosinus.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Nous utilisons la librairie langid pour identifier la langue de la phrase et appliquer le preprocessing en fonction de la langue.
import langid

# Nous utilisons la librairie nltk pour le preprocessing 
from nltk.tokenize import casual
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# On definie les seuils de similarité cosinus pour l'anglais et le français.
THRESHOLD_ENGLISH = 0.11
THRESHOLD_FRENCH = 0.2

# On definie la fonction qui permet d'identifier la langue de la phrase.
def identify_language(text):
    # identifier la langue
    lang, _ = langid.classify(text)
    if lang =='fr': return 'french'
    if lang =='en': return 'english'
    return lang

# On definie la fonction qui permet de supprimer les liaisons
def remove_contracted_forms(input_text):
    # Define la list des liaisons pour la langue française
    contracted_forms = ["d'", "l'","m'", "s'", "t'", "j'", "c'", "n'", "qu'", "jusqu'", "quoiqu'", "lorsqu'", "puisqu'", "parce qu'"]

    # Supprimer les liaisons
    for contracted_form in contracted_forms:
        input_text = input_text.replace(contracted_form, '')

    return input_text

# On definie la fonction qui permet de prétraiter le texte.
def preprocess_text(text, language='english'):

    # Supprimer les caractères de liaison pour la language française
    text = remove_contracted_forms(text) if language == 'french' else text


    # Tokenize le texte en utilisant un tokenizer pour l'anglais ou le français
    tokenizer = casual.TweetTokenizer() if language == 'english' else casual.TweetTokenizer('french')
    tokens = tokenizer.tokenize(text)

    # Supprimer la ponctuation et convertir en minuscules
    tokens = [token.lower() for token in tokens if token.isalpha()]

    # Supprimer les stop words en fonction de la langue
    stop_words = set(stopwords.words(language))
    tokens = [token for token in tokens if token not in stop_words]

    # Appliquer le stemming 
    stemmer = SnowballStemmer(language)
    tokens = [stemmer.stem(token) for token in tokens]

    # Joindre les tokens pour former une chaîne de texte prétraitée
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# On definie la fonction qui permet de détecter les mots clés.
def keyword_detection(keywords, sentence):
    # Identify the language of the sentence
    language = identify_language(sentence)


    results = []
    # Initialiser le TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Preprocessing de la phrase
    preprocessed_sentence = preprocess_text(sentence, language=language)

   
    
    for keyword in keywords:
        # Preprocessing des mots clés
        preprocessed_keyword = preprocess_text(keyword, language=language)
        
        # Vectorization des mots clés et de la phrase
        sentence_tfidf = vectorizer.fit_transform([preprocessed_sentence, preprocessed_keyword])
        
        # Calcul de la similarité cosinus
        similarity = cosine_similarity(sentence_tfidf)

        # Définir le seuil de similarité en fonction de la langue
        threshold = THRESHOLD_ENGLISH if language == 'english' else THRESHOLD_FRENCH
        
        # Ajouter le mot clé à la liste des résultats si la similarité cosinus est supérieure au seuil
        results.append(keyword) if similarity[1, 0] > threshold else None


    return results

