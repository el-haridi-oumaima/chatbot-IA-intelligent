# EL HARIDI OUMAIMA - ETTABTI ZAYNAB - LAAMIAR SALAMA
import streamlit as st
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import RegexpParser
from nltk import ne_chunk
from nltk.tree import Tree
from rapidfuzz import fuzz

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Fonction pour charger la base de connaissances depuis un fichier JSON
def charger_base_connaissances():
    with open('baseConnaissance.json', 'r') as f:
        base_connaissances = json.load(f)
    return base_connaissances

# Prétraitement amélioré avec étiquetage POS
def pretraiter_question(question):
    """
    Prétraiter la question de l'utilisateur :
    - Conversion en minuscules
    - Suppression de la ponctuation
    - Suppression des mots vides
    - Étiquetage POS et conservation des mots importants (noms, verbes, adjectifs, adverbes)
    """
    # Convertir la question en minuscules
    question = question.lower()
    
    # Supprimer la ponctuation
    question = question.translate(str.maketrans('', '', string.punctuation))
    
    # Tokeniser la phrase en mots
    mots = word_tokenize(question)
    
    # Supprimer les mots vides (en anglais)
    stop_words = set(stopwords.words('english'))
    mots = [mot for mot in mots if mot not in stop_words]
    
    # Appliquer l'étiquetage POS
    mots_tagges = pos_tag(mots)  # Retourne une liste de tuples (mot, POS)
    
    # Filtrer les mots pour ne garder que les noms (NN), verbes (VB), adjectifs (JJ), et adverbes (RB)
    mots_filtres = [mot for mot, tag in mots_tagges if tag.startswith(('N', 'V', 'J', 'R'))]
    
    # Rejoindre les mots filtrés dans une phrase
    question_pretraitee = ' '.join(mots_filtres)
    
    return question_pretraitee

# Fonction de chunking
def chunker(question):
    """
    Tokeniser et appliquer l'étiquetage POS sur la phrase, puis appliquer le chunking
     pour identifier les phrases nominales et verbales.
    """
    mots_tagges = pos_tag(word_tokenize(question))
    
    # Définir une grammaire simple de chunking
    grammar = """
    NP: {<DT>?<JJ>*<NN>}   # Phrase nominale
    VP: {<VB.*>}            # Phrase verbale
    """
    
    # Créer un parseur
    cp = RegexpParser(grammar)
    
    # Appliquer le parseur de chunking aux tokens étiquetés
    tree = cp.parse(mots_tagges)
    
    return tree

# Fonction d'extraction des entités nommées (NER)
def extraire_entities(question):
    """
    Extrait les entités nommées (noms de personnes, lieux, organisations, etc.) de la question.
    """
    mots_tagges = pos_tag(word_tokenize(question))
    arbre_ner = ne_chunk(mots_tagges)  # Applique le NER sur les mots taggés
    
    # Extraire les entités nommées de l'arbre
    entities = []
    for subtree in arbre_ner:
        if isinstance(subtree, Tree):  # Si c'est une entité nommée
            entity = " ".join(word for word, tag in subtree)
            entities.append(entity)
    
    return entities

# Fonction pour comparer les questions à l'aide de la correspondance floue
def comparer_question(question, base_connaissances, seuil=70):
    """
    Comparer la question de l'utilisateur aux motifs dans la base de connaissances
     à l'aide de la correspondance floue.
    """
    question = pretraiter_question(question)  # Prétraiter l'entrée utilisateur
    print("Question prétraitée:", question)  # Ajouter cette ligne pour déboguer
    meilleure_similarite = 0
    meilleure_reponse = "Sorry, I didn't understand."

    # Parcourir tous les motifs dans la base de connaissances
    for intention in base_connaissances['intents']:
        for motif in intention['patterns']:
            # Calculer la similarité
            similarite = fuzz.ratio(question, pretraiter_question(motif))

            print(f"Comparaison: {question} <-> {motif} | Similarité: {similarite}")  # Débogage

            # Vérifier si la similarité dépasse le seuil et est la meilleure correspondance
            if similarite > seuil and similarite > meilleure_similarite:
                meilleure_similarite = similarite
                meilleure_reponse = intention['responses'][0]

    return meilleure_reponse

# Fonction pour prétraiter et appliquer le chunking à la question
def pretraiter_question_avec_chunking_et_ner(question):
    """
    Prétraiter la question et appliquer le chunking et l'extraction des entités nommées.
    """
    question_pretraitee = pretraiter_question(question)
    
    # Appliquer le chunking
    chunked_tree = chunker(question_pretraitee)
    
    # Extraire les entités nommées
    entities = extraire_entities(question)
    
    return chunked_tree, entities

# Charger la base de connaissances
base_connaissances = charger_base_connaissances()

# Interface Streamlit
st.title("Welcome to ChaterBot")


# Ajouter une phrase de personnalisation sous le titre avec le thème IA
st.markdown("""
    <h6 style='text-align: left;'>I'm here to provide insights and knowledge about the fascinating world of AI!</h6>
""", unsafe_allow_html=True)

# Ajouter du CSS personnalisé pour colorer les messages
st.markdown("""
    <style>
    .user-message {
        background-color: #90EE90;  /* Fond vert pour l'utilisateur */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: black;  /* Texte en noir */
    }
    .chatbot-message {
        background-color: #D3D3D3;  /* Fond gris pour le chatbot */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: black;  /* Texte en noir */
    }
    </style>
""", unsafe_allow_html=True)

# Initialiser une variable d'état de session pour conserver l'historique des conversations
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Entrée utilisateur
input_utilisateur = st.text_input("You : ")

def afficher_message(message, is_user=True):
    """Afficher un message avec un style différent selon l'utilisateur ou le chatbot"""
    if is_user:
        st.markdown(f'<p class="user-message">{message}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="chatbot-message">{message}</p>', unsafe_allow_html=True)


if input_utilisateur:
    # Comparer la question à l'aide de la correspondance floue
    reponse = comparer_question(input_utilisateur, base_connaissances)
    
    # Appliquer le chunking et l'extraction NER à la question
    chunked_tree, entities = pretraiter_question_avec_chunking_et_ner(input_utilisateur)
    
    # Afficher l'arbre chunké pour le débogage
    print("Arbre Chunké:", chunked_tree)
    
    # Afficher les entités nommées extraites
    print("Entités Nommées:", entities)
    
    # Ajouter la question et la réponse à l'historique de la conversation
    st.session_state['conversation'].append(f"You : {input_utilisateur}")
    st.session_state['conversation'].append(f"ChaterBot : {reponse}")
    
   # Afficher l'historique de la conversation avec les styles personnalisés
    for message in reversed(st.session_state['conversation']):
        if message.startswith("You :"):
            afficher_message(message, is_user=True)
        else:
            afficher_message(message, is_user=False)