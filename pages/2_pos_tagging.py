import nltk
import streamlit as st
from nltk.corpus import treebank
from nltk.tag import UnigramTagger
nltk.download('treebank')

# Load the tagged sentences for training
treebank_corpus = treebank.tagged_sents()
train_sents = treebank_corpus[:3000]

# Train the UnigramTagger
tagger = UnigramTagger(train_sents)

def pos_tagging(sentence):
    # Tokenize the input sentence
    words = nltk.word_tokenize(sentence)

    # Perform part-of-speech tagging
    tagged_words = tagger.tag(words)

    return tagged_words

st.title('Part-of-Speech Tagging for Sentiment Analysis')
st.write('Part-of-Speech tagging')

# Input text box
sentence = st.text_area("Enter a sentence :")
if sentence:
    # Perform part-of-speech tagging
    tagged_words = pos_tagging(sentence)

