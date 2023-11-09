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

def highlight_tagged_words(tagged_words):
    # Define colors for different parts of speech
    pos_colors = {
        'NN': 'red',
        'VB': 'blue',
        'JJ': 'green',
        # Add more POS tags and colors as needed
    }

    # Create HTML with inline styles for highlighting
    html = ""
    for word, pos_tag in tagged_words:
        color = pos_colors.get(pos_tag[:2], 'black')  # Default to black if not defined
        html += f'<span style="color:{color};">{word}</span> '

    return html

st.title('Part-of-Speech Tagging for Sentiment Analysis')
st.write('Part-of-Speech tagging')

# Input text box
sentence = st.text_area("Enter a sentence:")
if sentence:
    # Perform part-of-speech tagging
    tagged_words = pos_tagging(sentence)

    # Highlight words with colors
    highlighted_html = highlight_tagged_words(tagged_words)

    # Display the result with highlighted words
    st.markdown(highlighted_html, unsafe_allow_html=True)