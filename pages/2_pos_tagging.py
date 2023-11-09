import nltk
import streamlit as st
from nltk.corpus import treebank
from nltk.tag import UnigramTagger

nltk.download('punkt')
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
    # Define background colors for different parts of speech
    pos_background_colors = {
        'NN': 'red',
        'VB': 'blue',
        'JJ': 'green',
        # Add more POS tags and colors as needed
    }

    # Create HTML with inline styles for background color
    html = ""
    for word, pos_tag in tagged_words:
        print(f"Word: {word}, POS Tag: {pos_tag}")
        background_color = pos_background_colors.get(pos_tag, 'transparent')  # Use full POS tag
        html += f'<span style="background-color:{background_color}; padding: 2px;">{word}</span> '

    return html

st.title('Part-of-Speech Tagging for Sentiment Analysis')
st.write('Part-of-Speech tagging')

# Input text box
sentence = st.text_area("Enter a sentence:")
if sentence:
    # Perform part-of-speech tagging
    tagged_words = pos_tagging(sentence)

    # Highlight words with background colors
    highlighted_html = highlight_tagged_words(tagged_words)

    # Display the result with highlighted words and background colors
    st.markdown(highlighted_html, unsafe_allow_html=True)