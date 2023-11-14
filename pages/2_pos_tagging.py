import nltk
import streamlit as st

nltk.download('punkt')
# nltk.download('treebank')
from nltk.tag import UnigramTagger

# Load the tagged sentences for training
train_sents = nltk.corpus.treebank.tagged_sents()[:3000]

# Train the UnigramTagger
tagger = UnigramTagger(train_sents)

def pos_tagging(sentence):
    # Tokenize the input sentence
    words = nltk.word_tokenize(sentence)

    # Perform part-of-speech tagging
    tagged_words = tagger.tag(words)

    return tagged_words

def highlight_tagged_words(tagged_words):
    pos_background_colors = {
        'NN': 'lightcoral',
        'VB': 'lightgreen',
        'JJ': 'lightskyblue',
        'RB': 'lightpink',
        'PR': 'lightgray',
    }

    highlighted_html = ""
    for word, pos_tag in tagged_words:
        background_color = pos_background_colors.get(pos_tag[:2], 'transparent') if pos_tag else 'transparent'
        highlighted_html += f'<span style="background-color: {background_color};">{word}</span> '

    return highlighted_html

st.title('Part-of-Speech Tagging for Sentiment Analysis')
st.write('Part-of-Speech tagging')

# Input text box
sentence = st.text_area("Enter a sentence:")

# Button to trigger tagging
if st.button("Tag and Highlight"):
    # Perform part-of-speech tagging
    tagged_words = pos_tagging(sentence)

    # Highlight words with background colors
    highlighted_html = highlight_tagged_words(tagged_words)

    # Display the result with highlighted words and background colors
    st.markdown(highlighted_html, unsafe_allow_html=True)

    # Display the legend
    st.write("Legend:")
    st.write("- Noun (NN): lightcoral")
    st.write("- Verb (VB): lightgreen")
    st.write("- Adjective (JJ): lightskyblue")
    st.write("- Adverb (RB): lightpink")
    st.write("- Pronoun (PR): lightgray")
