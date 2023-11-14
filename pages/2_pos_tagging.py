import nltk
import streamlit as st

nltk.download('punkt')
nltk.download('treebank')

# Load the tagged sentences for training
treebank_corpus = nltk.corpus.treebank.tagged_sents()
train_sents = treebank_corpus[:3000]

# Train the UnigramTagger
tagger = nltk.tag.UnigramTagger(train_sents)

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

    legend_html = "Legend: "
    highlighted_html = ""
    for word, pos_tag in tagged_words:
        background_color = pos_background_colors.get(pos_tag[:2], 'transparent') if pos_tag else 'transparent'
        highlighted_html += f'<span style="background-color: {background_color};">{word}</span> '
        
    for pos_tag, color in pos_background_colors.items():
        legend_html += f'<span style="background-color: {color}; margin-right: 10px;">{pos_tag}</span> '

    return legend_html, highlighted_html

st.title('Part-of-Speech Tagging for Sentiment Analysis')
st.write('Part-of-Speech tagging')

# Input text box
sentence = st.text_area("Enter a sentence:")
if sentence:
    # Perform part-of-speech tagging
    tagged_words = pos_tagging(sentence)

    # Highlight words with background colors and display legend
    legend_html, highlighted_html = highlight_tagged_words(tagged_words)
    st.markdown(legend_html, unsafe_allow_html=True)
    st.markdown(highlighted_html, unsafe_allow_html=True)
