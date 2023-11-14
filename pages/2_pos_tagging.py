import nltk
import streamlit as st

nltk.download('punkt')
nltk.download('treebank')
nltk.download('universal_tagset')

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
        'IN': 'lightyellow',  # Adposition
        'CC': 'lightorange',  # Conjunction
        'DT': 'lightpurple',  # Determiner
        'UH': 'lightcyan',  # Interjection
        'CD': 'lightmagenta',  # Numeral
        'RP': 'lightolive',  # Particle
        'SYM': 'lightbrown',  # Symbol
        'O': 'lightviolet'  # Other
    }

    highlighted_html = ""
    legend_html = ""
    for word, pos_tag in tagged_words:
        background_color = pos_background_colors.get(pos_tag[:2], 'transparent') if pos_tag else 'transparent'
        highlighted_html += f'<span style="background-color: {background_color};">{word}</span> '

        # Build legend
        if pos_tag:
            legend_html += f'{nltk.tag.mapping.map_tag("en-ptb", "universal", pos_tag)} '

    return highlighted_html, legend_html

st.title('Part-of-Speech Tagging for Sentiment Analysis')
st.write('Part-of-Speech tagging')

# Input text box
sentence = st.text_area("Enter a sentence:")
if sentence:
    # Perform part-of-speech tagging
    tagged_words = pos_tagging(sentence)

    # Highlight words with background colors and generate legend
    highlighted_html, legend_html = highlight_tagged_words(tagged_words)
    # Display the result with highlighted words and background colors
    st.subheader("Tagged Sentence:")
    st.markdown(highlighted_html, unsafe_allow_html=True)
    # Display legend
    st.subheader("Legend")
    st.markdown(f"**{legend_html}**")
