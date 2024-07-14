#sentiment analysis.py

from textblob import TextBlob
import streamlit as st
from wordcloud import WordCloud
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
from itertools import islice
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

st.header("Sentiment Analysis")

# Upload CSV file and read as pandas dataframe
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    string_data = uploaded_file.getvalue().decode("ISO-8859-1")
    stringio = StringIO(string_data)
    dataframe = pd.read_csv(stringio)

    # Select column to analyze
    column_name = st.sidebar.selectbox('Select column', dataframe.columns)

    if st.sidebar.checkbox("Show data"):
        st.dataframe(dataframe)

    with st.expander("Analyze Text"):
        texts = dataframe[column_name].dropna()
        if not texts.empty:
            texts_str = ' '.join(texts)
            blob = TextBlob(texts_str)
            st.write('polarity:', round(blob.sentiment.polarity, 2))
            st.write('subjectivity:', round(blob.sentiment.subjectivity, 2))

            # Clean text and apply word cloud
            stop_words = set(stopwords.words('english'))
            cleaned_text = ' '.join([word for word in blob.words if word.lower() not in stop_words])

            # Generate word cloud for all words
            wordcloud = WordCloud().generate(cleaned_text)
            st.write('word cloud for all text')
            st.image(wordcloud.to_array())

            # Generate word cloud for positive words
            positive_words = [word for word in blob.words if TextBlob(word).sentiment.polarity > 0]
            positive_text = ' '.join(positive_words)
            positive_wordcloud = WordCloud().generate(positive_text)
            st.write('word cloud for all positive words')
            st.image(positive_wordcloud.to_array())
            st.write('Positive words count:', len(positive_words))

            # Generate word cloud for negative words
            negative_words = [word for word in blob.words if TextBlob(word).sentiment.polarity < 0]
            negative_text = ' '.join(negative_words)
            negative_wordcloud = WordCloud().generate(negative_text)
            st.write('word cloud for all negative words')
            st.image(negative_wordcloud.to_array())
            st.write('Negative words count:', len(negative_words))

            word_freq = blob.word_counts
            sorted_word_freq = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)}
            top_5_word_freq = islice(sorted_word_freq.items(), 5)
            st.write('Top 5 Word Frequencies:')
            for word, freq in top_5_word_freq:
                st.write(f'{word}: {freq}')
        else:
            st.write('No text data found in selected column.')



#If the polarity of a word is 0.13 in a word cloud, it suggests that the sentiment or emotional tone associated with that word is slightly positive.
#Polarity typically ranges from -1 to 1, where values closer to 1 indicate a positive sentiment and values closer to -1 indicate a negative sentiment.
#In this case, a polarity of 0.13 suggests a mild positive sentiment associated with the word in the context of the word cloud.