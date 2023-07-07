import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
df = pd.read_csv(r'C:\Users\parul\OneDrive\Desktop\TCD\Final Year project\Womens Clothing E-Commerce Reviews.csv')
df = df.fillna('NA')
df['sentiment'] = df['Review Text'].apply(get_sentiment)
def get_sentiment_label(score):
    if score > 0.5:
        return 'positive'
    elif score < -0.5:
        return 'negative'
    else:
        return 'neutral'
df['sentiment'] = df['sentiment'].apply(get_sentiment_label)
X = df['Review Text']
y = df['sentiment']
vectorizer = CountVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)
clf = MultinomialNB()
clf.fit(X_vect, y)
st.title('Sentiment Analysis')
st.write('Enter some text to get the sentiment and associated features.')
input_text = st.text_area('Input text here:', value='', height=200)
if input_text:
    sentiment = get_sentiment(input_text)
    st.write('Sentiment:', sentiment)

    if sentiment > 0.3:
        st.write('The sentiment of the input is positive.')
    elif -0.1 < sentiment < 0.3:
        st.write('The sentiment of the input is neutral.')
    else:
        st.write('The sentiment of the input is negative.')
    input_vect = vectorizer.transform([input_text])
    pred = clf.predict(input_vect)[0]
## to run stremlit, write this command in the terminal : streamlit run webapp.py
    