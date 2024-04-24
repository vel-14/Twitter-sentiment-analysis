#importing all the necessary libraries

import streamlit as st 
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#downloading the stop words
nltk.download('stopwords')

#setting the title and background for the ui
st.set_page_config(page_title="Twitter Sentiment Analysis",
                   page_icon='/Users/velmurugan/Desktop/velu/python_works/twitter_sentiment_analysis/x.png.webp')

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                background: linear-gradient(to bottom, #483D8B, #ADD8E6);
            }}
           </style>""",
        unsafe_allow_html=True)

setting_bg()


#defining stemming function
portstem = PorterStemmer()

def stemming(tweet_post):
    stmmed_tweet = re.sub('[^ a-zA-Z]',' ',tweet_post).lower().split()
    stmmed_tweet = [ portstem.stem(word) for word in stmmed_tweet if not word in stopwords.words('english')]
    return (' ').join(stmmed_tweet)


#loading the fitted vectorizer and defining function for preprocessing 
with open('/Users/velmurugan/Desktop/velu/python_works/twitter_sentiment_analysis/vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)


def preprocessing(tweet_post):
    stmed_twt = stemming(tweet_post)
    vectorized_tweet = vectorizer.transform([stmed_twt])
    return vectorized_tweet


#loading the trained model
with open('/Users/velmurugan/Desktop/velu/python_works/twitter_sentiment_analysis/model.pkl','rb') as f:
    loaded_model = pickle.load(f)


#aligning the columns in ui
title_column, logo_column = st.columns([1, 0.3])

# Title in the left column
with title_column:
    st.markdown("<p style='font-size:38px; color:Darkgrey; text-align:right; font-weight:bold;'>Twitter Sentiment Analysis</p>",
                 unsafe_allow_html=True)

#logo gif 
with logo_column:
    st.image('/Users/velmurugan/Desktop/velu/python_works/twitter_sentiment_analysis/CLIPLY_372109260_TWITTER_LOGO_400.gif')


# predicting the tweeted post
    
tweet_post = st.text_area('Enter Tweet',height=200)

if st.button('Predict emotional polarity'):

    preprocessed_tweet = preprocessing(tweet_post)

    prediction = loaded_model.predict(preprocessed_tweet)

    if prediction[0] == 1:
        st.success('Positive Tweet')
    else:
        st.error('Negetive Tweet')



#sample tweets for the user

st.markdown("<p style='font-size:28px; color:Gold; font-weight:bold;'>Sample Tweets</p>", 
            unsafe_allow_html=True)

samples = ['Overjoyed to announce I passed with flying colors! 🎉 , a testament to dedication and perseverance! #SuccessStory',
           'Grateful for the opportunity to bring characters to life on screen  🎥✨  #Blessed #Actress',
           'Another crushing defeat for our team today #Disappointed #Underperforming']

for index,tweets in enumerate(samples):
    st.write(f"{index+1}.{tweets}")




    




