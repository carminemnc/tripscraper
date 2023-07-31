import streamlit as st
import pandas as pd
from streamlit_tags import *
import os
from charter import *


# scraper
import time
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By

# utils
import numpy as np,demoji
from datetime import datetime
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams,trigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from LeXmo import LeXmo

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline
from nltk.stem.wordnet import WordNetLemmatizer


# Settings & Functions

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

@st.cache_resource
def installff():
  os.system('sbase install geckodriver')
  os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/geckodriver /home/appuser/venv/bin/geckodriver')

_ = installff()


# Load models & vocabulary

@st.cache_resource
def load_absa_tokenizer():
    absa_tokenizer = AutoTokenizer.from_pretrained(
        "yangheng/deberta-v3-base-absa-v1.1")
    return absa_tokenizer

@st.cache_resource
def load_absa_model():
    absa_model = AutoModelForSequenceClassification \
        .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    return absa_model

@st.cache_resource
def load_zsc():
    zsc = pipeline(model="facebook/bart-large-mnli")
    return zsc

@st.cache_resource
def nltk_dwnld_stopwords():
    return nltk.download('stopwords')

absa_tokenizer = load_absa_tokenizer()
absa_model = load_absa_model()
zsc = load_zsc()
nltk_dwnld_stopwords()

stop_words = stopwords.words('english')

@st.cache_data
def hotelTripScraper(url,pages):
    
    # empty dataframe
    df = pd.DataFrame(columns=['review_title','review_rating','review_date','review_body','user_contributions'])
        

    options = FirefoxOptions()
    options.add_argument('--headless')

    brwsr = webdriver.Firefox(options=options)
    
    # get URL
    brwsr.get(url)
    
    # wait for DOM
    time.sleep(2)

    # handling cookies
    brwsr.find_element(By.XPATH,'//*[@id="onetrust-accept-btn-handler"]').click()
    
    for page in range(0,pages):
        
        for i in range(3,13):
            
            # review object
            review = brwsr.find_element(By.XPATH,f'/html/body/div[2]/div[2]/div[2]/div[9]/div/div[1]/div[1]/div/div/div[3]/div[{i}]')
            
            # review title
            title = review.find_element(By.XPATH,".//div[contains(@data-test-target, 'review-title')]").text
            
            # review rating
            rating = review.find_element(By.XPATH,".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
            
            
            # review date
            date = review.find_element(By.XPATH,".//span[contains(@class, 'teHYY _R Me S4 H3')]").text
            
            # expand review if it's needed
            try:
                review.find_element(By.XPATH,".//div[contains(@data-test-target, 'expand-review')]").click()
            except:
                pass
            finally:
                # review body
                body = review.find_element(By.XPATH,".//span[@class='QewHA H4 _a']").text
            
            
            user_contributions = review.find_element(By.XPATH,".//span[contains(@class, 'yRNgz')]").text
            
            # to a dataframe
            review_objects = [title,int(rating),date,body,user_contributions] 
            
            df.loc[len(df)] = review_objects
    

        brwsr.find_element(By.XPATH,'.//a[@class="ui_button nav next primary "]').click()
        
        time.sleep(3)
            

    
    return df

@st.cache_resource
def absa_predictor(sentence, word):

    sentence = sentence
    aspect = word
    inputs = absa_tokenizer(
        f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    labels = dict(zip(["negative", "neutral", "positive"], probs))
    out = max(labels, key=labels.get)
    
    return out


@st.cache_data
def absa_analyzer(sentence, word):

    sentence = sentence
    aspect = word
    inputs = absa_tokenizer(
        f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    df = pd.DataFrame({'labels':['negative','neutral','positive'],'probs':probs})
    
    return df


@st.cache_data
def zero_shot_predictor(data,labels):
    
    out = zsc(data,
    candidate_labels=labels)
    labels = dict(zip(out['labels'],out['scores']))
    out = max(labels, key=labels.get)
    
    return out


@st.cache_data
def zero_shot_analyzer(data,labels):
    
    out = zsc(data,
    candidate_labels=labels)
    df = pd.DataFrame({'labels':out['labels'],'score':out['scores']})
    
    return df

@st.cache_data
def dfCleaner(data):
    
    lemmatizer = WordNetLemmatizer()
    
    
    data['review_short_date'] = data['review_date'].replace('Date of stay: ','',regex=True)
    data['review_date'] = data['review_date'].replace('Date of stay: ','',regex=True).apply(lambda x: datetime.strptime(x,'%B %Y') )
    data['review_year'] = data['review_date'].apply(lambda x: x.strftime('%Y'))
    data['review_month'] = data['review_date'].apply(lambda x: x.strftime('%B'))
    
    
    data['review_rating'] = data['review_rating'].map({10: 'Terrible',
                                                   20: 'Poor',
                                                   30: 'Average',
                                                   40: 'Very Good',
                                                   50: 'Excellent'})
    
    
    # # review body
    # data['review_body_emojis'] = data['review_body'].apply(demoji.findall).apply(
    #     lambda x: '' if not list(x.keys()) else list(x.keys()))
    
    # # review title
    # data['review_title_emojis'] = data['review_title'].apply(demoji.findall).apply(
    #     lambda x: '' if not list(x.keys()) else list(x.keys()))
    # emojis = []
    # temp_ = [x for x in data['review_body_emojis'].to_list() if x]
    # temp_t = [x for x in data['review_title_emojis'].to_list() if x]
  
    # for item in temp_:
    #     emojis = emojis + item

    # for item in temp_t:
    #     emojis = emojis + item
        
    # emojis = pd.DataFrame.from_dict(Counter(emojis), orient='index').reset_index(
    # ).rename(columns={'index': 'emojis', 0: 'frequency'}).sort_values('frequency',ascending=False)
        
    
    
    def clean_text(text):
        
        text = re.sub(r'\n', ' ', text)  # Remove line breaks
        text = re.sub('@mention', ' ', text)  # remove mentions
        text = re.sub('{link}', ' ', text)  # remove links
        text = re.sub('Ûª', ' ', text)
        text = re.sub('  ', ' ', text)
        text = re.sub('RT', ' ', text)
        text = re.sub('//', ' ', text)
        text = re.sub('&quot', ' ', text)
        text = re.sub('&amp', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.lower()
        
        return text
    
    def tokenize_text(data):
        
        tokens = word_tokenize(data)
        tokens_without_sw = [word for word in tokens if not word in stop_words]
        tokens_without_sw = [word for word in tokens_without_sw if len(word) > 2]
        text = tokens_without_sw
        
        return text
    
    def sentiment_analysis(text):
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(text)
        
        if score['compound'] > 0:
            out = 'positive'
        if score['compound'] < 0:
            out= 'negative'
        if score['compound'] == 0:
            out = 'neutral'    
        return out
    
    data['review_title'] = data['review_title'].apply(demoji.replace)
    data['review_title'] = data['review_title'].apply(lambda x: clean_text(x))
    data['review_title_length'] = data['review_title'].apply(word_tokenize).apply(lambda x : len(x))
    
    
    # remove emojis
    data['review_body'] = data['review_body'].apply(demoji.replace)
    
    # cleaning text
    data['review_body'] = data['review_body'].apply(lambda x: clean_text(x))
    
    # reviews length
    data['review_length'] = data['review_body'].apply(word_tokenize).apply(lambda x : len(x))
    
    # overall_sentiment
    data['overall_sentiment'] = data['review_body'].apply(lambda x: sentiment_analysis(x))
    
    # tokenize text
    data['tokens'] = data['review_body'].apply(lambda x: tokenize_text(x))
    
    # tokens list
    tokens = []
    for word in data['tokens']:
        tokens = tokens + word
        
    
    # bigrams dataframe (from tokens without stopwords)
    bigr = nltk.FreqDist(list(bigrams(tokens)))
    bigr = pd.DataFrame(list(bigr.items()),columns=['bigram','frequency'])
    bigr['bigrams'] = bigr['bigram'].apply(lambda x: ' '.join(x))
    bigr.sort_values('frequency',ascending=False,inplace=True)
    
    # trigrams dataframe (from tokens without stopwords)
    trigr = nltk.FreqDist(list(trigrams(tokens)))
    trigr = pd.DataFrame(list(trigr.items()),columns=['trigram','frequency'])
    trigr['trigrams'] = trigr['trigram'].apply(lambda x: ' '.join(x))
    trigr.sort_values('frequency',ascending=False,inplace=True)
    
    
    # corpus of words
    corpus = ' '.join(data['review_body'])
    words = word_tokenize(corpus)

    
    # bigrams dataframe
    bigr_with_sw = nltk.FreqDist(list(bigrams(words)))
    bigr_with_sw = pd.DataFrame(list(bigr_with_sw.items()), columns=[
                                'bigram', 'frequency'])
    bigr_with_sw['bigrams'] = bigr_with_sw['bigram'].apply(lambda x: ' '.join(x))
    bigr_with_sw.sort_values('frequency', ascending=False, inplace=True)

    # trigrams dataframe
    trigr_with_sw = nltk.FreqDist(list(trigrams(words)))
    trigr_with_sw = pd.DataFrame(list(trigr_with_sw.items()), columns=['trigram', 'frequency'])
    trigr_with_sw['trigrams'] = trigr_with_sw['trigram'].apply(lambda x: ' '.join(x))
    trigr_with_sw.sort_values('frequency', ascending=False, inplace=True)
    
    
    # tokens dataframe (tokens without stopwords)    
    tokens_df = pd.DataFrame.from_dict(Counter(tokens), orient='index').reset_index(
    ).rename(columns={'index': 'tokens', 0: 'frequency'}).sort_values('frequency',ascending=False)
    tokens_df['relative_frequency'] = tokens_df['frequency'].apply(lambda x : x/tokens_df['frequency'].sum())
    
   
    lemm_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemm_tokens_df = pd.DataFrame.from_dict(Counter(lemm_tokens), orient='index').reset_index(
    ).rename(columns={'index': 'tokens', 0: 'frequency'}).sort_values('frequency',ascending=False)
    
    
    return data, tokens, tokens_df, bigr, trigr, bigr_with_sw, trigr_with_sw, lemm_tokens_df


@st.cache_resource
def emotion_analysis(data):
    
    df = data['review_body'].apply(lambda x: LeXmo.LeXmo(x)).apply(pd.Series).drop(columns='text').transpose().mean(1).reset_index().rename(columns={'index': 'emotion', 0: 'mean_score'})
    
    df = df[~df['emotion'].isin(['negative', 'positive'])]
    
    return df

@st.cache_resource
def absa_analysis(data,list):
    
    for word in list:
        data[word] = data['review_body'].apply(lambda x : absa_predictor(x,word))
        
    return data


###############
###############
###############
###############
###############
###############

# Title

st.markdown("""
            <div class="container">
                <source srcset="https://fonts.gstatic.com/s/e/notoemoji/latest/1f30d/512.webp" type="image/webp">
                <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f30d/512.gif" alt="🌍" width="100" height="100">
                    <div class="text">
                        <h1>TripScraper</h1>
                    </div>
            </div>
            """, unsafe_allow_html=True)

st.write('#')

# Main tabs

tab_about, tab_app, tab_play = st.tabs([':earth_africa: About',':microscope: Application',':basketball: Playground'])

st.markdown("""
            <br>
            <br>
            <div class="containerIcons">
                <a href="https://github.com/carminemnc"><img src="https://raw.githubusercontent.com/carminemnc/images/main/github.png" title="Follow me on Github!" width="30" height="30"></a>
                <a href="https://www.linkedin.com/in/carmine-minichini"><img src="https://raw.githubusercontent.com/carminemnc/images/main/linkedin.png" alt="Follow me on linkedin" width="30" height="30"></a>
            </div>
            """,unsafe_allow_html=True)


# About Section

with tab_about:
    
    st.write("""
             
             :green[TripScraper] is an open-source Natural Language Processing (NLP) tool that can be used to scrape TripAdvisor reviews and analyze them by ratings, words, and other criteria. 
             
             It uses a variety of NLP techniques, including sentiment analysis, word frequency analysis, word trends over time and aspect based sentiment analysis.
             
             The Playground section of :green[TripScraper] is a sandbox where users can experiment with the NLP models that are used by the tool. This section allows users to try out different features of the models, to see how they work, and to learn more about how NLP can be used to analyze TripAdvisor reviews.
             
             :orange[This application is not designed for business purposes but only as a NLP playground.]
             
             
             """)


# Application

with tab_app:

    with st.form('main'):
        
        st.write("""
                 :green[TripScraper] can only scrape up to second-last page of TripAdvisor :green[Hotel] link. 
                 
                 If the page that you're trying to gather has $n$ pages select $n-1$ pages to scrape.
                 
                 """)
        
        
        # TripAdivor
        url = st.text_input('TripAdvisor link', '',label_visibility='hidden')
        number = st.number_input('Number of pages to scrape:',min_value=1,step=1)
        
        submit_scrape = st.form_submit_button('Run Analysis')
        
        if submit_scrape:
            
            data, tokens, tokens_df, bigr, trigr, bigr_with_sw, trigr_with_sw, lemm_tokens_df = dfCleaner(hotelTripScraper(url,number))
        
            tab1, tab2, tab3, tab4, tab5 = st.tabs(['Ratings', "Reviews",'Over Time Analysis','Emotion Analysis','Aspect Based Analysis'])
                    
            with tab1:
                
                print(data)
                c1,c2,c3 = rating_charts(data)
                
                st.altair_chart(c1,theme="streamlit")
                st.altair_chart(c2,theme='streamlit')
                st.altair_chart(c3,theme='streamlit')
                
            with tab2:
                           
                # Metrics
                
                col1,col2,col3 = st.columns(3)
                
                with col1:
                    st.metric('Average Review Length',value=round(data['review_length'].mean(),1))
                
                with col2:
                    st.metric('Without Stopwords',value=round(
                        data['tokens'].apply(lambda x: len(x)).mean()
                    ,1))  
                with col3:
                    st.metric('Average Title Length',value=round(data['review_title_length'].mean(),1))
                    
                st.write('#')
                
                c1 = barchart(lemm_tokens_df.head(10),'frequency','tokens','10 most common words')
                c2 = barchart(bigr.head(10),'frequency','bigrams','10 most common bigrams')
                c3 = barchart(trigr_with_sw.head(10),'frequency','trigrams','10 most common trigrams')

                c =  c1 | c2 | c3
                
                st.altair_chart(c)
                
                st.markdown('<h1>Word network graph</h1>', unsafe_allow_html=True)
                st.write("""
                         We often want to understand the relationship between words in a review. What sequences of words are common across review text? Given a sequence of words, what word is most likely to follow? What words have the strongest relationship with each other? Therefore, many interesting text analysis are based on the relationships. When we exam pairs of two consecutive words, it is called :green[bigrams]. We can visualize :green[bigrams] in a word network chart.
                         """)
                st.pyplot(words_network_graph(bigr[bigr['frequency']>5],'bigram'))
                
            with tab3:
                
                st.write("""
                         What words and topics have become more frequent, or less frequent, over time? These could give us a sense of the hotel changing ecosystem, such as :green[service], :orange[room cleanliness], :violet[food tastiness] and let us predict what topics will continue to grow in relevance.
                         """)
                
                st.write('#')
                
                words = ['room','service','food']
                st.altair_chart(word_overtime_chart(data,words),theme='streamlit')
                        
                
            with tab4:
                
                st.write("""
                         :green[Emotion analysis] is a field of study that uses natural language processing (NLP) to identify and classify emotions in text. This can be done by identifying words and phrases that are associated with different emotions, as an example in this section is been used :orange[LeXmo] package for classifying emotion based on :blue[EmoLex (NRC Emotion Lexicon)].
                         """)
                st.write('#')              
            
                
                c2 = emotion_radar_chart(emotion_analysis(data))
                st.plotly_chart(c2,theme="streamlit")
                
            with tab5:
                
                st.write("""
                         :green[Aspect-Based Sentiment Analysis (ABSA)] is a Natural Language Processing task that aims to identify and extract the sentiment of specific aspects or components of a product or service. ABSA typically involves a multi-step process that begins with identifying the aspects or features of the product or service that are being discussed in the text. This is followed by sentiment analysis, where the sentiment polarity (positive, negative, or neutral) is assigned to each aspect based on the context of the sentence or document. Finally, the results are aggregated to provide an overall sentiment for each aspect.
                         """)
                st.write('#')
                
                words_ = ['room','service']
                
                data = absa_analysis(data,words_)
                
                c = words_sentiment_chart(data,words_)
                
                st.altair_chart(c,theme='streamlit')

                c2 = words_sentiment_over_time(data,words_[0])
                c3 = words_sentiment_over_time(data,words_[1])  
                
                st.altair_chart(alt.vconcat(c2,c3),theme='streamlit')
                                  
with tab_play:

        st.write("""
                 Provide below an example text...
                 """)
        
        txt = st.text_area('Text:',
                           'Perfectly located for accessing the old town and wandering its narrow streets. The breakfast was fresh and really tasty. The staff was also very gentle and helped us to find most beautiful attractions nearby the hotel. Hotel is average, room was ok , we had an attic room which had noises from the pigeons on the roof.',height=200,label_visibility='hidden')
        
        st.markdown('<h1>Zero-shot analysis</h1>', unsafe_allow_html=True)
        
        st.write("""
                 :green[Zero-Shot Classification] is the task of predicting a class that wasn't seen by the model during training. This method, which leverages a pre-trained language model, can be thought of as an instance of transfer learning which generally refers to using a model trained for one task in a different application than what it was originally trained for. This is particularly useful for situations where the amount of labeled data is small.
                 
                 The aim is to use Zero-Shot Learning models as a :green[topic modelling] algorithm, providing enough labels it is able to classify correctly which is the topic of the text.
                 
                 Model:
                 [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
                 
                 Meaningful article:
                 [Zero-Shot Learning in Modern NLP](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
                 
                 """)
        
        
        with st.form('zeroclassification'):
            
            keywords = st_tags(
                label='Enter Keywords:',
                text='Press enter to add more',
                value=['room','service','location','food','cleanliness','comfort'],
                suggestions=['staff','wifi','sea','another'],
                maxtags = 15,
                key='3')
            
            submit_class = st.form_submit_button('Run')
            
            if submit_class:
            
                df = zero_shot_analyzer(txt,keywords)
                
                st.write('###')
                st.write('###')
                
                st.dataframe(
                    df,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Scores",
                            help="Score",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            
        st.markdown('<h1>Aspect based sentiment analysis</h1>', unsafe_allow_html=True)
        st.write("""
                 The Classic Sentiment Analysis of a sentence can be useful to gather the overall feeling when you aim to find the first impression about something. However in business context such as TripAdvisor reviews, it's meaningful understanding the sentiment on specific aspects or products.
                 
                 Model:
                 [deberta-v3-base-absa-v1.1](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1) 
                 """)

        
        with st.form('absa'):

            option = st.selectbox(
                'Choose term:',
                ('food','location','service','room'),
                label_visibility='hidden'
            )
        
            submit_absa = st.form_submit_button('Run')
        
            if submit_absa:
                
                df = absa_analyzer(txt,option)
                
                st.write('#')
                
                st.dataframe(
                    df,
                    column_config={
                        'labels': 'Sentiment',
                        "probs": st.column_config.ProgressColumn(
                            "Probabilities",
                            help="The probability associated with sentiment",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )   
            
            
            
            
            



