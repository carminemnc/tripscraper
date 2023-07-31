import pandas as pd,numpy as np,demoji
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
import streamlit as st

''' settings '''

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


''' zero-shot classification'''

def zero_shot_predictor(data,labels):
    
    out = zsc(data,
    candidate_labels=labels)
    labels = dict(zip(out['labels'],out['scores']))
    out = max(labels, key=labels.get)
    
    return out

def zero_shot_analyzer(data,labels):
    
    out = zsc(data,
    candidate_labels=labels)
    df = pd.DataFrame({'labels':out['labels'],'score':out['scores']})
    
    return df

''' df cleaner '''

def dfCleaner(data):
    
    lemmatizer = WordNetLemmatizer()
    
    ''' review date '''
    
    data['review_short_date'] = data['review_date'].replace('Date of stay: ','',regex=True)
    data['review_date'] = data['review_date'].replace('Date of stay: ','',regex=True).apply(lambda x: datetime.strptime(x,'%B %Y') )
    data['review_year'] = data['review_date'].apply(lambda x: x.strftime('%Y'))
    data['review_month'] = data['review_date'].apply(lambda x: x.strftime('%B'))
    
    ''' review ratings '''
    
    data['review_rating'] = data['review_rating'].map({10: 'Terrible',
                                                   20: 'Poor',
                                                   30: 'Average',
                                                   40: 'Very Good',
                                                   50: 'Excellent'})
    
    ''' extract emojis from review body and review title '''
    
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
        
    
    ''' functions '''  
    
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
    
    ''' review title '''
    
    data['review_title'] = data['review_title'].apply(demoji.replace)
    data['review_title'] = data['review_title'].apply(lambda x: clean_text(x))
    data['review_title_length'] = data['review_title'].apply(word_tokenize).apply(lambda x : len(x))
    
    ''' review body '''
    
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
        
    ''' bigrams and trigrams (from tokens without stopwords)'''
    
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
    
    ''' corpus of words & tokens(with stopwords)'''
    
    # corpus of words
    corpus = ' '.join(data['review_body'])
    words = word_tokenize(corpus)

    ''' bigrams and trigrams (with stopwords)'''
    
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
    
    ''' tokens dataframe '''
    
    # tokens dataframe (tokens without stopwords)    
    tokens_df = pd.DataFrame.from_dict(Counter(tokens), orient='index').reset_index(
    ).rename(columns={'index': 'tokens', 0: 'frequency'}).sort_values('frequency',ascending=False)
    tokens_df['relative_frequency'] = tokens_df['frequency'].apply(lambda x : x/tokens_df['frequency'].sum())
    
    ''' lemmatized tokens dataframe '''
   
    lemm_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemm_tokens_df = pd.DataFrame.from_dict(Counter(lemm_tokens), orient='index').reset_index(
    ).rename(columns={'index': 'tokens', 0: 'frequency'}).sort_values('frequency',ascending=False)
    
    
    return data, tokens, tokens_df, bigr, trigr, bigr_with_sw, trigr_with_sw, lemm_tokens_df

''' emotion analysis '''

def emotion_analysis(data):
    
    df = data['review_body'].apply(lambda x: LeXmo.LeXmo(x)).apply(pd.Series).drop(columns='text').transpose().mean(1).reset_index().rename(columns={'index': 'emotion', 0: 'mean_score'})
    
    df = df[~df['emotion'].isin(['negative', 'positive'])]
    
    return df

''' aspect based sentiment analysis '''

def absa_analysis(data,list):
    
    for word in list:
        data[word] = data['review_body'].apply(lambda x : absa_predictor(x,word))
        
    return data

