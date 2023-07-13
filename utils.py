from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd,numpy as np,time,demoji
from datetime import datetime
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams,trigrams
from LeXmo import LeXmo

# import chromedriver_autoinstaller
stop_words = stopwords.words('english')


''' Df Cleaner '''
def dfCleaner(data):
    
    # review dates
    
    data['review_short_date'] = data['review_date'].replace('Date of stay: ','',regex=True)
    data['review_date'] = data['review_date'].replace('Date of stay: ','',regex=True).apply(lambda x: datetime.strptime(x,'%B %Y') )
    data['review_year'] = data['review_date'].apply(lambda x: x.strftime('%Y'))
    data['review_month'] = data['review_date'].apply(lambda x: x.strftime('%B'))
    
    # review rating
    
    data['review_rating'] = data['review_rating'].map({10: 'Terrible',
                                                   20: 'Poor',
                                                   30: 'Average',
                                                   40: 'Very Good',
                                                   50: 'Excellent'})
    
    # extract emojis from review body and review title
    
    data['review_body_emojis'] = data['review_body'].apply(demoji.findall).apply(
        lambda x: '' if not list(x.keys()) else list(x.keys()))
    data['review_title_emojis'] = data['review_title'].apply(demoji.findall).apply(
        lambda x: '' if not list(x.keys()) else list(x.keys()))
    emojis = []
    temp_ = [x for x in data['review_body_emojis'].to_list() if x]
    temp_t = [x for x in data['review_title_emojis'].to_list() if x]
    
    for item in temp_:
        emojis = emojis + item

    for item in temp_t:
        emojis = emojis + item
        
    emojis = pd.DataFrame.from_dict(Counter(emojis), orient='index').reset_index(
    ).rename(columns={'index': 'emojis', 0: 'frequency'}).sort_values('frequency',ascending=False)
        
    ''' Functions '''  
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

        text = tokens_without_sw
        
        return text
    
    # remove emojis
    data['review_body'] = data['review_body'].apply(demoji.replace)
    
    # cleaning text
    data['review_body'] = data['review_body'].apply(lambda x: clean_text(x))
    
    # tokenize text
    data['tokens'] = data['review_body'].apply(lambda x: tokenize_text(x))
    
    # tokens list
    tokens = []
    for word in data['tokens']:
        tokens = tokens + word
    
    # # bigrams dataframe (from tokens without stopwords)
    # bigr = nltk.FreqDist(list(bigrams(tokens)))
    # bigr = pd.DataFrame(list(bigr.items()),columns=['bigram','frequency'])
    # bigr['bigram'] = bigr['bigram'].apply(lambda x: ' '.join(x))
    # bigr.sort_values('frequency',ascending=False,inplace=True)
    
    # # trigrams dataframe (from tokens without stopwords)
    # trigr = nltk.FreqDist(list(trigrams(tokens)))
    # trigr = pd.DataFrame(list(trigr.items()),columns=['trigram','frequency'])
    # trigr['trigram'] = trigr['trigram'].apply(lambda x: ' '.join(x))
    # trigr.sort_values('frequency',ascending=False,inplace=True)
    
    # corpus of words
    corpus = ' '.join(data['review_body'])
    words = word_tokenize(corpus)

    # bigrams dataframe
    bigr_with_sw = nltk.FreqDist(list(bigrams(words)))
    bigr_with_sw = pd.DataFrame(list(bigr_with_sw.items()), columns=[
                                'bigram', 'frequency'])
    bigr_with_sw['bigram'] = bigr_with_sw['bigram'].apply(lambda x: ' '.join(x))
    bigr_with_sw.sort_values('frequency', ascending=False, inplace=True)

    # trigrams dataframe
    trigr_with_sw = nltk.FreqDist(list(trigrams(words)))
    trigr_with_sw = pd.DataFrame(list(trigr_with_sw.items()), columns=['trigram', 'frequency'])
    trigr_with_sw['trigram'] = trigr_with_sw['trigram'].apply(lambda x: ' '.join(x))
    trigr_with_sw.sort_values('frequency', ascending=False, inplace=True)
    
    # tokens dataframe (tokens without stopwords)    
    tokens_df = pd.DataFrame.from_dict(Counter(tokens), orient='index').reset_index(
    ).rename(columns={'index': 'tokens', 0: 'frequency'}).sort_values('frequency',ascending=False)
    
    emo_df = data['review_body'].apply(lambda x: LeXmo.LeXmo(x)).apply(pd.Series).drop(
    'text', 1).transpose().mean(1).reset_index().rename(columns={'index': 'emotion', 0: 'mean_score'})
    
    return data,emojis,tokens_df,bigr_with_sw,trigr_with_sw,tokens,corpus,words

def hotelTripScraper(url,pages):
    
    # empty dataframe
    df = pd.DataFrame(columns=['review_title','review_rating','review_date','review_body'])
    
    # ChromeDriver Options
    
    chromeOptions = Options()
    chromeOptions.add_argument("--headless")
    brwsr = webdriver.Chrome(options=chromeOptions) # Chrome webdriver
    
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
            
            
            user_experience = review.find_element(By.XPATH,".//span[contains(@class, 'yRNgz')]")
            
            
            # to a dataframe
            review_objects = [title,rating,date,body] 
            
            df.loc[len(df)] = review_objects
        
    # try:    
        brwsr.find_element(By.XPATH,'.//a[@class="ui_button nav next primary "]').click()
    # except:
    #     pass

        time.sleep(3)
    
    
    return df


#hotelTripScraper('https://www.tripadvisor.ca/Hotel_Review-g304551-d1200682-Reviews-Hotel_The_Royal_Plaza-New_Delhi_National_Capital_Territory_of_Delhi.html',2)


