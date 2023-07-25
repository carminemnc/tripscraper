from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd,numpy as np,time,demoji
from datetime import datetime
from collections import Counter
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.util import bigrams,trigrams
# from LeXmo import LeXmo
from utils import *
def hotelTripScraper(url,pages):
    
    # empty dataframe
    df = pd.DataFrame(columns=['review_title','review_rating','review_date','review_body','user_contributions'])
    
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
            
            
            user_contributions = review.find_element(By.XPATH,".//span[contains(@class, 'yRNgz')]").text
            
            # to a dataframe
            review_objects = [title,int(rating),date,body,user_contributions] 
            
            df.loc[len(df)] = review_objects
            
            #print(len(df))
    

        brwsr.find_element(By.XPATH,'.//a[@class="ui_button nav next primary "]').click()
        
        time.sleep(3)
            

    
    return df