import pandas as pd,time
from selenium import webdriver
from selenium.webdriver import FirefoxOptions,ChromeOptions
from selenium.webdriver.common.by import By
import os


def reviews_scraper(url,pages):
    
    # empty dataframe
    df = pd.DataFrame(columns=['review_title','review_rating','review_date','review_body','user_contributions'])
    
    ' Settings'
    options = ChromeOptions()
    options.add_argument('--headless')
    brwsr = webdriver.Chrome(options=options)
    main_url = url
    
    # iterate trough pages
    page_counter = 0
    for page in range(pages):
        
        # if it's the first page
        if page_counter==0:
            url = main_url
        else:
            url = main_url.split('-Reviews-')[0] + '-Reviews-or' + str(page_counter) + '-' + main_url.split('-Reviews-')[1]
            
        # add 'or + page_counter'
        page_counter +=10
        
        # print(url)
        
        # get URL
        brwsr.get(url)
        # wait for DOM
        time.sleep(2)
        
        # handling cookies
        brwsr.find_element(By.XPATH,'//*[@id="onetrust-accept-btn-handler"]').click()
        
        # review tabs
        rev_tabs = brwsr.find_element(By.XPATH,".//div[contains(@data-test-target, 'reviews-tab')]")
        # reviews
        reviews = rev_tabs.find_elements(By.XPATH,".//div[contains(@data-test-target, 'HR_CC_CARD')]")
        
        for review in reviews:
            
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
            
            review_objects = [title,int(rating),date,body,user_contributions]
            
            df.loc[len(df)] = review_objects
            
        # print # of reviews scraped
        print(f'dataframe length: {len(df)}')
    
    return df