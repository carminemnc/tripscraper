import streamlit as st
import pandas as pd
from streamlit_tags import *
import os
from charter import *
from utils import *
from scraper import *


# settings

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
###############
###############

# title

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

# tabs

tab_about, tab_app, tab_play = st.tabs([':earth_africa: About',':microscope: Application',':basketball: Playground'])

st.markdown("""
            <br>
            <br>
            <div class="containerIcons">
                <a href="https://github.com/carminemnc"><img src="https://raw.githubusercontent.com/carminemnc/imgs/main/github-tripscraper.png" title="Follow me on Github!" width="30" height="30"></a>
                <a href="https://www.linkedin.com/in/carmine-minichini"><img src="https://raw.githubusercontent.com/carminemnc/imgs/main/linkedin-tripscraper.png" alt="Follow me on linkedin" width="30" height="30"></a>
            </div>
            """,unsafe_allow_html=True)

# about

with tab_about:
    
    st.write("""
             
             :green[TripScraper] is an open-source Natural Language Processing (NLP) tool that can be used to scrape TripAdvisor reviews and analyze them by ratings, words, and other criteria. 
             
             It uses a variety of NLP techniques, including sentiment analysis, word frequency analysis, word trends over time and aspect based sentiment analysis.
             
             The Playground section of :green[TripScraper] is a sandbox where users can experiment with the NLP models that are used by the tool. This section allows users to try out different features of the models, to see how they work, and to learn more about how NLP can be used to analyze TripAdvisor reviews.
             
             :orange[This application is not designed for business purposes but only as a NLP playground.]
             
             
             """)

# application

with tab_app:

    with st.form('main'):
        
        st.write("""
                 :green[TripScraper] can only scrape up to second-last page of TripAdvisor :green[Hotel] link. 
                 
                 If the page that you're trying to gather has $n$ pages select $n-1$ pages to scrape.
                 
                 """)
        
        
        # trip advisor url
        url = st.text_input('TripAdvisor link', '',label_visibility='hidden')
        
        # trip advisor pages
        number = st.number_input('Number of pages to scrape:',min_value=1,step=1)
        
        # submit scrape
        submit_scrape = st.form_submit_button('Run Analysis')
        
        if submit_scrape:
            
            
            data, tokens, tokens_df, bigr, trigr, bigr_with_sw, trigr_with_sw, lemm_tokens_df = dataframe_cleaner(reviews_scraper(url,number))
        
            tab1, tab2, tab3, tab4, tab5 = st.tabs(['Ratings', "Reviews",'Over time analysis','Emotion analysis','ABSA analysis'])
                    
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
                    st.metric('Average reviews length',value=round(data['review_length'].mean(),1))
                
                with col2:
                    st.metric('Without stopwords',value=round(
                        data['tokens'].apply(lambda x: len(x)).mean()
                    ,1))  
                with col3:
                    st.metric('Average titles length',value=round(data['review_title_length'].mean(),1))
                    
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
                         :green[Aspect-based sentiment analysis (ABSA)] is a Natural Language Processing task that aims to identify and extract the sentiment of specific aspects or components of a product or service. ABSA typically involves a multi-step process that begins with identifying the aspects or features of the product or service that are being discussed in the text. This is followed by sentiment analysis, where the sentiment polarity (positive, negative, or neutral) is assigned to each aspect based on the context of the sentence or document. Finally, the results are aggregated to provide an overall sentiment for each aspect.
                         """)
                st.write('#')
                
                words_ = ['room','service']
                
                data = absa_analysis(data,words_)
                
                c = words_sentiment_chart(data,words_)
                
                st.altair_chart(c,theme='streamlit')

                c2 = words_sentiment_over_time(data,words_[0])
                c3 = words_sentiment_over_time(data,words_[1])  
                
                st.altair_chart(alt.vconcat(c2,c3),theme='streamlit')
   
# playground
                               
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
            
            
            
            
            



