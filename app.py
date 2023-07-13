import streamlit as st
from styler import *
import altair as alt
from utils import *
import pandas as pd


data, emojis = dfCleaner(pd.read_csv('tripadvisor.csv'))
temp_ = data.groupby(['review_month', 'review_rating'])['review_date'].count(
).reset_index().rename(columns={'review_date': 'count'})

#st.set_page_config(layout="wide")

st.markdown(streamlit_style, unsafe_allow_html=True)

st.title(':earth_africa: TripScraper')

st.markdown('<p class="header-font">Hello World !!</p>', unsafe_allow_html=True)

url = st.text_input('URL', 's')

tab1, tab2 = st.tabs(['Ratings', "Words"])

with tab1:
    
    chart = alt.Chart(temp_).mark_arc().encode(
        theta="count",
        color="review_rating",
        column='review_month'
    ).properties(
    width=180,
    height=180
).interactive()
    
    st.altair_chart(chart, theme="streamlit")



