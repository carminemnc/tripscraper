
# Tripscraper

![Screenshot](./imgs/tripscraper.gif)

Powered by [Streamlit](https://streamlit.io/).

***You can find interesting reading my article on some of text mining techniques and their business application [here](https://carminemnc.github.io/projects/tripscraper/)***

Tripscraper is an open source Natural Language Processing (NLP) tool that can be used to scrape TripAdvisor reviews and analyze them by ratings, words, and other criteria.
             
It uses a variety of NLP techniques, including sentiment analysis, word frequency analysis, word trends over time and aspect based sentiment analysis.

The Playground section of Tripscraper is a sandbox where users can experiment with the NLP models that are used by the tool. This section allows users to try out different features of the models, to see how they work, and to learn more about how NLP can be used to analyze TripAdvisor reviews.

This application is not designed for business purposes but only as a NLP playground.

# How it works

1. Clone the repository
```
git clone https://github.com/carminemnc/tripscraper
```

2. Create virtual environment (first time) inside the repository
```
py -m venv venv
```

3. Activate virtual environment
```
venv\Scripts\activate
```

4. Install packages

```
py -m pip install -r requirements.txt
```

5. Run the app
```
streamlit run app.py
```

6. Quit the app
```
Ctrl + c
```

7. Deactivate virtual environment
```
deactivate
```