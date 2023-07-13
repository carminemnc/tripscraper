import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained(
    "yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification \
    .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

def absa_predictor(sentence, word):

    sentence = sentence
    aspect = word
    inputs = absa_tokenizer(
        f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    labels = dict(zip(["negative", "neutral", "positive"], probs))
    result = max(labels, key=labels.get)
    inf_output = f'Sentiment for {aspect} is {result} with a prob of: {max(labels.values())}'
    return inf_output


st.title(':earth_africa: Changes')
text = st.text_input('Provide your text here:', 'example_text')
word = st.text_input('Provide your word here', 'position')


if text:
    st.write(absa_predictor(text,word))