import requests
import json
import pandas as pd
import nltk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


api_key = "wwFVxDThbk6roswX4uS8aA7zC7Akuqt0"
year = 2022
articles_list = []

#split it in 2 because of too large requests

for month in range(1, 7):
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"

    response = requests.get(url)
    data = response.json()

    # Check if the response was successful
    if response.status_code != 200:
        continue

    # Check if 'response' key exists in the data
    if 'response' not in data:
        continue

    articles = data['response']['docs']
    
    # Process the articles
    for article in articles:
        headline = article['headline']['main']
        lead_paragraph = article['lead_paragraph']
        section_name = article['section_name']
        
        # Tokenize the words in the lead paragraph
        space_tokenizer = nltk.tokenize.SpaceTokenizer()
        tokenized_paragraph = space_tokenizer.tokenize(lead_paragraph)
        
        # Convert words to lowercase
        tokenized_paragraph_lower = [w.lower() for w in tokenized_paragraph]
        
        # Remove stopwords from the tokenized paragraph
        stop_words = set(stopwords.words('english'))
        tokens_without_stopwords = [w for w in tokenized_paragraph_lower if w not in stop_words]
        
        # POS Tag the token
        paragraph_processed = nltk.pos_tag(tokens_without_stopwords)
        
        # Lemmatize the token
        wordnet = nltk.WordNetLemmatizer()
        #tokens_lemmatized = [wordnet.lemmatize(t) for t in tokenized_paragraph_lower]
        
        articles_list.append({
            "headline": headline,
            "lead_paragraph": lead_paragraph,
            "section_name": section_name,
         #   "tokens_lemmad": tokens_lemmatized,
            "processed_paragraph": paragraph_processed
        })
        

articles_df = pd.DataFrame(articles_list)



# for month in range(1, 7):
#     url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"

#     response = requests.get(url)
#     data = response.json()

#     # Check if the response was successful
#     if response.status_code != 200:
#         continue

#     # Check if 'response' key exists in the data
#     if 'response' not in data:
#         continue

#     articles = data['response']['docs']
    
#     # Process the articles
#     for article in articles:
#         headline = article['headline']['main']
#         lead_paragraph = article['lead_paragraph']
#         section_name = article['section_name']
        
#         # Tokenize the words in the lead paragraph
#         space_tokenizer = nltk.tokenize.SpaceTokenizer()
#         tokenized_paragraph = space_tokenizer.tokenize(lead_paragraph)
        
#         # Convert words to lowercase
#         tokenized_paragraph_lower = [w.lower() for w in tokenized_paragraph]
        
#         # Remove stopwords from the tokenized paragraph
#         stop_words = set(stopwords.words('english'))
#         tokens_without_stopwords = [w for w in tokenized_paragraph_lower if w not in stop_words]
        
#         # POS Tag the token
#         paragraph_processed = nltk.pos_tag(tokens_without_stopwords)
        
#         # Lemmatize the token
#         wordnet = nltk.WordNetLemmatizer()
#         #tokens_lemmatized = [wordnet.lemmatize(t) for t in tokenized_paragraph_lower]
        
#         articles_list.append({
#             "headline": headline,
#             "lead_paragraph": lead_paragraph,
#             "section_name": section_name,
#          #   "tokens_lemmad": tokens_lemmatized,
#             "processed_paragraph": paragraph_processed
#         })

# articles_df = pd.DataFrame(articles_list)