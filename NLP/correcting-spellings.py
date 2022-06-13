#!/usr/bin/env/ 'conda':base

import pandas as pd
import nltk
from textblob import TextBlob

# ------------- Create data -----------------
text=['Introduction to NLP',
        'It is likely to be useful, to people ',
        'Machine learning is the new electrcity', 
        'R is good langauage',
        'I like this book',
        'I want more books like this']

data = pd.DataFrame({'tweet': text})
#print(data)

data['tweet'].apply(lambda x: str(TextBlob(x).correct()))
print(data)

# Doesn't correct any incorrect spellings.
# Dont use textblob for spell correction.
# There's a library called 'pattern en' which seems far better than nltk.
