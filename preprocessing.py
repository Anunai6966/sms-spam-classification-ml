import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords',quiet=True)

stop_words=set(stopwords.words('english'))

def clean_text(text):
    
    # Lower casing all the words
    text=text.lower()
    
    # removing all punctuations and numbers
    text=re.sub(r'[^a-z\s]','',text)

    # Tokenisation
    words=text.split()

    # Stopword removal
    words=[word for word in words if word not in stop_words]

    return ' '.join(words)