
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(string):
    """
    Sources 
    ----------
    https://stackabuse.com/using-regex-for-text-manipulation-in-python/
    https://lionbridge.ai/articles/using-natural-language-processing-for-spam-detection-in-emails/

    Description
    -----------
    clean text by handling unneeded words

    Parameters
    ----------
    string : str
        string to be processed

    Returns
    -------
    str
        cleaned string
    """
    string = str(string) if type(string) != type('aa') else string
    string = string.lower()
    string = re.sub(r"http\S+", "", string)
    string = re.sub(r"\W", " ", string, flags=re.I)
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " is ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"can't", "cannot ", string)
    string = re.sub(r"n't", " not ", string)
    string = re.sub(r"I'm", "I am", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r"e-mail", "email", string)
    string = re.sub(r" usa ", " america ", string)
    string = re.sub(r" uk ", " england ", string)
    string = re.sub(r"\s+", " ", string, flags=re.I)
    string = string[7:] if re.search(r"^subject", string) else string
    string = re.sub(r"^\s+", "", string)
    string = string[7:] if re.search(r"^re", string) else string
    string = re.sub(r"^\s+", "", string)
    string = re.sub(r"\s+$", "", string)
    string = re.sub(r"\s+[a-zA-Z]\s+", " ", string)

    return string


def preprocess_text(text, tokenizer=word_tokenize, stops_remove=True, stemmer=SnowballStemmer('english'), stop_words=stopwords.words('english')):
    """
    Sources 
    ----------

    Description
    -----------
    Perform the whole preprocessing pipeline for a given text

    Parameters
    ----------
    text : str
        text to be processed

    tokenizer: function (optional)
        used tokenizer

    stemmer: object (optional)
        used stemmer

    stops_remove: list (optional)
        stop words to be removed



    Returns
    -------
    str
        processed string
    """

    # (1) Cleaning text
    text = clean_text(text)

    # (2) Tokenizing
    text = tokenizer(text)

    # (3) Removing stopwords
    text = [word for word in text if word not in stop_words]

    # (4) Stemming
    text = [stemmer.stem(word) for word in text]

    return text


def preprocess_df(df, X, y, tokenizer=word_tokenize, stops_remove=True, stemmer=SnowballStemmer('english'), stop_words=stopwords.words('english')):
    """
    Sources 
    ----------

    Description
    -----------
    Perform the whole preprocessing pipeline for a given dataframe

    Parameters
    ----------
    text : pd.DataFrame
        dataframe to be processed

    tokenizer: function (optional)
        used tokenizer

    stemmer: object (optional)
        used stemmer

    stops_remove: list (optional)
        stop words to be removed



    Returns
    -------
    pd.DataFrame
        processed dataframe
    """
    df_unique = df.drop_duplicates()
    df_ = pd.DataFrame(df_unique[X].apply(lambda x: preprocess_text(
        x, tokenizer=tokenizer, stops_remove=stops_remove, stemmer=stemmer, stop_words=stop_words)))
    df_[y] = df_unique[y]
    return df_
