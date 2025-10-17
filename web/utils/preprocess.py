import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

stopwords = set(thai_stopwords())

def clean_and_tokenize(text: str):
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^ก-๙a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, engine='newmm')
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)
