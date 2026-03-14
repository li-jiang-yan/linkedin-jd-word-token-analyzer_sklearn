from collections import Counter
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def extract_text(html_doc):
    soup = BeautifulSoup(html_doc, "html.parser")
    lines = list(line.strip() for line in soup.get_text().split("\n") if line.strip())
    return "\n".join(lines)


def count_tokens(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    tokens = vectorizer.get_feature_names_out()
    counts = map(int, np.sum(np.where(X.toarray() > 0, 1, 0), axis=0))
    return Counter(dict(zip(tokens, counts)))
