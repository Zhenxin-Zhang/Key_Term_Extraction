from lxml import etree
import nltk
import collections
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_root():
    xml_file = "news.xml"
    root = etree.parse(xml_file).getroot()
    return root


def normalize_text(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(t) for t in tokens]

    norm_text = []
    stop_words = set(stopwords.words('english'))
    punc_words = set(list(string.punctuation))

    for l in lemma:
        if (l not in stop_words and l not in punc_words
                and nltk.pos_tag([l])[0][1] == "NN"):
            norm_text.append(l)
    return norm_text


def find_frequent(tokens):
    c = collections.Counter(tokens)
    tmp = sorted(c.items(), key=lambda item: item[0], reverse=True)
    tmp = sorted(tmp, key=lambda item: -item[1])
    most_freq_5 = tmp[:5]
    freq_words = [a for a, b in most_freq_5]
    return freq_words


def process_title_body(root):
    titles = []
    cor = []
    corpus = root[0]
    for news in corpus:
        title, body = "", ""
        for value in news:
            if value.get('name') == "head":
                title = value.text
            if value.get('name') == "text":
                body = value.text
        titles.append(title)
        tokens = normalize_text(body)
        cor.append(" ".join(tokens))
    return titles, cor


def main():
    r = get_root()
    tit, c = process_title_body(r)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(c)
    tfidf_array = tfidf_matrix.toarray()
    terms = vectorizer.get_feature_names_out()

    for i in range(10):
        tf = tfidf_array[i]
        tmp = []
        print(f"{tit[i]}:")
        for t, t_val in zip(terms, tf):
            tmp.append((t, t_val))
        tmp = sorted(tmp, key=lambda item: item[0], reverse=True)
        tmp = sorted(tmp, key=lambda item: -item[1])
        most_freq_5 = tmp[:5]
        for a, b in most_freq_5:
            print(a, end=" ")
        print("")




if __name__ == "__main__":
    main()



