import math
from nltk.stem import PorterStemmer
from keyword_search_cli import tokenize_text

stemmer = PorterStemmer()

def get_tf(inverted_index, doc_id: int, term: str) -> int:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("Can only have 1 token.")

    stemmed_term = stemmer.stem(tokens[0])
    return inverted_index.term_frequencies[doc_id][stemmed_term]


def get_idf(inverted_index, term: str) -> float:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("Can only have 1 token.")

    stemmed_term = stemmer.stem(tokens[0])
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.index[stemmed_term])

    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def get_tfidf(inverted_index, doc_id: int, term: str) -> float:
    tf = get_tf(inverted_index, doc_id, term)
    idf = get_idf(inverted_index, term)
    return tf * idf