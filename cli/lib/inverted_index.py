import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from nltk.stem import PorterStemmer
import os

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.bm25_map_document = {}


    def __add_document(self, doc_id, text):
        from cli.keyword_search_cli import tokenize_text
        tokens = tokenize_text(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        for token in stemmed_tokens:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(stemmed_tokens)
        self.doc_lengths[doc_id] = len(stemmed_tokens)

    def get_documents(self, term):
        from cli.keyword_search_cli import tokenize_text
        tokens = tokenize_text(term)
        if not tokens:
            return []
        stemmed_term = stemmer.stem(tokens[0])
        return sorted(list(self.index.get(stemmed_term, set())))

    def get_tf(self, doc_id, term):
        from cli.keyword_search_cli import tokenize_text

        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("Can only have 1 token.")

        stemmed_term = stemmer.stem(tokens[0])
        return self.get_tf_stemmed(doc_id, stemmed_term)

    def get_tf_stemmed(self, doc_id, stemmed_term):
        return self.term_frequencies[doc_id].get(stemmed_term, 0)

    def get_idf(self, term):
        from cli.keyword_search_cli import tokenize_text
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Can only have 1 token.")

        stemmed_term = stemmer.stem(tokens[0])
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[stemmed_term])

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf


    def get_bm25_idf(self, term: str) -> float:
        from cli.keyword_search_cli import tokenize_text, load_movies
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 token.")

        stemmed_term = stemmer.stem(token[0])
        N = len(self.docmap)
        df = len(self.index[stemmed_term])
        # Numerator (N - df + 0.5): Documents without the term, plus smBM25_K1 = 1.5oothing
        # Denominator (df + 0.5): Documents withen.")

        stemmed_term = stemmer.stem(token[0])
        N = len(self.docmap)
        df = len(self.index[stemmed_term])
        # Numerator (N - df + 0.5): Documents without the term, plus smBM25_K1 = 1.5oothing
        # Denominator (df + 0.5): Documents with the term, plus smoothing
        # Why add 0.5? To prevent division by 0. The fancy term for this is Laplace smoothing.
        # Why add 1? The final + 1 ensures IDF is always positive, which handles some edge cases.
        bm25_idf_score = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf_score


    def get_bm25_tf(self, doc_id, term, k1, b):
        """
        So a word appearing 20 times gets 20× the weight of appearing once.

        BM25 says:

        The first few occurrences are useful, but after a while, 
        seeing the same word again doesn't tell us that much more.
        """
        from cli.keyword_search_cli import tokenize_text

        term = tokenize_text(term)
        stemmed_term = stemmer.stem(term[0])
        # print(f"DEBUG: doc_id={doc_id}, term={term!r}, stemmed_term={stemmed_term!r}")
        tf = self.get_tf_stemmed(doc_id, stemmed_term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        # b controls how strongly document length matters.
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component


    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
    
        if len(lengths) == 0:
            return 0.0
    
        total_length = sum(lengths)
        return total_length / len(lengths)


    def get_bm25(self, doc_id, term, k1, b):
        bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
        bm25_idf = self.get_bm25_idf(term)
        bm25 = bm25_tf * bm25_idf

        return bm25


    def bm25_search(self, query, k1, b, limit):
        from cli.keyword_search_cli import tokenize_text
    
        query_tokens = tokenize_text(query)
        for doc_id in self.docmap:
            bm25_score = 0
            for token in query_tokens:
                bm25_score += self.get_bm25(doc_id, token, k1, b)
            self.bm25_map_document[doc_id] = bm25_score
        # scores.items() creates pairs: It turns a dictionary into a list of tuples formatted as (key, value). In this context, that means (doc_id, score)
        # he lambda x: x[1] line tells Python, "When you look at each pair (x), use its second element (x[1]) to determine the sorting order."
        sorted_scores = sorted(self.bm25_map_document.items(),
                              key=lambda x: x[1],
                              reverse=True)
        return sorted_scores[:limit]
        # list_bm25_score = []
        # for i, score in enumerate(sorted_scores[:limit]):
        # #   print(f"{i+1}. ({score[0]}) {self.docmap[score[0]]['title']} - Score: {score[1]:.2f}")
        #     list_bm25_score.append(score[1])
        # return list_bm25_score

    def build(self):
        from cli.keyword_search_cli import load_movies
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        cache_path = Path("cache")
        cache_path.mkdir(parents=True, exist_ok=True)

        with open(cache_path / "index.pkl", "wb") as f:
            pickle.dump(dict(self.index), f)

        with open(cache_path / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(cache_path / "term_frequencies.pkl", "wb") as f:
            pickle.dump(dict(self.term_frequencies), f)

        with open(cache_path / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

        print(f"Successfully saved index and docmap to '{cache_path}/'")

    def load(self):
        cache_path = Path("cache")

        with open(cache_path / "index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open(cache_path / "docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open(cache_path / "term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(cache_path / "doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)


    def build_command(self):
        idx = InvertedIndex()
        idx.build()
        idx.save()

    def build_tf_command(self, doc_id, term):
        idx = InvertedIndex()
        idx.load()
        score = idx.get_tf(doc_id, term)

        print(f"Term frequency of '{term}' in document {doc_id}: {score}")
        

    def build_idf_command(self, term):
        idx = InvertedIndex()
        idx.load()
        idf = idx.get_idf(term)

        print(f"Inverse document frequency of '{term}': {idf:.2f}")
        

    def build_tfifd_command(self, doc_id, term):
        idx = InvertedIndex()
        idx.load()
        tf_idf = idx.get_tfidf(doc_id, term)

        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


    def build_bm25idf_command(self, term):
        idx = InvertedIndex()
        idx.load()
        bm25idf = idx.get_bm25_idf(term)

        print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")     

    
    def bm25_tf_command(self, doc_id, term, k1, b):
        idx = InvertedIndex()
        idx.load()
        bm25tf = idx.get_bm25_tf(doc_id, term, k1, b)

        print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


    def bm25search_command(self, query, k1, b):
        idx = InvertedIndex()
        idx.load()

        idx.bm25_search(query, k1, b, limit=5)