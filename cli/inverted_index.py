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

    def __add_document(self, doc_id, text):
        from cli.keyword_search_cli import tokenize_text
        tokens = tokenize_text(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        for token in stemmed_tokens:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(stemmed_tokens)

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
        return self.term_frequencies[doc_id][stemmed_term]

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


    def get_bm25_tf(self, doc_id, term, k1):
        """
        So a word appearing 20 times gets 20× the weight of appearing once.

        BM25 says:

        The first few occurrences are useful, but after a while, 
        seeing the same word again doesn't tell us that much more.
        """
        from cli.keyword_search_cli import tokenize_text

        term = tokenize_text(term)
        stemmed_term = stemmer.stem(term[0])
        tf = self.get_tf(doc_id, stemmed_term)

        BM25_saturation = (tf * (k1 + 1)) / (tf + k1)
        return BM25_saturation


    def __get_avg_doc_length(self) -> float:
        pass


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

        with open(cache_path / "doc_lengths.pkl", "rb") as f:
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

    
    def bm25_tf_command(self, doc_id, term, k1):
        idx = InvertedIndex()
        idx.load()
        bm25tf = idx.get_bm25_tf(doc_id, term, k1)

        print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

        