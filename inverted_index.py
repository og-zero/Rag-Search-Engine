import pickle
from collections import Counter, defaultdict
from pathlib import Path
from nltk.stem import PorterStemmer
from keyword_search_cli import load_movies, tokenize_text

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        for token in stemmed_tokens:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(stemmed_tokens)

    def get_documents(self, term: str):
        tokens = tokenize_text(term)
        if not tokens:
            return []
        stemmed_term = stemmer.stem(tokens[0])
        return sorted(list(self.index.get(stemmed_term, set())))

    def build(self):
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

        print(f"Successfully saved index and docmap to '{cache_path}/'")

    def load(self):
        cache_path = Path("cache")

        with open(cache_path / "index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open(cache_path / "docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open(cache_path / "term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)