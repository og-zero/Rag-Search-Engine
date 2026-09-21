from collections import defaultdict, Counter
from keyword_search_cli import tokenize_text, load_movies
from pathlib import Path
from nltk.stem import PorterStemmer
import pickle
import math

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        for token in stemmed_tokens:
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(stemmed_tokens)
        
    
    def get_documents(self, term):
        return sorted(list(self.index.get(term, set())))


    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("Can only have 1 token.")

        stemmed_term = stemmer.stem(tokens[0])
        return self.term_frequencies[doc_id][stemmed_term]


    def get_idf(self, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
                    raise ValueError("Can only have 1 token.")

        stemmed_term = stemmer.stem(tokens[0])
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[stemmed_term])

        idf_value = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf_value


    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

        
    def save(self) -> None:
        cache_path = Path("cache")
        

        cache_path.mkdir(parents=True, exist_ok=True)

        index_file = cache_path / "index.pkl"
        docmap_file = cache_path / "docmap.pkl"
        term_frequencies_file = cache_path / "term_frequencies.pkl"

        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(term_frequencies_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        print(f"Successfully saved index and docmap to '{cache_path}/'")
        

    def load(self):
        cache_path = Path("cache")

        index_file = cache_path / "index.pkl"
        docmap_file = cache_path / "docmap.pkl"
        term_frequencies_file = cache_path / "term_frequencies.pkl"

        with open(index_file, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_file, "rb") as f:
            self.docmap = pickle.load(f)

        with open(term_frequencies_file, "rb") as f:
                self.term_frequencies = pickle.load(f)


    def build_command(self):
        idx = InvertedIndex()
        idx.build()
        idx.save()


    def build_tf_command(self, doc_id, term):        
        idx = InvertedIndex()
        idx.build()
        print(idx.get_tf(doc_id, term))
        idx.save()


    def build_idf_command(self, term):
        self.load()
        idf = self.get_idf(term)
        print(f"Inverse document frequency of '{term}': {idf:.2f}")


    def build_tfifd_command(self, doc_id, term):
        self.load()
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        tf_idf = tf * idf
        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
