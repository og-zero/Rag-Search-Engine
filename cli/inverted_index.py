from collections import defaultdict, Counter
from keyword_search_cli import tokenize_text, load_movies
from pathlib import Path
import pickle
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter())
        
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            stemmed_token = stemmer.stem(token)
            self.index[stemmed_token].add(doc_id)
            
    
    def get_documents(self, term):
            return sorted(list(self.index.get(term, set())))

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
                    pickle.dump(self.index, f)

        print(f"Successfully saved index and docmap to '{cache_path}/'")
        

    def load(self):
        cache_path = Path("cache")

        index_file = cache_path / "index.pkl"
        docmap_file = cache_path / "docmap.pkl"
        term_frequencies_file = cache_path / "term_frequencies.pkl"

        try:
            with open(index_file, "rb") as f:
                self.index = pickle.load(f)

            with open(docmap_file, "rb") as f:
                self.docmap = pickle.load(f)

            with open(term_frequencies_file, "rb") as f:
                self.term_frequencies = pickle.load(f)
        
        except FileNotFoundError as e:
            print(f"Error: {e}")

    
    def build_command(self):
        idx = InvertedIndex()
        idx.build()
        idx.save()