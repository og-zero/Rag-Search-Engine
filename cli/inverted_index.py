from collections import defaultdict
from keyword_search_cli import tokenize_text, load_movies
from pathlib import Path
import pickle

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}

        
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            
    
    def get_documents(self, term):
        return sorted(list(self.index[term]))
    

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']}{movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

        
    def save(self, cache_dir: str = "cache") -> None:
        cache_path = Path(cache_dir)
        

        cache_path.mkdir(parents=True, exist_ok=True)

        index_file = cache_path / "index.pkl"
        docmap_file = cache_path / "docmap.pkl"


        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)

        print(f"Successfully saved index and docmap to '{cache_dir}/'")
        
        
    def build_command(self):
        pass