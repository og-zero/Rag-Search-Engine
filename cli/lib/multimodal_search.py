from PIL import Image
from sentence_transformers import SentenceTransformer
from cli.keyword_search_cli import load_movies
from lib.semantic_search import cosine_similarity

class MultimodalSearch():
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image):
        image_load = Image.open(image)
        embedding = self.model.encode(image_load)

        return [embedding][0]

    def search_with_image(self, image):
        image_embedding = self.embed_image(image)
        similaritys = []
        for idx, text_embedding in enumerate(self.text_embeddings):
            similaritys.append((idx, cosine_similarity(image_embedding, text_embedding)))

        similaritys.sort(key=lambda x: x[1], reverse=True)

        return similaritys


def embed_image_command(image):
    movies = load_movies()
    mlt_model = MultimodalSearch(movies)

    result = mlt_model.search_with_image(image)

    for i, sim in enumerate(result[:3], 1):
        print(f"{i}. {movies[sim[0]]['title']} (similarity: {sim[1]:.1f})")
        print(f"{movies[sim[0]]['description'][:100]}")