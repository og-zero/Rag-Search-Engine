from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image):
        image_load = Image.open(image)
        embedding = self.model.encode(image_load)

        return [embedding][0]


def embed_image_command(image):
    mlt_model = MultimodalSearch()
    embedding = mlt_model.embed_image(image)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")