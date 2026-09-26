import argparse
<<<<<<< HEAD
from lib.semantic_search import SemanticSearch


def main() -> None:
    sentence_transformer = SemanticSearch()
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
            dest="command", help="Available commands"
        )
    
    verify_sent = subparsers.add_parser("verify", help="Verify model")
    
=======
from lib.semantic_search import (
    SemanticSearch, embed_text, verify_embeddings
)

sim_search = SemanticSearch()

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")\
    
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )
    
    verify_model = subparsers.add_parser("verify", description="Verify model")

    embedding = subparsers.add_parser("embed_text", description="Embed text")
    embedding.add_argument("text", type=str, help="Text to embedd")

    movies_embedding = subparsers.add_parser("verify_embeddings", description="Embed movies")

>>>>>>> 0f7d832 (feat(cli): add text embedding generation to semantic search)
    args = parser.parse_args()

    match args.command:
        case "verify":
<<<<<<< HEAD
            sentence_transformer.verify_model()
=======
            sim_search.verify_model()
        case "embed_text":
            embed_text(
                args.text
            )
>>>>>>> 0f7d832 (feat(cli): add text embedding generation to semantic search)
        case _:
            parser.print_help()


if __name__ == "__main__":
<<<<<<< HEAD
    main()

=======
    main()
>>>>>>> 0f7d832 (feat(cli): add text embedding generation to semantic search)
