import argparse
from lib.semantic_search import SemanticSearch
from lib.semantic_search import (
    SemanticSearch, embed_text, verify_embeddings,
    embed_query_text
)


def main() -> None:
    sentence_transformer = SemanticSearch()
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
            dest="command", help="Available commands"
        )
    
    verify_sent = subparsers.add_parser("verify", help="Verify model")


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

    query_embedding = subparsers.add_parser("embed_query", description="Embed query")
    query_embedding.add_argument("query", type=str, help="Query to embedd")

    args = parser.parse_args()

    match args.command:
        case "verify":
            sim_search.verify_model()
        case "embed_text":
            embed_text(
                args.text
            )
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()