import argparse
from lib.semantic_search import SemanticSearch
from lib.semantic_search import (
    SemanticSearch, embed_text, verify_embeddings,
    embed_query_text, search_command, chunking_command,
    simantic_chunking, embed_chunks_command, seach_chunks_command
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

    search_embedding = subparsers.add_parser("search", description="Search embeddings")
    search_embedding.add_argument("query", type=str, help="Query to embedd")  
    search_embedding.add_argument("--limit", type=int, default=5, help="Limit result")

    chunking = subparsers.add_parser("chunk", description="Chunking text")
    chunking.add_argument("text", type=str, help="The text to chunk")
    chunking.add_argument("--chunk-size", type=int, default=200, help="Chunk size argument")
    
    sem_chunk = subparsers.add_parser("semantic_chunk", description="Semantic chunk")
    sem_chunk.add_argument("text", type=str, help="The text to chunk")
    sem_chunk.add_argument("--max-chunk-size", type=int, default=200, help="Chunk size argument")
    sem_chunk.add_argument("--overlap", type=int, default=0, help="overlaping")
    
    chunks_embed = subparsers.add_parser("embed_chunks", description="Chunks embedings.")
    
    search_chunkeds = subparsers.add_parser("search_chunked", description="Search chenked.")
    search_chunkeds.add_argument("query", type=str, help="Query to search.")
    search_chunkeds.add_argument("--limit", type=int, default=5, help="Limit result")
    
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
        case "search":
            search_command(
                args.query, args.limit
            )
        case "chunk":
            chunking_command(
                args.text, args.chunk_size
            )
        case "semantic_chunk":
            simantic_chunking(
                args.text, args.max_chunk_size,
                args.overlap
            )
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            seach_chunks_command(args.query, 
                                 args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()