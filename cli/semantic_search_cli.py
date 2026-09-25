import argparse
from lib.semantic_search import SemanticSearch


def main() -> None:
    sentence_transformer = SemanticSearch()
    
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
            dest="command", help="Available commands"
        )
    
    verify_sent = subparsers.add_parser("verify", help="Verify model")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            sentence_transformer.verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

