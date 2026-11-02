import argparse
from lib.augmented_generation import (rag_command, summarize_command,
                                      citations_command, question_command)

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
            "summarize", help="Summarization pipeline that synthesizes search results."
        )
    summarize_parser.add_argument("query", type=str, help="Search query for Summarization")
    summarize_parser.add_argument("--limit", type=int, default=5)

    citations_parser = subparsers.add_parser(
                "citations", help="citation-aware answer command that references its sources"
            )
    citations_parser.add_argument("query", type=str, help="Search query for Summarization")
    citations_parser.add_argument("--limit", type=int, default=5)

    question_answering_parser = subparsers.add_parser(
                "question", help="conversational question-answering command for direct responses."
            )
    question_answering_parser.add_argument("query", type=str, help="Search query for Summarization")
    question_answering_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
        case "summarize":
            query = args.query
            summarize_command(args.query,
                        args.limit)
        case "citations":
            citations_command(args.query,
                        args.limit)
        case "question":
            question_command(args.query,
                        args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
