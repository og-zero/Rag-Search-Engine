import argparse

from lib.hybrid_search import (normalize_command, weighted_search_command,
                        rrf_search_command         
                        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", description="Min-Max normalization")
    normalize_parser.add_argument("inputs", nargs="*", help="Files or data to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", description="Weighted-search command")
    weighted_search_parser.add_argument("query", type=str, help="Query to pass")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help='alpha (or "α") is just a constant that we can use to dynamically control the weighting between the two scores')
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    rrf_search = subparsers.add_parser("rrf-search", description="Reciprocal Rank Fusion")
    rrf_search.add_argument("query", type=str, help="Query to pass")
    rrf_search.add_argument("-k", type=int, default=60)
    rrf_search.add_argument("--limit", type=int, default=5)
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.inputs)
        case "weighted-search":
            weighted_search_command(args.query,
                            args.alpha,
                            args.limit
                        )
        case "rrf-search":
            rrf_search_command(args.query,
                        args.k,
                        args.limit,
                        args.enhance
                    )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
