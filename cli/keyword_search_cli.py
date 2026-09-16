import argparse
import json
import string 
from pathlib import Path

def search(File_path: str, search_query: list) -> None:
    path = Path(File_path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for v in data.values():
        for d in v:
            if "title" in d:
                clean_title = d["title"].translate(
                    str.maketrans("", "", string.punctuation)
                ).lower()

                title_tokenizing = clean_title.split()

                for token_q in search_query:
                    for token_t in title_tokenizing:
                        if token_q in token_t:
                            res.append(d["title"])
                            break

            
    for i, movie in enumerate(res):
        if i == 5:
            break

        print(f"{i + 1}. {movie}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyword Search CLI"
        )
    
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies using keywords"
    )

    search_parser.add_argument(
        "query", type=str, help="Search query"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            clean_query = "".join(
                  args.query.lower().translate(
                      str.maketrans("", "", string.punctuation)
                   )
               )
            
            search("data/movies.json", clean_query.lower().split())
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()