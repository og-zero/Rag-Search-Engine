import argparse
import json
import string 
from pathlib import Path



def search(File_path: str, search_query: str) -> None:
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

                if search_query in clean_title:
                    res.append(d["title"])

            
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

    clean_query = "".join(
       args.query.lower().translate(
           str.maketrans("", "", string.punctuation)
        )
    )

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            pass
        case _:
            parser.print_help()

    search("data/movies.json", clean_query.lower())

if __name__ == "__main__":
    main()