import argparse
import json
import string 
from pathlib import Path
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex



def clean_text(Text: str) -> str:
    clean_title = Text.translate(
        str.maketrans("", "", string.punctuation)
    ).lower()

    return clean_title


def tokenization(Text: str) -> list:

    return Text.split()


def tokenize_text(Text: str) -> list:
    clean_title = clean_text(Text)
    tokens = tokenization(clean_title)

    return tokens


def load_stopwords(File_path: str) -> list:
    path = Path(File_path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        stopwords = content.splitlines()

    clean_stopwords = [clean_text(stop_w) for stop_w in stopwords]

    return clean_stopwords


def load_movies():
    
    with open("data/movies.json", "r", encoding="utf-8") as f:
            movies = json.load(f)
            
    return movies
    
    
def filter_stopwords(tokens: list, stopwords: list) -> list:
    return [token for token in tokens if token not in stopwords]


def search(stopwords_path: str, 
           search_query: str
        ) -> None:
    stemmer = PorterStemmer()

    stopwords = load_stopwords(stopwords_path)
    clean_query = clean_text(search_query)
    query_tokens = tokenization(clean_query)
    filtered_query_tokens = filter_stopwords(
        query_tokens, stopwords
    )

    movies = load_movies()

    result = []
    for category in movies.values():
        for movie in category:
            if "title" in movie:
                clean_title = clean_text(movie["title"])
                title_tokens = tokenization(clean_title)
                filtered_title_tokens = filter_stopwords(
                    title_tokens, stopwords
                )

            if any(stemmer.stem(q_token) in stemmer.stem(t_token)
                    for q_token in filtered_query_tokens 
                    for t_token in filtered_title_tokens
                ):
                result.append(movie["title"])        

    for i, movie in enumerate(result):
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

            search("data/stopwords.txt", 
                args.query
            )
            pass
        case "build":
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()