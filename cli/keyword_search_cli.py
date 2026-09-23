import argparse
import json
import string 
from pathlib import Path
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
BM25_K1 = 1.5
BM25_B = 0.75

def clean_text(Text: str) -> str:
    clean_title = Text.translate(
        str.maketrans("", "", string.punctuation)
    ).lower()

    return clean_title


def tokenization(Text: str) -> list:
    return Text.split()


def load_stopwords() -> list:

    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        content = f.read()
        stopwords = content.splitlines()

    clean_stopwords = [clean_text(stop_w) for stop_w in stopwords]

    return clean_stopwords


def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)
            
    return data['movies']
    

def filter_stopwords(tokens: list, stopwords: list) -> list:
    return [token for token in tokens if token not in stopwords]


def tokenize_text(Text: str) -> list:
    clean_title = clean_text(Text)
    tokens = tokenization(clean_title)
    stopwords = load_stopwords()
    tokens = filter_stopwords(
        tokens, stopwords
        )
    return tokens


def search(search_query: str) -> None:
    from cli.inverted_index import InvertedIndex

    inverted_index = InvertedIndex()
    inverted_index.load()
    
    tokens_query = tokenize_text(search_query)
    stemmed_query = [stemmer.stem(token) 
                    for token in tokens_query]

    seen = set()
    result = []  

    for token_q in stemmed_query:
        indexes = inverted_index.get_documents(token_q)
        for index in indexes:
            if index in seen:
                continue
            seen.add(index)
            matching_doc = inverted_index.docmap[index]
            result.append(matching_doc)

    for i, movie in enumerate(result[:5]):
        print(f"{i + 1}. {movie}")


def main() -> None:
    from cli.inverted_index import InvertedIndex

    inverted_index = InvertedIndex()

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

    build_parser = subparsers.add_parser(
        "build", help="Build the inverted index"
    )

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check")

    idf_parser = subparsers.add_parser("idf", help="Get Inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to check")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get Inverse document frequency")
    tfidf_parser.add_argument("doc_id", type=int, help="dDocument ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
            pass
        case "build":
            inverted_index.build_command()
        case "tf":
            inverted_index.build_tf_command(
                args.doc_id, args.term
            )
        case "idf":
            inverted_index.build_idf_command(
                args.term
            )
        case "tfidf":
            inverted_index.build_tfifd_command(
                args.doc_id, args.term
            )
        case "bm25idf":
            inverted_index.build_bm25idf_command(
                args.term
            )
        case "bm25tf":
            inverted_index.bm25_tf_command(
                args.doc_id, args.term, args.k1
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()