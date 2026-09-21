import argparse
import sys
from inverted_index import InvertedIndex
import tfidf

def handle_build(args):
    idx = InvertedIndex()
    idx.build()
    idx.save()

def handle_tf(args):
    idx = InvertedIndex()
    idx.load()
    score = tfidf.get_tf(idx, args.doc_id, args.term)
    print(f"Term frequency of '{args.term}' in document {args.doc_id}: {score}")

def handle_idf(args):
    idx = InvertedIndex()
    idx.load()
    score = tfidf.get_idf(idx, args.term)
    print(f"Inverse document frequency of '{args.term}': {score:.2f}")

def handle_tfidf(args):
    idx = InvertedIndex()
    idx.load()
    score = tfidf.get_tfidf(idx, args.doc_id, args.term)
    print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Keyword Search Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and cache inverted index")
    build_parser.set_defaults(func=handle_build)


    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")
    tf_parser.set_defaults(func=handle_tf)

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")
    idf_parser.set_defaults(func=handle_idf)

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")
    tfidf_parser.set_defaults(func=handle_tfidf)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()