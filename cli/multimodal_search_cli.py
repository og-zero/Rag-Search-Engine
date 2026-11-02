import argparse
from cli.lib.multimodal_search import embed_image_command

def main() -> None:

    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("image_search", help="generating an embedding for the images")
    verify_parser.add_argument(
        "image",
        help="Path to the image file"
    )

    args = parser.parse_args()

    match args.command:
        case "image_search":
            embed_image_command(
                args.image
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()