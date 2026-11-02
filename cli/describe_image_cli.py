import argparse
from cli.lib.describe_image import describe_image_command
import mimetypes

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )

    parser.add_argument(
        "--image",
        help="the path to an image file"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="a text query to rewrite based on the image"
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    describe_image_command(
        args.query, mime
    )

if __name__ == "__main__":
    main()