import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import base64
from openai.types.chat import ChatCompletionMessageParam


def describe_image_command(query, mime):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    with open("/home/mzougari/Desktop/Rag-Search-Engine/data/paddington.jpeg", "rb") as f:
        img = f.read()

    with open("/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/question_answering.md", "r") as f:
        system_prompt = f.read()

        
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]
    model = "openrouter/free"

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("No text in API response")

    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")
