import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def enhance_query(prompte, query):
    prompte = prompte.format(query=query)
    model='openrouter/free'
    messages = [
        {
            "role": "user",
            "content": prompte,
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def fetch_prompte(query, path):
    with open(Path(path), 'r') as f:
        prompte = f.read()

    return enhance_query(prompte, query)