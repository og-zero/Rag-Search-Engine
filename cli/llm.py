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


def re_ranking(prompte, query, title, document):
    prompte = prompte.format(query=query, title=title, document=document)
    model='openrouter/free'
    messages = [
        {
            "role": "user",
            "content": prompte,
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def batch_re_ranking(prompte, query, doc_list_str):
    prompte = prompte.format(query=query, doc_list_str=doc_list_str)
    model='openrouter/free'
    messages = [
        {
            "role": "user",
            "content": prompte,
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content

    if content is None:
        raise ValueError("The model returned an empty response")

    return content.strip()


def llm_evaluation(prompte, query, result):
    prompte = prompte.format(query=query, formatted_results=result)
    model='openrouter/free'
    messages = [
        {
            "role": "user",
            "content": prompte,
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content

    if content is None:
        raise ValueError("The model returned an empty response")
    
    return content

def Augmented_Generation(prompte, query, docs):
    prompte = prompte.format(query=query, docs=docs)
    model='openrouter/free'
    messages = [
        {
            "role": "user",
            "content": prompte,
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content

    if content is None:
        raise ValueError("The model returned an empty response")

    return content.strip()
    

def fetch_rag_prompte(path, query, docs):
    with open(Path(path), 'r') as f:
        prompte = f.read()

    return Augmented_Generation(prompte, query, docs)


def fetch_llm_evl_prompte(path, query, result):
    with open(Path(path), 'r') as f:
        prompte = f.read()

    return llm_evaluation(prompte, query, result)

def fetch_batch_rr_prompte(path, query, list_docs):
    with open(Path(path), 'r') as f:
        prompte = f.read()
    
    return batch_re_ranking(prompte, query, list_docs)

def fetch_rr_prompte(path, query, title, document):
    with open(Path(path), 'r') as f:
        prompte = f.read()

    return re_ranking(prompte, query, title, document)

def fetch_prompte(query, path):
    with open(Path(path), 'r') as f:
        prompte = f.read()

    return enhance_query(prompte, query)