import openai
import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Use OpenAI's tokenizer for the embedding model
tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
MAX_TOKENS = 8191  # 1 token less than max

def get_embedding(text):
    tokens = tokenizer.encode(text)

    # Truncate if over token limit
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = tokenizer.decode(tokens)

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return list(response.data[0].embedding)  # ✅ Convert to list explicitly



