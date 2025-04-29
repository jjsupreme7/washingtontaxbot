from fastapi import FastAPI
from pydantic import BaseModel
from chatbot2 import process_query, get_all_filenames  # <-- Import your real code
from dotenv import load_dotenv
import os
from pinecone import Pinecone

app = FastAPI()

# Load environment variables
load_dotenv()

# Set up Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Preload filenames once (otherwise too slow every time)
all_filenames = get_all_filenames(index)
conversation_context = []

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_taxbot(request: QuestionRequest):
    question = request.question
    answer = process_query(question, all_filenames, conversation_context)
    return {"answer": answer}

