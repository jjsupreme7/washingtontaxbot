from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot2 import process_query, get_all_filenames
from dotenv import load_dotenv
import os
from pinecone import Pinecone

app = FastAPI()

# Load environment variables
load_dotenv()

# Set up Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Preload filenames once
all_filenames = get_all_filenames(index)
conversation_context = []

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_taxbot(request: QuestionRequest):
    try:
        question = request.question
        answer = process_query(question, all_filenames, conversation_context)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


