from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot2 import process_query, get_all_filenames
from dotenv import load_dotenv
import os
import uvicorn
from pinecone import Pinecone
import nltk

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Configure Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Preload filenames and setup conversation context
all_filenames = get_all_filenames(index)
conversation_context = []

# Configure NLTK Data Directory
nltk_data_path = "/opt/render/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Define request/response schemas
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# Main /ask endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_taxbot(request: QuestionRequest):
    try:
        question = request.question
        answer = process_query(question, all_filenames, conversation_context)
        return {"answer": answer}
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
