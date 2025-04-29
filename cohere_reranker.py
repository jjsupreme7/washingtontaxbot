"""
Cohere reranking implementation for Washington Tax Chatbot.
"""

import os
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Cohere API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize the Cohere client
co = cohere.Client(COHERE_API_KEY)

def cohere_rerank(query, documents, top_n=10):
    """
    Rerank documents using Cohere's reranking API.
    
    Args:
        query (str): The user's query
        documents (list): List of document chunks with their text
        top_n (int): Number of documents to return after reranking
    
    Returns:
        list: Reranked document chunks
    """
    try:
        # Prepare documents for reranking
        docs_for_reranking = []
        for doc in documents:
            # Get document text
            doc_id = doc['id']
            
            # Try to get full text from retrieve_document_content function
            try:
                from question_type import retrieve_document_content
                text = retrieve_document_content(doc_id)
            except ImportError:
                text = None
                
            # Fall back to metadata text if needed
            if not text:
                text = doc['metadata'].get('text', '').replace("\n", " ").strip()
            
            # Skip if no text
            if not text:
                continue
                
            # Add to reranking list
            docs_for_reranking.append({
                'text': text[:1000],  # Limit text length to avoid token limits
                'doc_obj': doc  # Keep reference to original doc object
            })
        
        if not docs_for_reranking:
            print("No documents to rerank")
            return documents
        
        # Call Cohere reranking API
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[doc['text'] for doc in docs_for_reranking],
            top_n=min(top_n, len(docs_for_reranking))
        )
        
        # Process reranked results
        reranked_docs = []
        for result in rerank_response.results:
            # Get the original document with all metadata
            original_doc = docs_for_reranking[result.index]['doc_obj']
            
            # Add reranking score to document
            original_doc['relevance_score'] = result.relevance_score
            reranked_docs.append(original_doc)
        
        return reranked_docs
    
    except Exception as e:
        print(f"Error during Cohere reranking: {e}")
        # Fallback to original documents if reranking fails
        return documents

def enhanced_document_retrieval_with_reranking(query, all_filenames, index, docs_needed=10, relevance_threshold=0.3):
    """
    Enhanced document retrieval with Cohere reranking.
    
    Args:
        query (str): The user's query
        all_filenames (list): List of all available filenames
        index: Pinecone index
        docs_needed (int): Number of documents to retrieve
        relevance_threshold (float): Minimum relevance score to include a document
    
    Returns:
        list: Retrieved and reranked document chunks
    """
    # Import functions from chatbot.py
    from chatbot import (
        enhanced_document_retrieval,
        assess_document_relevance,
        enhanced_fuzzy_match_documents,
        get_embedding,
        query_pinecone_by_filename,
        hybrid_query_by_documents
    )
    
    # Step 1: Initial fuzzy matching
    matched_files = enhanced_fuzzy_match_documents(query, all_filenames)
    
    # Step 2: Retrieve chunks for matched files (get more for reranking)
    if matched_files:
        query_vector = get_embedding(query)
        initial_chunks = query_pinecone_by_filename(index, matched_files, query_vector, top_k=docs_needed * 2)
    else:
        # Fallback to hybrid query if no fuzzy matches
        initial_chunks = hybrid_query_by_documents(query, index, docs_needed * 2)
    
    # Step 3: Assess basic document relevance to filter out obviously irrelevant docs
    relevance_scored_chunks = []
    for chunk in initial_chunks:
        # Retrieve full text
        chunk_id = chunk['id']
        try:
            from question_type import retrieve_document_content
            text = retrieve_document_content(chunk_id)
        except ImportError:
            text = None
            
        if not text:
            text = chunk['metadata'].get('text', '').replace("\n", " ").strip()
        
        # Calculate relevance score
        relevance_score = assess_document_relevance(query, text)
        
        # Add relevance information to chunk
        chunk['relevance_score'] = relevance_score
        
        # Only keep chunks above relevance threshold
        if relevance_score >= relevance_threshold:
            relevance_scored_chunks.append(chunk)
    
    # Step 4: Apply Cohere reranking to the relevant chunks
    reranked_chunks = cohere_rerank(query, relevance_scored_chunks, top_n=docs_needed)
    
    return reranked_chunks



