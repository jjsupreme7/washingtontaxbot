import os
import re
import csv
import time
import zipfile
import json
import sys
import shutil
import math
import datetime
import numpy as np
from dateutil import parser
from openai import OpenAI
from dotenv import load_dotenv
from legal_patterns import LEGAL_PATTERNS
from cohere_reranker import cohere_rerank
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from pinecone import Pinecone
from rapidfuzz import process, fuzz
from embedder import get_embedding
from keyword_extraction import extract_keywords
from citation_utils import enhanced_legal_citation_extraction, extract_citations_from_chunks, format_citations_summary
from document_matcher import enhanced_fuzzy_match_documents

# NEW IMPORT FOR CHUNK_METADATA-BASED FILTERING
import chromadb
from chromadb import PersistentClient

with open("metadata/chunk_metadata.json") as f:
    METADATA_DB = json.load(f)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# 🔁 Only unzip if _chroma_storage isn't already present
if not os.path.exists("_chroma_storage"):
    print("📦 Recombining ChromaDB split archive...")

    # Create the full zip file from parts
    with open("chroma_full.zip", "wb") as f_out:
        for i in range(1, 100):  # adjust if more than 99 parts
            part_ext = f"z{str(i).zfill(2)}"
            part_file = f"chroma_split_24mb.{part_ext}"
            if not os.path.exists(part_file):
                break
            with open(part_file, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)

        # Append the final .zip tail
        with open("chroma_split_24mb.zip", "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)

    print("📂 Extracting ChromaDB from chroma_full.zip...")
    with zipfile.ZipFile("chroma_full.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Updated: Use PersistentClient for ChromaDB
chroma_client = PersistentClient(path="./_chroma_storage")

chroma_collection = chroma_client.get_or_create_collection(name="taxlaw")

EMBED_DIM = 3072

def get_metadata_by_filename(filename):
    for entry in METADATA_DB:
        if entry.get("document_id") == filename:
            return entry
    return {}

def extract_keywords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    return list(set(keywords))

def match_legal_patterns(text):
    matches = []
    for label, pattern in LEGAL_PATTERNS.items():
        if re.search(pattern["regex"], text, re.IGNORECASE):
            matches.append((label, pattern["weight"]))
    return matches


def build_topic_filter(query, doc_ids):
    topics = extract_keywords(query)
    rule_refs = enhanced_legal_citation_extraction(query)
    metadata_filter = {}
    if topics:
        metadata_filter["tags"] = {"$in": list(set(topics))}
    if rule_refs:
        rules = rule_refs.get("rcw", []) + rule_refs.get("wac", []) + rule_refs.get("eta", [])
        if rules:
            metadata_filter["rule_cited"] = {"$in": rules}
    if doc_ids:
        metadata_filter["document_id"] = {"$in": doc_ids}
    return metadata_filter or None



def get_chunk_texts_by_ids(chunk_ids):
    result = chroma_collection.get(ids=chunk_ids)
    return result['documents'] if result and result.get('documents') else []

def get_chunk_text_by_id(chunk_id):
    texts = get_chunk_texts_by_ids([chunk_id])
    return texts[0] if texts else ""

def add_in_text_citations(answer, chunk_text_pairs):
    """
    Add in-text citations to the generated answer by matching meaningful phrases or quotes
    from the retrieved chunk texts.

    Args:
        answer (str): The answer text from GPT
        chunk_text_pairs (list): List of (chunk_id, chunk_text) tuples

    Returns:
        str: Annotated answer with in-text citations and source appendix
    """
    cited = []
    seen_texts = set()
    for i, (chunk_id, chunk_text) in enumerate(chunk_text_pairs, 1):
        # Extract the first few clear fragments to check
        phrases = re.split(r'[.!?]', chunk_text[:500])[:2]  # Just two short chunks
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase and phrase in answer and phrase not in seen_texts:
                answer = answer.replace(phrase, f"{phrase} [{i}]")
                cited.append((i, chunk_id, phrase))
                seen_texts.add(phrase)
                break  # Only cite once per chunk

    if cited:
        appendix = "\n\n📎 **In-Text Sources Cited:**\n"
        for i, chunk_id, phrase in cited:
            appendix += f"- [{i}] {chunk_id}: {phrase[:140].strip()}...\n"
        answer += appendix

    return answer



def search_index(query, top_k=30):
    # Embed the query and retrieve top matches
    # You will need to replace this with your actual embedding logic
    pinecone_results = index.query(
        vector=[],  # ← Replace with actual embedding of the query
        top_k=top_k,
        include_metadata=True
    )

    docs = []
    for result in pinecone_results["matches"]:
        doc_text = result["metadata"]["text"]
        relevance_score = assess_document_relevance(doc_text, query)
        docs.append({
            "id": result["id"],
            "text": doc_text,
            "score": relevance_score,
            "metadata": result["metadata"]
        })

    reranked = rerank_with_cohere(query, docs)
    return reranked


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1 (list): First vector
        vec2 (list): Second vector
    
    Returns:
        float: Cosine similarity between the vectors
    """
    # Ensure vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

def assess_document_relevance(doc_text, query, metadata=None):
    """
    Computes a weighted relevance score based on:
    - keyword overlap
    - legal pattern overlap
    - optional metadata (rule cited, tags)
    """

    # 🔧 Weights
    keyword_weight = 0.3
    pattern_weight = 0.5
    rule_match_weight = 0.1
    tag_match_weight = 0.1

    # 🧠 Step 1: Keyword overlap
    query_keywords = extract_keywords(query)
    doc_keywords = extract_keywords(doc_text)
    matched_keywords = set(query_keywords) & set(doc_keywords)
    keyword_score = len(matched_keywords) / max(len(query_keywords), 1)

    # 🧠 Step 2: Legal pattern match
    query_patterns = match_legal_patterns(query)
    doc_patterns = match_legal_patterns(doc_text)
    matched_labels = set(label for label, _ in query_patterns) & set(label for label, _ in doc_patterns)
    pattern_score = sum(weight for label, weight in query_patterns if label in matched_labels) / 10

    # 🧠 Step 3: Metadata-aware boosts (optional)
    rule_boost = 0
    tag_boost = 0

    if metadata:
        # ✅ Rule matching (for WTDs)
        cited_rules = metadata.get("rule_cited", [])
        if cited_rules:
            from citation_utils import enhanced_legal_citation_extraction
            query_rules = enhanced_legal_citation_extraction(query)
            query_all_rules = query_rules.get("rcw", []) + query_rules.get("wac", []) + query_rules.get("eta", [])
            if any(rule in cited_rules for rule in query_all_rules):
                rule_boost = 1

        # ✅ Tag matching
        tags = metadata.get("tags", [])
        if any(tag.lower() in query.lower() for tag in tags):
            tag_boost = 1

    # 🧮 Final score
    total_score = (
        keyword_weight * keyword_score +
        pattern_weight * pattern_score +
        rule_match_weight * rule_boost +
        tag_match_weight * tag_boost
    )

    return total_score


    

def rerank_with_cohere(query, docs):
    return cohere_rerank(query=query, documents=docs)


def enhanced_document_retrieval(query, all_filenames, index, docs_needed=10, relevance_threshold=0.3):
    """
    Enhanced document retrieval with relevance scoring and fallback mechanisms.

    Args:
        query (str): The user's query
        all_filenames (list): List of all available filenames
        index: Pinecone index
        docs_needed (int): Number of documents to retrieve
        relevance_threshold (float): Minimum relevance score to include a document

    Returns:
        list: Retrieved and relevance-filtered document chunks
    """
    # Step 1: Initial fuzzy matching
    matched_files = enhanced_fuzzy_match_documents(query, all_filenames)

    # Step 2: Retrieve chunks for matched files
    if matched_files:
        query_vector = get_embedding(query)
        initial_chunks = query_pinecone_by_filename(index, matched_files, query_vector)
        #print("🔍 Retrieved from Pinecone:", initial_chunks)  # Logging line to show text from Pinecone
    else:
        # Fallback to hybrid query if no fuzzy matches
        initial_chunks = hybrid_query_by_documents(query, index, docs_needed)
        #print("🔍 Retrieved from Hybrid Query:", initial_chunks)  # Logging line to show text from Hybrid query

    # Step 3: Assess document relevance
    relevance_scored_chunks = []
    for chunk in initial_chunks:
        chunk_id = chunk['id']
        #print(f"🔍 Processing chunk {chunk_id}...")

        # ✅ Check if text is available in ChromaDB (if available)
        text = get_chunk_text_by_id(chunk_id)
        
        if text:
            #print(f"🔍 Retrieved text from ChromaDB for chunk {chunk_id}: {text}")
            pass
        else:
            # If not found in ChromaDB, fallback to Pinecone metadata
            text = chunk['metadata'].get('text', '').replace("\n", " ").strip()
            #print(f"🔍 Retrieved text from Pinecone metadata for chunk {chunk_id}: {text}")
            pass

        # Score and filter based on relevance
        relevance_score = assess_document_relevance(text, query)

        chunk['relevance_score'] = relevance_score

        if relevance_score >= relevance_threshold:
            relevance_scored_chunks.append(chunk)

    # Step 4: Sort and return top N
    relevance_scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
    #print(f"🔍 Final sorted results: {relevance_scored_chunks[:docs_needed]}")
    
    return relevance_scored_chunks[:docs_needed]





def multiline_input(prompt):
    """
    Allows multi-line input with flexible pasting.
    
    Args:
        prompt (str): The initial prompt to display
    
    Returns:
        str: The complete multi-line input
    """
    print(prompt)
    print("(Paste your question. Press Enter twice to finish)")
    
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    return ' '.join(lines)

def extract_date_from_metadata(metadata):
    """Extract date from various metadata fields."""
    date_patterns = [
        # Format: YYYY-MM-DD or YYYYMMDD
        r'(\d{4}[-/]?\d{2}[-/]?\d{2})',
        # Format: MM-DD-YYYY or MM/DD/YYYY
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        # Format: Month names
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+\d{1,2}[\s,]+\d{4}',
        # Format: Year only
        r'\b(19\d{2}|20\d{2})\b'
    ]
    
    # Sources to check for dates
    sources = [
        metadata.get('filename', ''),
        metadata.get('chunk_id', ''),
        metadata.get('text', '')[:200]  # Check beginning of text
    ]
    
    for source in sources:
        for pattern in date_patterns:
            matches = re.findall(pattern, source)
            if matches:
                # Parse the matched date
                try:
                    # Convert extracted date to timestamp
                    date_str = matches[0]
                    if isinstance(date_str, tuple):
                        date_str = date_str[0]
                    
                    # Handle different date formats
                    if re.match(r'\d{4}[-/]?\d{2}[-/]?\d{2}', date_str):
                        date_str = re.sub(r'[/-]', '', date_str)
                        date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
                    elif re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', date_str):
                        # Convert MM/DD/YYYY to datetime
                        date_parts = re.split(r'[-/]', date_str)
                        date_obj = datetime.datetime(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]))
                    elif re.match(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+\d{1,2}[\s,]+\d{4}', date_str):
                        # Process month name format
                        date_obj = datetime.datetime.strptime(date_str, '%b %d, %Y')
                    elif re.match(r'\b(19\d{2}|20\d{2})\b', date_str):
                        # Just a year - set to middle of year
                        date_obj = datetime.datetime(int(date_str), 6, 15)
                    else:
                        continue
                    
                    return date_obj.timestamp()
                except Exception as e:
                    continue
    
    # Default to oldest possible date if no date found
    return 0

def get_all_filenames(index, sample_size=10000):
    results = index.query(vector=[0.0]*EMBED_DIM, top_k=sample_size, include_metadata=True)
    return list({match['metadata'].get('document_id') or match['id'] for match in results['matches'] if match})




def query_pinecone_by_filename(index, filenames, query_vector, top_k=10):
    results = []
    for doc_id in filenames:
        res = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"document_id": {"$eq": doc_id}}
        )
        results.extend(res['matches'])
    return results

def detect_doc_ids_in_query(query_text, known_doc_ids):
    return [doc_id for doc_id in known_doc_ids if doc_id.lower() in query_text.lower()]



def hybrid_query_by_documents(query_text, index, docs_needed=10, metadata_filter=None,
                               recency_weight=0.0, doc_id_mode="ignore", filtered_doc_ids=None):
    """
    Enhanced hybrid query with support for strict filtering, boosting, and recency reweighting.

    Args:
        query_text (str): The user’s query
        index: Pinecone index
        docs_needed (int): Number of documents to return
        metadata_filter (dict): Optional metadata filter (e.g., doc_type, tags, date)
        recency_weight (float): How much to prioritize recent documents (0-1)
        doc_id_mode (str): "strict", "boost", or "ignore"
        filtered_doc_ids (list): List of document_ids the user filtered by (can be empty)

    Returns:
        List of top ranked Pinecone matches
    """
    import time

    query_vector = get_embedding(query_text)
    filtered_doc_ids = set(filtered_doc_ids or [])

    # 🔍 Step 1: Perform broad search (apply metadata filter if available)
    broad_query = {
        "vector": query_vector,
        "top_k": docs_needed * 3,
        "include_metadata": True,
    }

    if metadata_filter:
        broad_query["filter"] = metadata_filter  # 🔥 ENFORCE strict metadata filtering

    try:
        broad_results = index.query(**broad_query).get("matches", [])
    except Exception as e:
        print("❌ Error during broad query:", e)
        return []

    # 🔍 Step 2: Perform filtered doc ID search (only if doc_id_mode is "strict" or "boost")
    filtered_results = []
    if doc_id_mode in ["strict", "boost"] and filtered_doc_ids:
        try:
            filter_query = {
                "vector": query_vector,
                "top_k": docs_needed * 2,
                "include_metadata": True,
                "filter": {
                    "document_id": {"$in": list(filtered_doc_ids)}
                }
            }
            if metadata_filter:
                # 🔥 Merge metadata_filter with doc_id filter
                filter_query["filter"].update(metadata_filter)

            filtered_results = index.query(**filter_query).get("matches", [])
        except Exception as e:
            print("❌ Error during filtered query:", e)

    # 🧠 Step 3: Decide match list based on doc_id_mode
    if doc_id_mode == "strict" and filtered_doc_ids:
        matches = filtered_results
    else:
        combined = {match['id']: match for match in (broad_results + filtered_results)}.values()

        # Boost manually if document ID matches
        for match in combined:
            doc_id = match.get('metadata', {}).get('document_id')
            if doc_id in filtered_doc_ids:
                match['score'] = match.get('score', 0) + 0.2  # manual boost

        matches = list(combined)

    # 🕒 Step 4: Apply recency reweighting
    if recency_weight > 0:
        current_time = time.time()
        for match in matches:
            metadata = match.get("metadata") or {}
            date_score = extract_date_from_metadata(metadata)
            recency_factor = min(1, max(0, 1 - ((current_time - date_score) / 94608000)))  # 3 years
            match["score"] = match["score"] * (1 - recency_weight) + recency_factor * recency_weight

    # 📊 Step 5: Sort by final score and return
    matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
    return matches[:docs_needed]




def format_sources(matches):
    seen = set()
    sources = []
    for match in matches:
        filename = match['metadata'].get('filename')
        if filename and filename not in seen:
            seen.add(filename)
            sources.append(filename)
    return sources

def add_in_text_citations(answer, chunk_text_pairs):
    """
    Add in-text citations to the generated answer by matching quotes or facts with source chunks.
    `chunk_text_pairs` is a list of (chunk_id, text) tuples.
    """
    cited_chunks = []
    for chunk_id, text in chunk_text_pairs:
        # Check if some portion of the chunk is directly quoted or strongly resembles text in the answer
        if any(fragment.strip() in answer for fragment in text.split('.')[:3]):
            cited_chunks.append((chunk_id, text))

    if not cited_chunks:
        return answer  # No match found

    # Append inline citation tags at the bottom
    sources_section = "\n\n📎 **In-Text Sources Cited:**\n"
    for i, (chunk_id, text) in enumerate(cited_chunks, 1):
        preview = text[:180].replace("\n", " ").strip()
        sources_section += f"- [{i}] {chunk_id}: {preview}...\n"

    return answer + sources_section


def ask_chatgpt(question, chunks):
    context_parts = []
    all_chunk_texts = []

    for i, chunk in enumerate(chunks, 1):
        filename = chunk['metadata'].get('filename', 'Unknown')
        chunk_id = chunk['id']

        text = get_chunk_text_by_id(chunk_id)
        if not text:
            text = chunk['metadata'].get('text', '').replace("\n", " ").strip()

        all_chunk_texts.append((chunk_id, text))
        citations = enhanced_legal_citation_extraction(text)

        citation_info = []
        if citations['rcw']:
            citation_info.append(f"RCW: {', '.join(citations['rcw'])}")
        if citations['wac']:
            citation_info.append(f"WAC: {', '.join(citations['wac'])}")
        if citations['wtd']:
            citation_info.append(f"WTD: {', '.join(citations['wtd'])}")

        citation_info_text = f" ({'; '.join(citation_info)})" if citation_info else ""
        context_parts.append(f"[{i}] Source: {filename}{citation_info_text}\n{text}")

    context = "\n\n".join(context_parts)[:24000]

    # ⬇️ Replace system_prompt with Iris's persona
    iris_prompt = """
You are Iris, a legal AI trained specifically on Washington State tax law, including RCWs, WACs, WTDs, ETAs, and court rulings. You assist licensed Washington tax attorneys who already understand legal context—your job is to be clear, confident, well-cited, and insightful.

Role and Goal:
- Your primary role is to provide legally accurate, citation-supported answers to questions involving Washington State tax law.
- You specialize in interpreting and applying RCWs, WACs, WTDs, ETAs, and court rulings to complex legal scenarios.
- Your goal is to assist attorneys by delivering clear, structured, and insightful legal analysis tailored to their advanced understanding.

Constraints:
- Only provide information relevant to Washington State tax law.
- All answers must include precise citations, such as RCW 82.04.460(1)(b), WAC 458-20-19402(303)(c)(ii), Det. No. 20-0123, 39 WTD 240 (2020), or ETA 3199.2017.
- Avoid providing personal tax advice or addressing issues outside Washington State jurisdiction.
- Do not provide legal basics unless explicitly requested.

Guidelines:
- For cascading hierarchies in the law (e.g., WAC 458-20-19402), walk through each step and explain its application.
- Distinguish between market-based attribution and business location-based attribution with precision.
- Identify whether a reasonable proportional method was used or if fallback provisions (e.g., subsections (c)(ii) or (c)(iii)) apply.
- When addressing client-specific scenarios, use a structured legal analysis format:
  - **Issue**: Identify the legal question.
  - **Rule**: Cite the governing statutes, rules, or determinations.
  - **Application**: Apply the rules to the facts provided.
  - **Counterarguments**: Present both taxpayer and DOR positions where applicable.
  - **Conclusion**: State the most likely legal outcome or defensible position, including risks of multiple interpretations.
- When building or rebutting arguments, present both taxpayer and Department of Revenue perspectives, flagging weak documentation or unclear facts.
- Recognize and summarize referenced documents from the connected document store, applying them to the question or fact pattern with precise citations.

Clarification:
- If a question is unclear or lacks sufficient detail, request clarification to ensure accuracy.
- If the law is unsettled, explain the ambiguity and provide possible interpretations.

Personalization:
- Maintain a professional, confident, and structured tone.
- Use bullet points, headings, and clean formatting for readability.
- Avoid verbosity; focus on clarity and precision to suit the needs of legal professionals.

Special Instructions:
- Never hallucinate citations. If uncertain, clearly state what is missing or needed (e.g., “The Department has not issued guidance directly on this issue, but…”).
- Stay updated on changes to Washington State tax laws and regulations.
- Assume the user is highly trained; avoid oversimplification unless requested.
"""

    messages = [
        {"role": "system", "content": iris_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.2
    )

    generated_answer = response.choices[0].message.content.strip()
    citations = enhanced_legal_citation_extraction(generated_answer)

    if citations['rcw']:
        generated_answer += f"\n\n**Referenced RCWs:**\n" + "\n".join([f"- {r}" for r in citations['rcw']])
    if citations['wac']:
        generated_answer += f"\n\n**Referenced WACs:**\n" + "\n".join([f"- {w}" for w in citations['wac']])
    if citations['wtd']:
        generated_answer += f"\n\n**Referenced WTDs:**\n" + "\n".join([f"- {w}" for w in citations['wtd']])

    final_answer = add_in_text_citations(generated_answer, all_chunk_texts)

    source_list = "\n\n📎 **Sources Used:**\n"
    for i, chunk in enumerate(chunks, 1):
        filename = chunk['metadata'].get('filename', 'Unknown')
        chunk_id = chunk['id']
        preview_text = get_chunk_text_by_id(chunk_id)[:200].replace("\n", " ").strip()
        source_list += f"- [{i}] {filename} ({chunk_id}): {preview_text}...\n"

    final_answer += source_list
    return final_answer


def convert_to_timestamp(date_str):
    try:
        if not date_str:
            return None
        dt = parser.parse(date_str, default=datetime.datetime(2000, 1, 1))
        return int(dt.timestamp())
    except Exception:
        return None

def process_query(query, all_filenames, conversation_context):
    print(f"🔑 Extracted topics from query: {extract_keywords(query)}")
    doc_ids = detect_doc_ids_in_query(query, all_filenames)
    metadata_filter = build_topic_filter(query, doc_ids)
    chunks = hybrid_query_by_documents(
        query_text=query,
        index=index,
        docs_needed=10,
        metadata_filter=metadata_filter,
        recency_weight=0.2,
        doc_id_mode="boost",
        filtered_doc_ids=doc_ids
    )
    if not chunks:
        return "❌ No relevant information found."
    return ask_chatgpt(query, chunks)




def prepare_contextual_query(follow_up_query, conversation_context, max_context_length=3):
    """
    Prepare a contextual query by incorporating previous conversation context.

    Args:
        follow_up_query (str): The new follow-up query
        conversation_context (list): Previous conversation interactions
        max_context_length (int): Maximum number of previous interactions to include

    Returns:
        str: Enhanced query with contextual information
    """
    # Take the last max_context_length interactions
    recent_context = conversation_context[-max_context_length:]

    # Construct context string
    context_str = "Previous conversation context:\n"
    for interaction in recent_context:
        context_str += f"Question: {interaction['query']}\n"
        context_str += f"Previous Answer: {interaction['response'][:500]}...\n\n"

    # Combine context with follow-up query
    enhanced_query = f"{context_str}Follow-up Question: {follow_up_query}"

    return enhanced_query



def main():
    print("\n🗂️ Washington Tax Chatbot (GPT-4 + Pinecone Hybrid + Enhanced Citations)\n")

    all_filenames = get_all_filenames(index)
    conversation_context = []

    while True:
        query = multiline_input("💬 Enter your multi-line question:") if input("💬 Do you want to enter a multi-line question? (y/n): ").lower() == 'y' else input("💬 Ask a question (or type 'exit'): ")

        if query.lower() in ["exit", "quit"]:
            break

        # ✅ Always call process_query — it will ask about filters internally
        answer = process_query(query, all_filenames, conversation_context)

        print("\n💬 **Answer:**")
        print(answer)

        conversation_context.append({'query': query, 'response': answer})

        while True:
            follow_up = input("\n💬 Do you want to ask a follow-up question? (y/n): ").lower()
            if follow_up != 'y':
                break

            follow_up_query = multiline_input("💬 Enter your follow-up question:") if input("💬 Do you want to enter a multi-line follow-up question? (y/n): ").lower() == 'y' else input("💬 Enter your follow-up question: ")
            enhanced_follow_up = prepare_contextual_query(follow_up_query, conversation_context)

            # ✅ Again: always call process_query (it handles filters inside)
            answer = process_query(enhanced_follow_up, all_filenames, conversation_context)

            print("\n💬 **Answer:**")
            print(answer)

            conversation_context.append({'query': follow_up_query, 'response': answer})


#if __name__ == "__main__":
    #main()

