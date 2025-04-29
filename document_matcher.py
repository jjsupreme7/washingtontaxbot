import os
import re
from rapidfuzz import process, fuzz

def get_all_filenames(index, sample_size=1000):
    """
    Retrieve unique document identifiers (formerly filenames) from the Pinecone index metadata.
    Now uses 'document_id' as the canonical identifier for matching.
    """
    try:
        results = index.query(
            vector=[0.0] * 3072,  # Use dummy vector
            top_k=sample_size,
            include_metadata=True
        )

        all_ids = set()
        for match in results['matches']:
            doc_id = match['metadata'].get('document_id') or match['id']
            if doc_id:
                all_ids.add(doc_id)

        if not all_ids:
            print("⚠️ Warning: No document_ids found in the index.")
        else:
            print(f"🔍 Index contains approximately {len(all_ids)} unique document identifiers.")

        return list(all_ids)

    except Exception as e:
        print(f"Error in get_all_filenames(): {e}")
        return []


def filter_document_filenames(filenames, extensions=['.pdf', '.txt']):
    """
    Filter filenames to include only specified document extensions.
    
    Args:
        filenames (list): List of all filenames
        extensions (list): List of file extensions to keep (default: .pdf and .txt)
    
    Returns:
        list: Filtered list of document filenames
    """
    return [
        filename for filename in filenames 
        if any(filename.lower().endswith(ext.lower()) for ext in extensions)
    ]

def enhanced_fuzzy_match_documents(user_query, all_filenames, threshold=70):
    """
    Enhanced fuzzy matching with improved handling for WTD citations.
    
    Args:
        user_query (str): The user's search query
        all_filenames (list): List of all available filenames
        threshold (int): Fuzzy matching threshold (lower = more lenient)
        
    Returns:
        list: List of matched filenames
    """
    # Handle empty or invalid input
    if not user_query or not all_filenames:
        print("⚠️ Empty query or no filenames to match against.")
        return []
    
    # Convert filenames to lowercase for matching
    filenames_lower = [fn.lower() for fn in all_filenames]
    
    # Extract quoted document references
    quoted_docs = re.findall(r'"([^"]+)"', user_query)
    
    # Generate potential matching patterns
    potential_refs = []

        # --- ETA matcher (full form: ETA 3213.2023) ---
    eta_matches = re.findall(r'\bETA[\s\-]*(\d{4}\.\d{4})\b', user_query, re.I)
    for match in eta_matches:
        eta = match.strip()
        potential_refs.extend([
            f"ETA {eta}", f"{eta}", f"ETA{eta}", f"ETA-{eta}"
        ])

    # --- ETA matcher (short form: ETA 3213 or just 3213) ---
    short_eta_matches = re.findall(r'\b(?:ETA\s*)?(\d{4})\b', user_query, re.I)
    for short in short_eta_matches:
        potential_refs.extend([
            f"ETA {short}.2023",
            f"ETA {short}",
            f"{short}.2023",
            f"{short}"
        ])
    
    # --- WTD matcher (spaced and compact) ---
    wtd_matches = re.findall(r'\b(\d{2}\s*WTD\s*\d{3})\b', user_query, re.I)
    for match in wtd_matches:
        clean_wtd = re.sub(r'\s+', '', match)
        potential_refs.extend([
            clean_wtd,
            f"WTD{clean_wtd}",
            f"WTD {match.strip()}",
        ])
        parts = re.findall(r'(\d{2})\s*WTD\s*(\d{3})', match, re.I)
        if parts:
            volume, page = parts[0]
            potential_refs.append(f"{volume}-{page}")

    # --- Compact WTD matcher (like 43WTD069) ---
    compact_wtds = re.findall(r'\b(\d{2}WTD\d{3})\b', user_query, re.I)
    for match in compact_wtds:
        potential_refs.append(match)
        try:
            volume = match[:2]
            page = match[-3:]
            potential_refs.append(f"{volume}-{page}")
            potential_refs.append(f"{volume} WTD {page}")
        except Exception:
            continue
    
    # Handle RCW references 
    rcw_matches = re.findall(r'\b(RCW\s*(\d+\.\d+\.\d+))\b', user_query, re.I)
    for full_match, number in rcw_matches:
        potential_refs.extend([
            f"RCW {number}.pdf",  # Full PDF filename format
            number,               # Just the number
            number + ".pdf",      # Number with PDF extension
            full_match,           # Full RCW reference
        ])
    
    # Handle WAC references
    wac_matches = re.findall(r'\b(WAC\s*(\d+\-\d+\-\d+))\b', user_query, re.I)
    for full_match, number in wac_matches:
        potential_refs.extend([
            f"WAC {number}.pdf",
            number,
            number + ".pdf",
            full_match,
        ])
    
    # Handle other numeric references
    number_matches = re.findall(r'\b(\d+\.\d+\.\d+)\b', user_query)
    for number in number_matches:
        potential_refs.extend([
            f"RCW {number}.pdf",
            number,
            number + ".pdf"
        ])
    
    # Add quoted references
    potential_refs.extend(quoted_docs)
    
    # Remove duplicates and convert to lowercase for matching
    potential_refs = list(set(potential_refs))
    potential_refs_lower = [ref.lower() for ref in potential_refs]
    
    # Print for debugging
    print(f"DEBUG: Potential refs: {potential_refs}")
    
    # Exact filename matching
    exact_matches = []
    for fn in all_filenames:
        fn_lower = fn.lower()
        for ref_lower in potential_refs_lower:
            # Different matching strategies
            if (ref_lower in fn_lower or 
                fn_lower.startswith(ref_lower) or 
                ref_lower in fn_lower.replace('.pdf', '') or
                ref_lower.replace(' ', '') in fn_lower):
                exact_matches.append(fn)
                break
    
    # Remove duplicates
    exact_matches = list(dict.fromkeys(exact_matches))
    print(f"DEBUG: Exact matches: {exact_matches}")
    
    if exact_matches:
        return exact_matches
    
    # Fuzzy matching if no exact matches
    fuzzy_matches = []
    for ref in potential_refs:
        match = process.extractOne(
            ref.lower(), 
            filenames_lower, 
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold
        )
        
        if match and match[1] >= threshold:
            # Find the original filename
            matching_files = [fn for fn, norm_fn in zip(all_filenames, filenames_lower) 
                             if norm_fn == match[0]]
            fuzzy_matches.extend(matching_files)
    
    # Remove duplicates while preserving order
    fuzzy_matches = list(dict.fromkeys(fuzzy_matches))
    print(f"DEBUG: Fuzzy matches: {fuzzy_matches}")
    
    return fuzzy_matches

def check_index_content(index, EMBED_DIM=3072, query="WTD"):
    """
    Check what's in the Pinecone index
    
    Args:
        index: Pinecone index object
        EMBED_DIM (int): Embedding dimension size
        query (str): Term to search for in filenames
        
    Returns:
        set: Set of unique filenames found in the index
    """
    # Simple query to check content
    results = index.query(
        vector=[0.0]*EMBED_DIM,
        top_k=100,
        include_metadata=True
    )
    
    # Check filenames in index
    filenames = set()
    for match in results['matches']:
        if 'filename' in match['metadata']:
            filename = match['metadata']['filename']
            filenames.add(filename)
            if query.lower() in filename.lower():
                print(f"Found matching file: {filename}")
    
    print(f"Total unique filenames in index: {len(filenames)}")
    return filenames

# Example usage
if __name__ == '__main__':
    # Test with a sample query
    all_files = [
        "41 WTD 371.pdf", 
        "RCW 82.87.110.pdf", 
        "WAC 458-20-10001.pdf",
        "38WTD290.pdf"
    ]
    query = "What does \"41 WTD 371\" discuss?"
    matched_documents = enhanced_fuzzy_match_documents(query, all_files)
    print("Matched Documents:", matched_documents)