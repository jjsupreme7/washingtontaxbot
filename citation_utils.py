import re

def enhanced_legal_citation_extraction(text):
    """
    Extract detailed legal citations with comprehensive subsection parsing.
    
    Handles various citation formats including:
    - RCW and WAC sections
    - Washington Tax Determinations
    - Specific case determinations
    - Specific legal references
    """
    # Comprehensive RCW citation patterns
    rcw_patterns = [
        r'RCW\s+(\d+\.\d+(?:\.\d+)?(?:\(\d+\))?(?:\([a-zA-Z]\))?(?:\(\d+\))?)',
        r'RCW\s+(\d+\.\d+(?:\.\d+)?(?:\(\d+\)[a-zA-Z]?(?:\(\d+\))?)?)'
    ]
    
    # Enhanced WAC citation patterns
    wac_patterns = [
        r'WAC\s+(\d+-\d+-\d+(?:\(\d+\))?(?:\([a-zA-Z]\))?(?:\(\d+\))?)',
        r'WAC\s+(\d+-\d+(?:\(\d+\))?)'
    ]
    
    # Expanded citation patterns
    citations = {
        'rcw': [],
        'wac': [],
        'wtd': [],
        'cases': [],
        'specific_refs': []
    }
    
    # Ensure text is a string and not None
    if not isinstance(text, str):
        return citations
    
    # Extract RCW citations
    for pattern in rcw_patterns:
        citations['rcw'].extend(re.findall(pattern, text, re.I))
    
    # Extract WAC citations
    for pattern in wac_patterns:
        citations['wac'].extend(re.findall(pattern, text, re.I))
    
    # Extract WTD citations
    wtd_patterns = [
        r'(\d+\s*WTD\s*\d+)',  # Standard WTD format
        r'Determination\s+No\.\s+(\d+-\d+)'  # Determination number format
    ]
    for pattern in wtd_patterns:
        citations['wtd'].extend(re.findall(pattern, text))
    
    # Extract specific case references
    case_patterns = [
        r'(Tele-Vue\s+Systems,\s+No\.\s+\d+-\d+)',  # Specific case citation
        r'(Board of Tax Appeals)',  # Specific legal body
        r'(No\.\s+\d+-\d+)'  # Generic case number format
    ]
    for pattern in case_patterns:
        citations['cases'].extend(re.findall(pattern, text))
    
    # Extract specific legal references
    specific_ref_patterns = [
        r'(RCW\s+\d+\.\d+\.\d+)',  # Full RCW references
        r'(WAC\s+\d+-\d+-\d+)',  # Full WAC references
        r'(Determination\s+No\.\s+\d+-\d+)'  # Determination references
    ]
    for pattern in specific_ref_patterns:
        citations['specific_refs'].extend(re.findall(pattern, text))
    
    # Remove duplicates while preserving order
    for key in citations:
        citations[key] = list(dict.fromkeys(citations[key]))
    
    return citations

def extract_citations_from_chunks(chunks):
    """
    Extract citations across multiple chunks
    
    Args:
        chunks (list): List of document chunks from Pinecone
    
    Returns:
        dict: Comprehensive citations across all chunks
    """
    all_citations = {
        'rcw': set(),
        'wac': set(),
        'wtd': set(),
        'cases': set(),
        'specific_refs': set()
    }
    
    for chunk in chunks:
        # Safely extract text, defaulting to empty string
        text = chunk.get('metadata', {}).get('text', '')
        
        # Extract citations from this chunk
        try:
            chunk_citations = enhanced_legal_citation_extraction(text)
            
            # Aggregate citations
            for key in all_citations:
                all_citations[key].update(chunk_citations[key])
        except Exception as e:
            print(f"Error extracting citations from chunk: {e}")
    
    # Convert back to sorted lists
    return {
        key: sorted(list(all_citations[key]))
        for key in all_citations
    }

def format_citations_summary(citations):
    """
    Create a formatted summary of citations
    
    Args:
        citations (dict): Citation dictionary
    
    Returns:
        str: Formatted citation summary
    """
    summary_parts = []
    
    # Citation types with more detailed formatting
    citation_types = [
        ('rcw', "**RCW References:**"),
        ('wac', "**WAC References:**"),
        ('wtd', "**Washington Tax Determinations:**"),
        ('cases', "**Case References:**"),
        ('specific_refs', "**Specific Legal References:**")
    ]
    
    for key, title in citation_types:
        if citations.get(key):
            summary_parts.append(f"{title}\n" + 
                                 "\n".join(f"- {ref}" for ref in citations[key]))
    
    return "\n\n".join(summary_parts)