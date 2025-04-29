import re

LEGAL_PATTERNS = {
    "cascading hierarchy": {
        "regex": r"(unable to attribute|fallback attribution|no sufficient documentation|could not identify|failed to provide|taxpayer could not determine)",
        "weight": 6
    },
    "reasonable proportional method": {
        "regex": r"(reasonable method|percentage of use|based on usage data|proportional attribution|apportioned receipts|distributed according to)",
        "weight": 6
    },
    "market-based sourcing": {
        "regex": r"(customer'?s market|benefit received at location|where benefit is received|market for customer|service location is customer-based)",
        "weight": 5
    },
    "business location sourcing": {
        "regex": r"(customer'?s business location|customer headquarters|books and records kept|back-office location|main office|primary place of business)",
        "weight": 5
    },
    "intermediate use rejection": {
        "regex": r"(intermediate use not sufficient|department rejected|does not qualify as primary use|not final use location)",
        "weight": 4
    },
    "multiple points of use": {
        "regex": r"(multiple equipment locations|used in multiple states|MPU certificate|downloaded across states|multi-state use|remote software installation)",
        "weight": 4
    }
}

def match_legal_patterns(text: str) -> list:
    """
    Returns a list of matched legal pattern names and weights found in the text.
    Format: [(label, weight), ...]
    """
    matches = []
    for label, pattern in LEGAL_PATTERNS.items():
        if re.search(pattern["regex"], text, re.IGNORECASE):
            matches.append((label, pattern["weight"]))
    return matches


