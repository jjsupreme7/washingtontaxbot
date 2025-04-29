# src/keyword_extraction.py
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_keywords(text: str, top_k: int = 5):
    # Tokenize and clean
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [t for t in tokens if re.match(r'^\w+$', t) and t not in stop_words]

    # Run TF-IDF over this single text for ranking
    if not filtered_tokens:
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_tokens)])
    feature_names = vectorizer.get_feature_names_out()

    # Sort by score and return top k
    tfidf_scores = tfidf_matrix.toarray()[0]
    ranked_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in ranked_keywords[:top_k]]

