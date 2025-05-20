import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and extra whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word not in stop_words])
    return filtered_text

def lemmatize_text(text: str) -> str:
    """
    Lemmatize text to reduce words to their base form.
    
    Args:
        text: Input text string
        
    Returns:
        Lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])
    return lemmatized_text

def preprocess_text(text: str) -> str:
    """
    Apply all preprocessing steps to text.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def extract_features(text: str) -> List[str]:
    """
    Extract key features from text.
    
    Args:
        text: Input text string
        
    Returns:
        List of extracted features
    """
    # Basic features
    features = []
    
    # Text length
    features.append(f"length_{len(text)}")
    
    # Word count
    word_count = len(text.split())
    features.append(f"word_count_{word_count}")
    
    # Average word length
    avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
    features.append(f"avg_word_length_{avg_word_length:.1f}")
    
    return features 