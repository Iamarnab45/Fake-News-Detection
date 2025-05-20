import re
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import spacy
from textblob import TextBlob

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class NLPFeatures:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all NLP features from text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic text statistics
        features.update(self._get_basic_stats(text))
        
        # Sentiment analysis
        features.update(self._get_sentiment_features(text))
        
        # Readability metrics
        features.update(self._get_readability_metrics(text))
        
        # Named entities
        features.update(self._get_entity_features(text))
        
        # Part-of-speech features
        features.update(self._get_pos_features(text))
        
        return features
    
    def _get_basic_stats(self, text: str) -> Dict[str, float]:
        """Calculate basic text statistics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_no_stop = [w for w in words if w not in self.stop_words]
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'stop_word_ratio': len(words) - len(words_no_stop) / len(words) if words else 0
        }
    
    def _get_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment-related features"""
        blob = TextBlob(text)
        sia_scores = self.sia.polarity_scores(text)
        
        return {
            'subjectivity': blob.sentiment.subjectivity,
            'polarity': blob.sentiment.polarity,
            'compound_sentiment': sia_scores['compound'],
            'positive_sentiment': sia_scores['pos'],
            'negative_sentiment': sia_scores['neg'],
            'neutral_sentiment': sia_scores['neu']
        }
    
    def _get_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Flesch Reading Ease Score
        total_syllables = sum(self._count_syllables(word) for word in words)
        flesch_score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
        
        return {
            'flesch_score': flesch_score,
            'syllable_count': total_syllables,
            'avg_syllables_per_word': total_syllables / len(words) if words else 0
        }
    
    def _get_entity_features(self, text: str) -> Dict[str, float]:
        """Extract named entity features"""
        doc = nlp(text)
        entities = [ent.label_ for ent in doc.ents]
        entity_counts = Counter(entities)
        
        features = {
            'entity_count': len(entities),
            'unique_entity_types': len(set(entities))
        }
        
        # Add counts for common entity types
        for entity_type in ['PERSON', 'ORG', 'GPE', 'DATE']:
            features[f'{entity_type.lower()}_count'] = entity_counts.get(entity_type, 0)
        
        return features
    
    def _get_pos_features(self, text: str) -> Dict[str, float]:
        """Extract part-of-speech features"""
        doc = nlp(text)
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        features = {}
        for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']:
            features[f'{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        word = re.sub(r'[^a-z]', '', word)
        
        if not word:
            return 0
            
        if word[0] in vowels:
            count += 1
            
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
                
        if word.endswith("e"):
            count -= 1
            
        return max(1, count) 