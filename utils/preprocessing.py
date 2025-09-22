"""
Advanced text preprocessing utilities
"""
import re
import string
import logging
from typing import List, Dict, Any, Optional, Tuple
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from textblob import TextBlob
import emoji
import unicodedata

logger = logging.getLogger(__name__)


class AdvancedTextPreprocessor:
    """Advanced text preprocessing with multiple techniques"""
    
    def __init__(self, use_spacy: bool = True, use_nltk: bool = True):
        """
        Initialize the preprocessor
        
        Args:
            use_spacy: Whether to use spaCy for advanced processing
            use_nltk: Whether to use NLTK for basic processing
        """
        self.use_spacy = use_spacy
        self.use_nltk = use_nltk
        
        # Initialize NLTK components
        if self.use_nltk:
            self._download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
        
        # Initialize spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, falling back to NLTK")
                self.use_spacy = False
                self.nlp = None
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\b\d+\b')
        
        # Crisis-specific patterns
        self.crisis_patterns = {
            'suicide': re.compile(r'\b(suicide|kill\s+myself|end\s+my\s+life|take\s+my\s+life|die|death)\b', re.IGNORECASE),
            'hopelessness': re.compile(r'\b(hopeless|worthless|useless|burden|can\'t\s+go\s+on|no\s+reason|pointless)\b', re.IGNORECASE),
            'planning': re.compile(r'\b(plan|method|pills|rope|gun|jump|bridge|building|cut|wrist)\b', re.IGNORECASE),
            'goodbye': re.compile(r'\b(goodbye|last\s+message|final|farewell|see\s+you\s+never|this\s+is\s+it)\b', re.IGNORECASE)
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def preprocess(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phones: bool = True,
                   remove_mentions: bool = True,
                   remove_hashtags: bool = False,
                   remove_numbers: bool = False,
                   normalize_unicode: bool = True,
                   remove_extra_whitespace: bool = True,
                   convert_to_lowercase: bool = True,
                   remove_punctuation: bool = False,
                   remove_stopwords: bool = False,
                   lemmatize: bool = False,
                   stem: bool = False,
                   extract_entities: bool = False,
                   extract_sentiment: bool = False,
                   extract_crisis_indicators: bool = True) -> Dict[str, Any]:
        """
        Advanced text preprocessing with multiple options
        
        Args:
            text: Input text to preprocess
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_phones: Whether to remove phone numbers
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            remove_numbers: Whether to remove numbers
            normalize_unicode: Whether to normalize unicode characters
            remove_extra_whitespace: Whether to remove extra whitespace
            convert_to_lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            stem: Whether to stem words
            extract_entities: Whether to extract named entities
            extract_sentiment: Whether to extract sentiment
            extract_crisis_indicators: Whether to extract crisis indicators
            
        Returns:
            Dictionary containing preprocessed text and extracted features
        """
        if not text or not isinstance(text, str):
            return {
                'processed_text': '',
                'original_text': text,
                'features': {},
                'metadata': {}
            }
        
        original_text = text
        processed_text = text
        
        # Store metadata
        metadata = {
            'original_length': len(text),
            'original_word_count': len(text.split()),
            'preprocessing_steps': []
        }
        
        # URL removal
        if remove_urls:
            processed_text = self.url_pattern.sub('[URL]', processed_text)
            metadata['preprocessing_steps'].append('remove_urls')
        
        # Email removal
        if remove_emails:
            processed_text = self.email_pattern.sub('[EMAIL]', processed_text)
            metadata['preprocessing_steps'].append('remove_emails')
        
        # Phone number removal
        if remove_phones:
            processed_text = self.phone_pattern.sub('[PHONE]', processed_text)
            metadata['preprocessing_steps'].append('remove_phones')
        
        # Mention removal
        if remove_mentions:
            processed_text = self.mention_pattern.sub('[MENTION]', processed_text)
            metadata['preprocessing_steps'].append('remove_mentions')
        
        # Hashtag removal
        if remove_hashtags:
            processed_text = self.hashtag_pattern.sub('[HASHTAG]', processed_text)
            metadata['preprocessing_steps'].append('remove_hashtags')
        
        # Number removal
        if remove_numbers:
            processed_text = self.number_pattern.sub('[NUMBER]', processed_text)
            metadata['preprocessing_steps'].append('remove_numbers')
        
        # Unicode normalization
        if normalize_unicode:
            processed_text = unicodedata.normalize('NFKD', processed_text)
            metadata['preprocessing_steps'].append('normalize_unicode')
        
        # Extra whitespace removal
        if remove_extra_whitespace:
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            metadata['preprocessing_steps'].append('remove_extra_whitespace')
        
        # Case conversion
        if convert_to_lowercase:
            processed_text = processed_text.lower()
            metadata['preprocessing_steps'].append('convert_to_lowercase')
        
        # Extract features before further processing
        features = {}
        
        if extract_sentiment:
            features['sentiment'] = self._extract_sentiment(processed_text)
        
        if extract_entities and self.use_spacy and self.nlp:
            features['entities'] = self._extract_entities(processed_text)
        
        if extract_crisis_indicators:
            features['crisis_indicators'] = self._extract_crisis_indicators(processed_text)
        
        # Tokenization and further processing
        if self.use_spacy and self.nlp and not (lemmatize or stem or remove_stopwords):
            # Use spaCy for tokenization
            doc = self.nlp(processed_text)
            tokens = [token.text for token in doc]
        else:
            # Use NLTK for tokenization
            tokens = word_tokenize(processed_text)
        
        # Punctuation removal
        if remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
            metadata['preprocessing_steps'].append('remove_punctuation')
        
        # Stopword removal
        if remove_stopwords and self.use_nltk:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            metadata['preprocessing_steps'].append('remove_stopwords')
        
        # Lemmatization
        if lemmatize and self.use_nltk:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            metadata['preprocessing_steps'].append('lemmatize')
        
        # Stemming
        if stem and self.use_nltk:
            tokens = [self.stemmer.stem(token) for token in tokens]
            metadata['preprocessing_steps'].append('stem')
        
        # Join tokens back to text
        processed_text = ' '.join(tokens)
        
        # Update metadata
        metadata['processed_length'] = len(processed_text)
        metadata['processed_word_count'] = len(processed_text.split())
        metadata['processing_ratio'] = metadata['processed_length'] / metadata['original_length'] if metadata['original_length'] > 0 else 0
        
        return {
            'processed_text': processed_text,
            'original_text': original_text,
            'features': features,
            'metadata': metadata
        }
    
    def _extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment features from text"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            return {
                'polarity': sentiment.polarity,  # -1 to 1
                'subjectivity': sentiment.subjectivity,  # 0 to 1
                'sentiment_label': 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral'
            }
        except Exception as e:
            logger.warning(f"Sentiment extraction failed: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment_label': 'neutral'}
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': ent._.prob if hasattr(ent._, 'prob') else 1.0
                })
            
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_crisis_indicators(self, text: str) -> Dict[str, Any]:
        """Extract crisis indicators from text"""
        indicators = {
            'suicide_indicators': [],
            'hopelessness_indicators': [],
            'planning_indicators': [],
            'goodbye_indicators': [],
            'crisis_score': 0.0
        }
        
        for category, pattern in self.crisis_patterns.items():
            matches = pattern.findall(text)
            if matches:
                indicators[f'{category}_indicators'] = matches
                indicators['crisis_score'] += len(matches) * 0.2  # Weight each match
        
        # Normalize crisis score
        indicators['crisis_score'] = min(1.0, indicators['crisis_score'])
        
        return indicators
    
    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)) if self.use_nltk else text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
            'avg_sentence_length': len(text.split()) / max(1, len(sent_tokenize(text))) if self.use_nltk else len(text.split()) / max(1, text.count('.') + text.count('!') + text.count('?')),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'has_caps': any(c.isupper() for c in text if c.isalpha()),
            'all_caps_ratio': sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha())),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(1, len(text)),
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / max(1, len(text)),
            'emoji_count': emoji.emoji_count(text),
            'repeated_chars': self._count_repeated_chars(text),
            'repeated_words': self._count_repeated_words(text)
        }
        
        return features
    
    def _count_repeated_chars(self, text: str) -> int:
        """Count repeated characters in text"""
        count = 0
        for i in range(len(text) - 1):
            if text[i] == text[i + 1] and text[i].isalpha():
                count += 1
        return count
    
    def _count_repeated_words(self, text: str) -> int:
        """Count repeated words in text"""
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_count = sum(1 for count in word_counts.values() if count > 1)
        return repeated_count
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability features"""
        try:
            blob = TextBlob(text)
            
            # Basic readability metrics
            sentences = blob.sentences
            words = blob.words
            
            if not sentences or not words:
                return {'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0}
            
            # Flesch Reading Ease
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Flesch-Kincaid Grade Level
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            
            return {
                'flesch_reading_ease': max(0, min(100, flesch_score)),
                'flesch_kincaid_grade': max(0, fk_grade),
                'avg_sentence_length': avg_sentence_length,
                'avg_syllables_per_word': avg_syllables_per_word
            }
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return {'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0}
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Preprocess multiple texts in batch"""
        results = []
        
        for text in texts:
            try:
                result = self.preprocess(text, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch preprocessing failed for text: {e}")
                results.append({
                    'processed_text': '',
                    'original_text': text,
                    'features': {},
                    'metadata': {'error': str(e)}
                })
        
        return results
