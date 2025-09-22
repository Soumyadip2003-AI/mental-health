"""
Advanced feature extraction utilities
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Advanced feature extraction for text analysis"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.word2vec_model = None
        self.fasttext_model = None
        self.doc2vec_model = None
        self.lda_model = None
        self.svd_model = None
        self.is_fitted = False
    
    def extract_features(self, text: str, 
                        use_tfidf: bool = True,
                        use_count: bool = True,
                        use_word2vec: bool = False,
                        use_fasttext: bool = False,
                        use_doc2vec: bool = False,
                        use_lda: bool = False,
                        use_svd: bool = False,
                        use_linguistic: bool = True,
                        use_sentiment: bool = True,
                        use_readability: bool = True,
                        use_crisis_indicators: bool = True) -> Dict[str, Any]:
        """
        Extract comprehensive features from text
        
        Args:
            text: Input text
            use_tfidf: Whether to extract TF-IDF features
            use_count: Whether to extract count features
            use_word2vec: Whether to extract Word2Vec features
            use_fasttext: Whether to extract FastText features
            use_doc2vec: Whether to extract Doc2Vec features
            use_lda: Whether to extract LDA topic features
            use_svd: Whether to extract SVD features
            use_linguistic: Whether to extract linguistic features
            use_sentiment: Whether to extract sentiment features
            use_readability: Whether to extract readability features
            use_crisis_indicators: Whether to extract crisis indicators
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        try:
            # TF-IDF features
            if use_tfidf and self.tfidf_vectorizer:
                tfidf_features = self._extract_tfidf_features(text)
                features.update(tfidf_features)
            
            # Count features
            if use_count and self.count_vectorizer:
                count_features = self._extract_count_features(text)
                features.update(count_features)
            
            # Word2Vec features
            if use_word2vec and self.word2vec_model:
                word2vec_features = self._extract_word2vec_features(text)
                features.update(word2vec_features)
            
            # FastText features
            if use_fasttext and self.fasttext_model:
                fasttext_features = self._extract_fasttext_features(text)
                features.update(fasttext_features)
            
            # Doc2Vec features
            if use_doc2vec and self.doc2vec_model:
                doc2vec_features = self._extract_doc2vec_features(text)
                features.update(doc2vec_features)
            
            # LDA topic features
            if use_lda and self.lda_model:
                lda_features = self._extract_lda_features(text)
                features.update(lda_features)
            
            # SVD features
            if use_svd and self.svd_model:
                svd_features = self._extract_svd_features(text)
                features.update(svd_features)
            
            # Linguistic features
            if use_linguistic:
                linguistic_features = self._extract_linguistic_features(text)
                features.update(linguistic_features)
            
            # Sentiment features
            if use_sentiment:
                sentiment_features = self._extract_sentiment_features(text)
                features.update(sentiment_features)
            
            # Readability features
            if use_readability:
                readability_features = self._extract_readability_features(text)
                features.update(readability_features)
            
            # Crisis indicators
            if use_crisis_indicators:
                crisis_features = self._extract_crisis_indicators(text)
                features.update(crisis_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _extract_tfidf_features(self, text: str) -> Dict[str, float]:
        """Extract TF-IDF features"""
        try:
            if not self.tfidf_vectorizer:
                return {}
            
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            tfidf_array = tfidf_matrix.toarray()[0]
            
            features = {}
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Top TF-IDF features
            top_indices = np.argsort(tfidf_array)[-10:]  # Top 10 features
            for i, idx in enumerate(top_indices):
                if tfidf_array[idx] > 0:
                    features[f'tfidf_top_{i+1}'] = tfidf_array[idx]
                    features[f'tfidf_feature_{i+1}'] = feature_names[idx]
            
            # TF-IDF statistics
            features['tfidf_max'] = np.max(tfidf_array)
            features['tfidf_mean'] = np.mean(tfidf_array)
            features['tfidf_std'] = np.std(tfidf_array)
            features['tfidf_nonzero'] = np.count_nonzero(tfidf_array)
            
            return features
            
        except Exception as e:
            logger.warning(f"TF-IDF feature extraction failed: {e}")
            return {}
    
    def _extract_count_features(self, text: str) -> Dict[str, float]:
        """Extract count-based features"""
        try:
            if not self.count_vectorizer:
                return {}
            
            count_matrix = self.count_vectorizer.transform([text])
            count_array = count_matrix.toarray()[0]
            
            features = {}
            feature_names = self.count_vectorizer.get_feature_names_out()
            
            # Top count features
            top_indices = np.argsort(count_array)[-10:]  # Top 10 features
            for i, idx in enumerate(top_indices):
                if count_array[idx] > 0:
                    features[f'count_top_{i+1}'] = count_array[idx]
                    features[f'count_feature_{i+1}'] = feature_names[idx]
            
            # Count statistics
            features['count_max'] = np.max(count_array)
            features['count_mean'] = np.mean(count_array)
            features['count_std'] = np.std(count_array)
            features['count_nonzero'] = np.count_nonzero(count_array)
            
            return features
            
        except Exception as e:
            logger.warning(f"Count feature extraction failed: {e}")
            return {}
    
    def _extract_word2vec_features(self, text: str) -> Dict[str, float]:
        """Extract Word2Vec features"""
        try:
            if not self.word2vec_model:
                return {}
            
            words = text.lower().split()
            word_vectors = []
            
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if not word_vectors:
                return {}
            
            word_vectors = np.array(word_vectors)
            
            features = {}
            
            # Document vector (mean of word vectors)
            doc_vector = np.mean(word_vectors, axis=0)
            for i, val in enumerate(doc_vector):
                features[f'word2vec_dim_{i}'] = val
            
            # Statistics
            features['word2vec_mean'] = np.mean(doc_vector)
            features['word2vec_std'] = np.std(doc_vector)
            features['word2vec_max'] = np.max(doc_vector)
            features['word2vec_min'] = np.min(doc_vector)
            
            return features
            
        except Exception as e:
            logger.warning(f"Word2Vec feature extraction failed: {e}")
            return {}
    
    def _extract_fasttext_features(self, text: str) -> Dict[str, float]:
        """Extract FastText features"""
        try:
            if not self.fasttext_model:
                return {}
            
            words = text.lower().split()
            word_vectors = []
            
            for word in words:
                word_vector = self.fasttext_model.wv[word]
                word_vectors.append(word_vector)
            
            if not word_vectors:
                return {}
            
            word_vectors = np.array(word_vectors)
            
            features = {}
            
            # Document vector (mean of word vectors)
            doc_vector = np.mean(word_vectors, axis=0)
            for i, val in enumerate(doc_vector):
                features[f'fasttext_dim_{i}'] = val
            
            # Statistics
            features['fasttext_mean'] = np.mean(doc_vector)
            features['fasttext_std'] = np.std(doc_vector)
            features['fasttext_max'] = np.max(doc_vector)
            features['fasttext_min'] = np.min(doc_vector)
            
            return features
            
        except Exception as e:
            logger.warning(f"FastText feature extraction failed: {e}")
            return {}
    
    def _extract_doc2vec_features(self, text: str) -> Dict[str, float]:
        """Extract Doc2Vec features"""
        try:
            if not self.doc2vec_model:
                return {}
            
            # Create tagged document
            words = text.lower().split()
            tagged_doc = TaggedDocument(words, [0])
            
            # Infer document vector
            doc_vector = self.doc2vec_model.infer_vector(words)
            
            features = {}
            for i, val in enumerate(doc_vector):
                features[f'doc2vec_dim_{i}'] = val
            
            # Statistics
            features['doc2vec_mean'] = np.mean(doc_vector)
            features['doc2vec_std'] = np.std(doc_vector)
            features['doc2vec_max'] = np.max(doc_vector)
            features['doc2vec_min'] = np.min(doc_vector)
            
            return features
            
        except Exception as e:
            logger.warning(f"Doc2Vec feature extraction failed: {e}")
            return {}
    
    def _extract_lda_features(self, text: str) -> Dict[str, float]:
        """Extract LDA topic features"""
        try:
            if not self.lda_model:
                return {}
            
            # Transform text using the same vectorizer used for LDA training
            # This would need to be stored during model training
            # For now, return empty features
            return {}
            
        except Exception as e:
            logger.warning(f"LDA feature extraction failed: {e}")
            return {}
    
    def _extract_svd_features(self, text: str) -> Dict[str, float]:
        """Extract SVD features"""
        try:
            if not self.svd_model:
                return {}
            
            # Transform text using the same vectorizer used for SVD training
            # This would need to be stored during model training
            # For now, return empty features
            return {}
            
        except Exception as e:
            logger.warning(f"SVD feature extraction failed: {e}")
            return {}
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        features['avg_sentence_length'] = len(text.split()) / max(1, len(re.split(r'[.!?]+', text)))
        
        # Character statistics
        features['char_count'] = len(text)
        features['alpha_count'] = sum(1 for c in text if c.isalpha())
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        features['space_count'] = sum(1 for c in text if c.isspace())
        features['punct_count'] = sum(1 for c in text if c in '.,!?;:')
        
        # Case statistics
        features['upper_count'] = sum(1 for c in text if c.isupper())
        features['lower_count'] = sum(1 for c in text if c.islower())
        features['title_count'] = sum(1 for c in text if c.istitle())
        
        # Special characters
        features['has_question'] = '?' in text
        features['has_exclamation'] = '!' in text
        features['has_ellipsis'] = '...' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        # Word patterns
        words = text.lower().split()
        features['unique_words'] = len(set(words))
        features['repeated_words'] = len(words) - len(set(words))
        features['word_diversity'] = len(set(words)) / max(1, len(words))
        
        # Repeated characters
        features['repeated_chars'] = sum(1 for i in range(len(text)-1) if text[i] == text[i+1])
        
        return features
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features"""
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            features = {
                'sentiment_polarity': sentiment.polarity,
                'sentiment_subjectivity': sentiment.subjectivity,
                'sentiment_positive': 1 if sentiment.polarity > 0.1 else 0,
                'sentiment_negative': 1 if sentiment.polarity < -0.1 else 0,
                'sentiment_neutral': 1 if -0.1 <= sentiment.polarity <= 0.1 else 0
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Sentiment feature extraction failed: {e}")
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'sentiment_positive': 0,
                'sentiment_negative': 0,
                'sentiment_neutral': 1
            }
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability features"""
        try:
            from textstat import flesch_reading_ease, flesch_kincaid_grade, smog_index
            
            features = {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text),
                'smog_index': smog_index(text)
            }
            
            return features
            
        except ImportError:
            logger.warning("textstat not available, using basic readability features")
            return self._basic_readability_features(text)
        except Exception as e:
            logger.warning(f"Readability feature extraction failed: {e}")
            return self._basic_readability_features(text)
    
    def _basic_readability_features(self, text: str) -> Dict[str, float]:
        """Basic readability features without textstat"""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return {'avg_sentence_length': 0, 'avg_word_length': 0}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }
    
    def _extract_crisis_indicators(self, text: str) -> Dict[str, Any]:
        """Extract crisis indicators"""
        crisis_keywords = {
            'suicide': ['suicide', 'kill myself', 'end my life', 'take my life', 'die', 'death'],
            'hopelessness': ['hopeless', 'worthless', 'useless', 'burden', 'can\'t go on', 'no reason', 'pointless'],
            'planning': ['plan', 'method', 'pills', 'rope', 'gun', 'jump', 'bridge', 'building', 'cut', 'wrist'],
            'goodbye': ['goodbye', 'last message', 'final', 'farewell', 'see you never', 'this is it']
        }
        
        features = {}
        text_lower = text.lower()
        
        for category, keywords in crisis_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'crisis_{category}_count'] = count
            features[f'crisis_{category}_present'] = 1 if count > 0 else 0
        
        # Overall crisis score
        total_crisis_indicators = sum(features[f'crisis_{category}_count'] for category in crisis_keywords.keys())
        features['crisis_total_count'] = total_crisis_indicators
        features['crisis_present'] = 1 if total_crisis_indicators > 0 else 0
        
        return features
    
    def fit_vectorizers(self, texts: List[str], 
                        tfidf_max_features: int = 5000,
                        count_max_features: int = 5000,
                        ngram_range: Tuple[int, int] = (1, 2)):
        """Fit vectorizers on training data"""
        try:
            # TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            self.tfidf_vectorizer.fit(texts)
            
            # Count vectorizer
            self.count_vectorizer = CountVectorizer(
                max_features=count_max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            self.count_vectorizer.fit(texts)
            
            self.is_fitted = True
            logger.info("Vectorizers fitted successfully")
            
        except Exception as e:
            logger.error(f"Vectorizer fitting failed: {e}")
            raise
    
    def fit_embedding_models(self, texts: List[str], 
                            word2vec_size: int = 100,
                            word2vec_window: int = 5,
                            word2vec_min_count: int = 2,
                            fasttext_size: int = 100,
                            fasttext_window: int = 5,
                            fasttext_min_count: int = 2,
                            doc2vec_size: int = 100,
                            doc2vec_window: int = 5,
                            doc2vec_min_count: int = 2):
        """Fit embedding models on training data"""
        try:
            # Prepare data
            tokenized_texts = [text.lower().split() for text in texts]
            tagged_docs = [TaggedDocument(words, [i]) for i, words in enumerate(tokenized_texts)]
            
            # Word2Vec
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=word2vec_size,
                window=word2vec_window,
                min_count=word2vec_min_count,
                workers=4
            )
            
            # FastText
            self.fasttext_model = FastText(
                sentences=tokenized_texts,
                vector_size=fasttext_size,
                window=fasttext_window,
                min_count=fasttext_min_count,
                workers=4
            )
            
            # Doc2Vec
            self.doc2vec_model = Doc2Vec(
                documents=tagged_docs,
                vector_size=doc2vec_size,
                window=doc2vec_window,
                min_count=doc2vec_min_count,
                workers=4
            )
            
            logger.info("Embedding models fitted successfully")
            
        except Exception as e:
            logger.error(f"Embedding model fitting failed: {e}")
            raise
    
    def fit_topic_models(self, texts: List[str], 
                        lda_topics: int = 10,
                        svd_components: int = 50):
        """Fit topic models on training data"""
        try:
            # Prepare data using TF-IDF
            if not self.tfidf_vectorizer:
                self.fit_vectorizers(texts)
            
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            # LDA
            self.lda_model = LatentDirichletAllocation(
                n_components=lda_topics,
                random_state=42
            )
            self.lda_model.fit(tfidf_matrix)
            
            # SVD
            self.svd_model = TruncatedSVD(
                n_components=svd_components,
                random_state=42
            )
            self.svd_model.fit(tfidf_matrix)
            
            logger.info("Topic models fitted successfully")
            
        except Exception as e:
            logger.error(f"Topic model fitting failed: {e}")
            raise
