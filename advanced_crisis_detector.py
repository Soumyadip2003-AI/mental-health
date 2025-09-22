#!/usr/bin/env python3
"""
Advanced Crisis Detection System with Enhanced Dataset and Models
This system creates a comprehensive dataset with thousands of examples and implements
advanced machine learning models for better crisis detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import os
import re
import logging
from collections import Counter
from tqdm import tqdm
import warnings
import random
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception as e:
    logger.warning(f"NLTK setup issue: {e}")

warnings.filterwarnings('ignore')

class AdvancedCrisisDatasetGenerator:
    """
    Advanced dataset generator for crisis detection with thousands of examples
    """
    
    def __init__(self):
        self.crisis_patterns = self._initialize_crisis_patterns()
        self.non_crisis_patterns = self._initialize_non_crisis_patterns()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def _initialize_crisis_patterns(self):
        """Initialize comprehensive crisis patterns"""
        return {
            'direct_suicidal': [
                "I want to kill myself",
                "I'm going to end my life",
                "I want to die",
                "I'm going to commit suicide",
                "I want to take my own life",
                "I'm planning to end it all",
                "I want to end my suffering",
                "I'm going to kill myself tonight",
                "I want to die right now",
                "I'm going to end my pain forever"
            ],
            'suicidal_planning': [
                "I've been researching suicide methods",
                "I have a plan to end my life",
                "I know how I'm going to do it",
                "I've chosen my method",
                "I have everything I need",
                "I've written my suicide note",
                "I've made my final arrangements",
                "I've decided when to do it",
                "I've been planning this for weeks",
                "I have the means to end it"
            ],
            'hopelessness': [
                "I have no reason to live",
                "Life is meaningless",
                "I'm a burden to everyone",
                "No one would miss me",
                "I'm worthless and useless",
                "I can't see any future",
                "There's no point in trying",
                "I'm better off dead",
                "I don't deserve to live",
                "I'm a failure at everything"
            ],
            'goodbye_messages': [
                "This is my final goodbye",
                "I just wanted to say goodbye",
                "Thank you for everything, goodbye",
                "This will be my last message",
                "I'm saying goodbye forever",
                "By the time you read this, I'll be gone",
                "This is my final message",
                "I'm leaving this world tonight",
                "Goodbye, I love you all",
                "This is the end for me"
            ],
            'emotional_pain': [
                "The pain is unbearable",
                "I can't take this pain anymore",
                "I'm drowning in emotional pain",
                "The suffering is too much",
                "I'm in constant agony",
                "The emotional pain is killing me",
                "I can't escape this pain",
                "I'm drowning in despair",
                "The hurt is overwhelming",
                "I'm being crushed by pain"
            ],
            'isolation': [
                "I'm completely alone",
                "No one understands me",
                "I have no one to talk to",
                "I'm isolated from everyone",
                "I feel completely alone",
                "No one cares about me",
                "I'm invisible to everyone",
                "I have no support system",
                "I'm alone in this world",
                "No one would notice if I was gone"
            ]
        }
    
    def _initialize_non_crisis_patterns(self):
        """Initialize comprehensive non-crisis patterns"""
        return {
            'seeking_help': [
                "I'm going to therapy to work on my issues",
                "I've been talking to my therapist about my feelings",
                "I'm taking medication for my depression",
                "I'm working with a counselor to get better",
                "I've started seeing a mental health professional",
                "I'm getting help for my mental health",
                "I'm in treatment for my depression",
                "I'm working on my mental health with a professional",
                "I'm getting the help I need",
                "I'm committed to my recovery"
            ],
            'coping_strategies': [
                "I'm using my coping skills to get through this",
                "I'm practicing mindfulness to manage my thoughts",
                "I'm using breathing exercises to calm down",
                "I'm journaling to process my emotions",
                "I'm exercising to help with my mood",
                "I'm using positive self-talk",
                "I'm reaching out to my support system",
                "I'm taking it one day at a time",
                "I'm using the tools I've learned in therapy",
                "I'm focusing on self-care"
            ],
            'support_systems': [
                "My friends have been really supportive",
                "I have a great support system",
                "My family is there for me",
                "I'm not alone in this struggle",
                "I have people who care about me",
                "My loved ones are helping me through this",
                "I'm grateful for my support network",
                "I have people I can talk to",
                "I'm surrounded by caring people",
                "I have a strong support system"
            ],
            'hope_and_recovery': [
                "I know this feeling will pass",
                "I'm stronger than my depression",
                "I've overcome challenges before",
                "I believe I can get through this",
                "I have hope for the future",
                "I'm working towards recovery",
                "I'm making progress in my healing",
                "I'm committed to getting better",
                "I know I can overcome this",
                "I'm taking steps towards wellness"
            ],
            'bad_days_with_perspective': [
                "Today was really rough, but tomorrow is another day",
                "I'm struggling right now, but I know this will pass",
                "I'm having a hard time, but I'm hanging in there",
                "This week has been awful, but I'm taking it one day at a time",
                "I feel overwhelmed, but I'm not giving up",
                "I'm having a bad day, but I know it's temporary",
                "I'm struggling, but I'm using my coping skills",
                "I'm having a tough time, but I'm staying strong",
                "I'm feeling down, but I know I'll get through this",
                "I'm having a rough patch, but I'm not giving up"
            ],
            'mental_health_awareness': [
                "I'm learning about my mental health",
                "I'm educating myself about depression",
                "I'm understanding my triggers better",
                "I'm becoming more self-aware",
                "I'm learning to recognize my warning signs",
                "I'm developing better coping strategies",
                "I'm working on my emotional intelligence",
                "I'm learning to manage my mental health",
                "I'm becoming more resilient",
                "I'm growing through this experience"
            ]
        }
    
    def generate_enhanced_dataset(self, n_samples=10000, crisis_ratio=0.3):
        """
        Generate a comprehensive dataset with thousands of examples
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples to generate
        crisis_ratio : float
            Ratio of crisis samples (0.0 to 1.0)
        """
        logger.info(f"Generating enhanced dataset with {n_samples} samples...")
        
        n_crisis = int(n_samples * crisis_ratio)
        n_non_crisis = n_samples - n_crisis
        
        # Generate crisis samples
        crisis_samples = []
        for i in tqdm(range(n_crisis), desc="Generating crisis samples"):
            sample = self._generate_crisis_sample()
            crisis_samples.append(sample)
        
        # Generate non-crisis samples
        non_crisis_samples = []
        for i in tqdm(range(n_non_crisis), desc="Generating non-crisis samples"):
            sample = self._generate_non_crisis_sample()
            non_crisis_samples.append(sample)
        
        # Combine and shuffle
        all_samples = crisis_samples + non_crisis_samples
        all_labels = [1] * n_crisis + [0] * n_non_crisis
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': all_samples,
            'label': all_labels
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated dataset with {len(df)} samples")
        logger.info(f"Crisis samples: {df['label'].sum()}")
        logger.info(f"Non-crisis samples: {len(df) - df['label'].sum()}")
        
        return df
    
    def _generate_crisis_sample(self):
        """Generate a single crisis sample with variations"""
        pattern_type = random.choice(list(self.crisis_patterns.keys()))
        base_text = random.choice(self.crisis_patterns[pattern_type])
        
        # Add variations
        variations = [
            self._add_emotional_intensity,
            self._add_temporal_markers,
            self._add_personal_context,
            self._add_social_media_style,
            self._add_typos_and_abbreviations
        ]
        
        # Apply random variations
        for variation in random.sample(variations, random.randint(1, 3)):
            base_text = variation(base_text)
        
        return base_text
    
    def _generate_non_crisis_sample(self):
        """Generate a single non-crisis sample with variations"""
        pattern_type = random.choice(list(self.non_crisis_patterns.keys()))
        base_text = random.choice(self.non_crisis_patterns[pattern_type])
        
        # Add variations
        variations = [
            self._add_positive_elements,
            self._add_coping_language,
            self._add_support_references,
            self._add_social_media_style,
            self._add_typos_and_abbreviations
        ]
        
        # Apply random variations
        for variation in random.sample(variations, random.randint(1, 3)):
            base_text = variation(base_text)
        
        return base_text
    
    def _add_emotional_intensity(self, text):
        """Add emotional intensity to crisis texts"""
        intensifiers = ["really", "so", "extremely", "completely", "totally", "absolutely"]
        if random.random() > 0.5:
            intensifier = random.choice(intensifiers)
            text = text.replace("I", f"I {intensifier}")
        return text
    
    def _add_temporal_markers(self, text):
        """Add temporal markers"""
        temporal_markers = ["tonight", "today", "right now", "this week", "lately", "recently"]
        if random.random() > 0.6:
            marker = random.choice(temporal_markers)
            text += f" {marker}"
        return text
    
    def _add_personal_context(self, text):
        """Add personal context"""
        contexts = [
            "after everything that's happened",
            "with all the stress I'm under",
            "given my current situation",
            "after months of struggling",
            "with all the pressure I'm facing"
        ]
        if random.random() > 0.7:
            context = random.choice(contexts)
            text += f" {context}"
        return text
    
    def _add_social_media_style(self, text):
        """Add social media style elements"""
        if random.random() > 0.8:
            # Add emojis
            if random.random() > 0.5:
                emojis = ["😔", "😢", "😭", "💔", "😞", "😓", "😪", "🥀"]
                text += f" {random.choice(emojis)}"
        
        if random.random() > 0.7:
            # Add hashtags
            hashtags = ["#depression", "#mentalhealth", "#struggling", "#coping", "#recovery"]
            text += f" {random.choice(hashtags)}"
        
        return text
    
    def _add_typos_and_abbreviations(self, text):
        """Add realistic typos and abbreviations"""
        if random.random() > 0.6:
            # Common abbreviations
            replacements = {
                "you": "u", "are": "r", "to": "2", "for": "4", 
                "before": "b4", "see": "c", "later": "l8r",
                "because": "bc", "though": "tho", "through": "thru"
            }
            
            for old, new in replacements.items():
                if f" {old} " in f" {text} ":
                    text = text.replace(f" {old} ", f" {new} ")
        
        if random.random() > 0.8:
            # Add some typos
            text = text.replace("the", "teh").replace("and", "adn")
        
        return text
    
    def _add_positive_elements(self, text):
        """Add positive elements to non-crisis texts"""
        positive_additions = [
            "and I'm working on it",
            "but I'm staying positive",
            "and I'm not giving up",
            "but I'm taking care of myself",
            "and I'm using my coping skills"
        ]
        if random.random() > 0.6:
            addition = random.choice(positive_additions)
            text += f" {addition}"
        return text
    
    def _add_coping_language(self, text):
        """Add coping language to non-crisis texts"""
        coping_phrases = [
            "I'm using my coping strategies",
            "I'm practicing self-care",
            "I'm reaching out for support",
            "I'm taking it one day at a time",
            "I'm focusing on my recovery"
        ]
        if random.random() > 0.7:
            phrase = random.choice(coping_phrases)
            text += f" {phrase}"
        return text
    
    def _add_support_references(self, text):
        """Add support system references"""
        support_refs = [
            "my therapist", "my counselor", "my support group", 
            "my friends", "my family", "my support system"
        ]
        if random.random() > 0.6:
            ref = random.choice(support_refs)
            text += f" with {ref}"
        return text

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing with multiple techniques
    """
    
    def __init__(self, use_advanced_features=True):
        self.use_advanced_features = use_advanced_features
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def preprocess(self, texts):
        """Preprocess texts with advanced techniques"""
        processed_texts = []
        
        for text in tqdm(texts, desc="Preprocessing texts"):
            # Basic cleaning
            text = self._clean_text(text)
            
            # Tokenization and lemmatization
            tokens = self._tokenize_and_lemmatize(text)
            
            # Remove stop words
            tokens = [token for token in tokens if token not in self.stop_words]
            
            processed_texts.append(' '.join(tokens))
        
        return processed_texts
    
    def extract_advanced_features(self, texts):
        """Extract advanced features from texts"""
        features = []
        
        for text in tqdm(texts, desc="Extracting advanced features"):
            feature_dict = {}
            
            # Basic text features
            feature_dict['text_length'] = len(text)
            feature_dict['word_count'] = len(text.split())
            feature_dict['sentence_count'] = len(text.split('.'))
            feature_dict['avg_word_length'] = np.mean([len(word) for word in text.split()])
            
            # Sentiment features
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            feature_dict.update(sentiment_scores)
            
            # Crisis-specific features
            feature_dict['crisis_keywords'] = self._count_crisis_keywords(text)
            feature_dict['suicidal_language'] = self._count_suicidal_language(text)
            feature_dict['hopelessness_language'] = self._count_hopelessness_language(text)
            feature_dict['help_seeking_language'] = self._count_help_seeking_language(text)
            
            # Emotional intensity
            feature_dict['emotional_intensity'] = self._calculate_emotional_intensity(text)
            feature_dict['urgency_level'] = self._calculate_urgency_level(text)
            
            # Social media features
            feature_dict['has_emoji'] = self._has_emoji(text)
            feature_dict['has_hashtag'] = '#' in text
            feature_dict['has_mention'] = '@' in text
            feature_dict['has_url'] = 'http' in text.lower()
            
            # Typography features
            feature_dict['all_caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            feature_dict['exclamation_count'] = text.count('!')
            feature_dict['question_count'] = text.count('?')
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _clean_text(self, text):
        """Clean text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text)
            return [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        except:
            return text.split()
    
    def _count_crisis_keywords(self, text):
        """Count crisis-related keywords"""
        crisis_keywords = [
            'suicide', 'kill', 'die', 'death', 'end', 'pain', 'suffer',
            'hopeless', 'worthless', 'useless', 'burden', 'alone', 'isolated'
        ]
        return sum(1 for keyword in crisis_keywords if keyword in text.lower())
    
    def _count_suicidal_language(self, text):
        """Count suicidal language patterns"""
        suicidal_patterns = [
            'want to die', 'kill myself', 'end my life', 'commit suicide',
            'take my life', 'end it all', 'suicide', 'kill myself'
        ]
        return sum(1 for pattern in suicidal_patterns if pattern in text.lower())
    
    def _count_hopelessness_language(self, text):
        """Count hopelessness language"""
        hopelessness_words = [
            'hopeless', 'worthless', 'useless', 'burden', 'no reason',
            'pointless', 'meaningless', 'no future', 'can\'t go on'
        ]
        return sum(1 for word in hopelessness_words if word in text.lower())
    
    def _count_help_seeking_language(self, text):
        """Count help-seeking language"""
        help_words = [
            'help', 'support', 'therapy', 'counselor', 'therapist',
            'treatment', 'medication', 'coping', 'recovery'
        ]
        return sum(1 for word in help_words if word in text.lower())
    
    def _calculate_emotional_intensity(self, text):
        """Calculate emotional intensity"""
        intensity_words = [
            'really', 'so', 'extremely', 'completely', 'totally',
            'absolutely', 'incredibly', 'terribly', 'awfully'
        ]
        return sum(1 for word in intensity_words if word in text.lower())
    
    def _calculate_urgency_level(self, text):
        """Calculate urgency level"""
        urgency_words = [
            'now', 'tonight', 'today', 'immediately', 'right now',
            'asap', 'urgent', 'emergency', 'crisis'
        ]
        return sum(1 for word in urgency_words if word in text.lower())
    
    def _has_emoji(self, text):
        """Check if text has emojis"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return bool(emoji_pattern.search(text))

class AdvancedCrisisDetector:
    """
    Advanced crisis detection system with multiple models
    """
    
    def __init__(self, use_ensemble=True, use_advanced_features=True):
        self.use_ensemble = use_ensemble
        self.use_advanced_features = use_advanced_features
        self.preprocessor = AdvancedTextPreprocessor(use_advanced_features)
        self.models = {}
        self.ensemble_model = None
        self.is_fitted = False
        
    def fit(self, texts, labels):
        """Fit the crisis detection model"""
        logger.info("Training advanced crisis detection model...")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess(texts)
        
        # Extract features
        if self.use_advanced_features:
            advanced_features = self.preprocessor.extract_advanced_features(texts)
            # Combine with TF-IDF features
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
            tfidf_features = vectorizer.fit_transform(processed_texts)
            self.vectorizer = vectorizer
            
            # Convert to dense array and combine
            tfidf_dense = tfidf_features.toarray()
            combined_features = np.hstack([tfidf_dense, advanced_features.values])
        else:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
            combined_features = vectorizer.fit_transform(processed_texts).toarray()
            self.vectorizer = vectorizer
        
        if self.use_ensemble:
            # Train multiple models
            self.models['rf'] = RandomForestClassifier(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42
            )
            self.models['svm'] = SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
            self.models['nb'] = MultinomialNB(alpha=0.1)
            self.models['mlp'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
            
            # Train each model
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                model.fit(combined_features, labels)
            
            # Create ensemble
            self.ensemble_model = VotingClassifier(
                estimators=list(self.models.items()),
                voting='soft'
            )
            self.ensemble_model.fit(combined_features, labels)
        else:
            # Train single model
            self.models['rf'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42
            )
            self.models['rf'].fit(combined_features, labels)
        
        self.is_fitted = True
        logger.info("Model training completed!")
    
    def predict(self, texts):
        """Predict crisis labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess(texts)
        
        # Extract features
        if self.use_advanced_features:
            advanced_features = self.preprocessor.extract_advanced_features(texts)
            tfidf_features = self.vectorizer.transform(processed_texts)
            combined_features = np.hstack([tfidf_features.toarray(), advanced_features.values])
        else:
            combined_features = self.vectorizer.transform(processed_texts).toarray()
        
        if self.use_ensemble:
            return self.ensemble_model.predict(combined_features)
        else:
            return self.models['rf'].predict(combined_features)
    
    def predict_proba(self, texts):
        """Predict crisis probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess(texts)
        
        # Extract features
        if self.use_advanced_features:
            advanced_features = self.preprocessor.extract_advanced_features(texts)
            tfidf_features = self.vectorizer.transform(processed_texts)
            combined_features = np.hstack([tfidf_features.toarray(), advanced_features.values])
        else:
            combined_features = self.vectorizer.transform(processed_texts).toarray()
        
        if self.use_ensemble:
            return self.ensemble_model.predict_proba(combined_features)
        else:
            return self.models['rf'].predict_proba(combined_features)
    
    def save(self, filepath):
        """Save the model"""
        model_data = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'vectorizer': self.vectorizer,
            'use_ensemble': self.use_ensemble,
            'use_advanced_features': self.use_advanced_features,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load the model"""
        model_data = joblib.load(filepath)
        
        instance = cls(
            use_ensemble=model_data['use_ensemble'],
            use_advanced_features=model_data['use_advanced_features']
        )
        
        instance.models = model_data['models']
        instance.ensemble_model = model_data['ensemble_model']
        instance.vectorizer = model_data['vectorizer']
        instance.is_fitted = model_data['is_fitted']
        
        return instance

def evaluate_model_comprehensive(y_true, y_pred, y_proba=None):
    """Comprehensive model evaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics

def plot_comprehensive_evaluation(metrics, title="Advanced Crisis Detection Model"):
    """Plot comprehensive evaluation metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # Performance Metrics
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_values = [metrics[name] for name in metric_names]
    
    bars = axes[0, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[0, 1].set_title('Performance Metrics')
    axes[0, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # ROC Curve (if available)
    if 'roc_auc' in metrics:
        axes[0, 2].text(0.5, 0.5, f'ROC AUC: {metrics["roc_auc"]:.3f}', 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 2].set_title('ROC AUC Score')
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
    
    # Precision-Recall (if available)
    if 'average_precision' in metrics:
        axes[1, 0].text(0.5, 0.5, f'Average Precision: {metrics["average_precision"]:.3f}', 
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 0].set_title('Average Precision')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
    
    # Model Summary
    summary_text = f"""
    Model Performance Summary:
    
    Accuracy: {metrics['accuracy']:.3f}
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1-Score: {metrics['f1']:.3f}
    """
    
    if 'roc_auc' in metrics:
        summary_text += f"ROC AUC: {metrics['roc_auc']:.3f}\n"
    if 'average_precision' in metrics:
        summary_text += f"Avg Precision: {metrics['average_precision']:.3f}\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    axes[1, 1].set_title('Model Summary')
    axes[1, 1].axis('off')
    
    # Feature Importance (placeholder)
    axes[1, 2].text(0.5, 0.5, 'Feature Importance\nAnalysis Available\nin Model Object', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    axes[1, 2].set_title('Feature Importance')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the advanced crisis detection system"""
    logger.info("Starting Advanced Crisis Detection System...")
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate comprehensive dataset
    logger.info("Generating comprehensive dataset...")
    generator = AdvancedCrisisDatasetGenerator()
    
    # Generate large dataset
    df = generator.generate_enhanced_dataset(n_samples=15000, crisis_ratio=0.35)
    
    # Save the enhanced dataset
    df.to_csv(f"{data_dir}/enhanced_mental_health_posts.csv", index=False)
    logger.info(f"Saved enhanced dataset with {len(df)} samples")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Train model
    logger.info("Training advanced crisis detection model...")
    detector = AdvancedCrisisDetector(use_ensemble=True, use_advanced_features=True)
    
    # Fit model
    detector.fit(train_df['text'].tolist(), train_df['label'].values)
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_predictions = detector.predict(test_df['text'].tolist())
    test_probabilities = detector.predict_proba(test_df['text'].tolist())[:, 1]
    
    # Calculate metrics
    metrics = evaluate_model_comprehensive(
        test_df['label'].values, 
        test_predictions, 
        test_probabilities
    )
    
    # Print results
    logger.info("Model Evaluation Results:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"{metric}: {value:.4f}")
    
    # Plot results
    fig = plot_comprehensive_evaluation(metrics, "Advanced Crisis Detection Model")
    fig.savefig(f"{data_dir}/advanced_model_evaluation.png", dpi=300, bbox_inches='tight')
    logger.info(f"Evaluation plot saved to {data_dir}/advanced_model_evaluation.png")
    
    # Save model
    detector.save(f"{data_dir}/advanced_crisis_detector.joblib")
    logger.info(f"Model saved to {data_dir}/advanced_crisis_detector.joblib")
    
    # Generate additional synthetic data for edge cases
    logger.info("Generating additional edge case data...")
    edge_cases = generate_edge_case_data()
    edge_df = pd.DataFrame(edge_cases)
    edge_df.to_csv(f"{data_dir}/edge_cases.csv", index=False)
    logger.info(f"Saved {len(edge_df)} edge case samples")
    
    logger.info("Advanced Crisis Detection System completed successfully!")

def generate_edge_case_data():
    """Generate edge case data for better model training"""
    edge_cases = []
    
    # Ambiguous cases
    ambiguous_cases = [
        ("I'm having dark thoughts but I would never act on them", 0),
        ("Sometimes I think about death but I'm not suicidal", 0),
        ("I feel like ending it all but I'm getting help", 0),
        ("I want to die but I'm not going to do anything", 0),
        ("I'm thinking about suicide but I'm calling my therapist", 0),
        ("I feel hopeless but I'm not giving up", 0),
        ("I'm struggling but I'm using my coping skills", 0),
        ("I feel like a burden but I'm working on it", 0),
        ("I'm having suicidal thoughts but I'm safe", 0),
        ("I want to end my pain but I'm getting treatment", 0)
    ]
    
    # High-risk cases
    high_risk_cases = [
        ("I have a plan and I'm going to do it tonight", 1),
        ("I've written my suicide note and I'm ready", 1),
        ("I have the pills and I'm taking them now", 1),
        ("I'm going to jump off the bridge in an hour", 1),
        ("I've made my final decision and I'm at peace", 1),
        ("I'm going to shoot myself when I get home", 1),
        ("I have everything I need to end my life", 1),
        ("I'm going to hang myself tonight", 1),
        ("I've chosen my method and I'm doing it now", 1),
        ("I'm going to overdose on my medication", 1)
    ]
    
    # Recovery cases
    recovery_cases = [
        ("I used to be suicidal but I'm in recovery now", 0),
        ("I've been through crisis but I'm getting better", 0),
        ("I was planning suicide but I got help", 0),
        ("I'm a suicide attempt survivor and I'm healing", 0),
        ("I used to want to die but now I want to live", 0),
        ("I was in crisis but therapy saved my life", 0),
        ("I used to be hopeless but I found hope", 0),
        ("I was suicidal but I'm in treatment now", 0),
        ("I used to think about death but I'm recovering", 0),
        ("I was planning to die but I chose to live", 0)
    ]
    
    edge_cases.extend(ambiguous_cases)
    edge_cases.extend(high_risk_cases)
    edge_cases.extend(recovery_cases)
    
    return edge_cases

if __name__ == "__main__":
    main()
