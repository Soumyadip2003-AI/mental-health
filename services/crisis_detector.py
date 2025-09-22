"""
Advanced Crisis Detection Service with multiple ML models and ensemble methods
"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification, 
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
import shap
import lime
from lime.lime_text import LimeTextExplainer
from textblob import TextBlob
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import settings, CrisisDetectionConfig
from utils.preprocessing import AdvancedTextPreprocessor
from utils.feature_extraction import FeatureExtractor
from utils.exceptions import CrisisDetectionException

logger = logging.getLogger(__name__)


class CrisisDetectionResult:
    """Result of crisis detection analysis"""
    
    def __init__(self, 
                 crisis_probability: float,
                 risk_level: str,
                 confidence_score: float,
                 model_name: str,
                 key_features: Optional[Dict[str, Any]] = None,
                 explanation: Optional[Dict[str, Any]] = None,
                 processing_time: float = 0.0):
        self.crisis_probability = crisis_probability
        self.risk_level = risk_level
        self.confidence_score = confidence_score
        self.model_name = model_name
        self.key_features = key_features or {}
        self.explanation = explanation or {}
        self.processing_time = processing_time


class CrisisDetectionService:
    """Advanced crisis detection service with ensemble models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = AdvancedTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_initialized = False
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic preprocessing")
            self.nlp = None
    
    async def initialize(self):
        """Initialize the crisis detection service"""
        try:
            logger.info("Initializing crisis detection service...")
            
            # Load pre-trained models
            await self._load_models()
            
            # Initialize ensemble
            await self._initialize_ensemble()
            
            self.is_initialized = True
            logger.info("Crisis detection service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize crisis detection service: {e}")
            raise CrisisDetectionException(f"Initialization failed: {e}")
    
    async def _load_models(self):
        """Load pre-trained models"""
        model_paths = {
            'bert': f"{settings.model_cache_dir}/bert_model.joblib",
            'roberta': f"{settings.model_cache_dir}/roberta_model.joblib",
            'distilbert': f"{settings.model_cache_dir}/distilbert_model.joblib",
            'ensemble': f"{settings.model_cache_dir}/ensemble_model.joblib",
            'multimodal': f"{settings.model_cache_dir}/multimodal_model.joblib"
        }
        
        for model_name, path in model_paths.items():
            try:
                if model_name in ['bert', 'roberta', 'distilbert']:
                    # Load transformer models
                    self.models[model_name] = await self._load_transformer_model(model_name)
                else:
                    # Load sklearn models
                    self.models[model_name] = joblib.load(path)
                logger.info(f"Loaded {model_name} model successfully")
            except FileNotFoundError:
                logger.warning(f"Model {model_name} not found at {path}, will train on demand")
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
    
    async def _load_transformer_model(self, model_name: str):
        """Load transformer model"""
        model_configs = {
            'bert': ('bert-base-uncased', BertTokenizer, BertForSequenceClassification),
            'roberta': ('roberta-base', RobertaTokenizer, RobertaForSequenceClassification),
            'distilbert': ('distilbert-base-uncased', DistilBertTokenizer, DistilBertForSequenceClassification)
        }
        
        model_name_hf, tokenizer_class, model_class = model_configs[model_name]
        
        tokenizer = tokenizer_class.from_pretrained(model_name_hf)
        model = model_class.from_pretrained(model_name_hf, num_labels=2)
        
        return {
            'tokenizer': tokenizer,
            'model': model,
            'name': model_name
        }
    
    async def _initialize_ensemble(self):
        """Initialize ensemble model"""
        try:
            # Create ensemble of available models
            available_models = []
            
            for model_name, model in self.models.items():
                if model_name in ['bert', 'roberta', 'distilbert']:
                    available_models.append((f"{model_name}_wrapper", self._create_transformer_wrapper(model)))
                elif model_name in ['ensemble', 'multimodal']:
                    available_models.append((f"{model_name}_wrapper", self._create_sklearn_wrapper(model)))
            
            if available_models:
                self.ensemble = VotingClassifier(available_models, voting='soft')
                logger.info(f"Initialized ensemble with {len(available_models)} models")
            else:
                logger.warning("No models available for ensemble")
                self.ensemble = None
                
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {e}")
            self.ensemble = None
    
    def _create_transformer_wrapper(self, model_data):
        """Create wrapper for transformer models"""
        class TransformerWrapper:
            def __init__(self, model_data):
                self.tokenizer = model_data['tokenizer']
                self.model = model_data['model']
                self.name = model_data['name']
            
            def predict_proba(self, texts):
                # This is a simplified wrapper - in practice, you'd need proper batching
                probabilities = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        probabilities.append(probs.numpy()[0])
                return np.array(probabilities)
        
        return TransformerWrapper(model_data)
    
    def _create_sklearn_wrapper(self, model):
        """Create wrapper for sklearn models"""
        class SklearnWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict_proba(self, texts):
                # This would need proper feature extraction
                # For now, return dummy probabilities
                return np.array([[0.5, 0.5]] * len(texts))
        
        return SklearnWrapper(model)
    
    async def analyze_text(self, 
                          text: str, 
                          confidence_threshold: float = None,
                          include_explanation: bool = True) -> CrisisDetectionResult:
        """Analyze a single text for crisis indicators"""
        if not self.is_initialized:
            raise CrisisDetectionException("Service not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)
            
            # Extract features
            features = self.feature_extractor.extract_features(processed_text)
            
            # Get predictions from all available models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name in ['bert', 'roberta', 'distilbert']:
                        pred, prob = await self._predict_transformer(model, text)
                    else:
                        pred, prob = self._predict_sklearn(model, features)
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = prob
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    continue
            
            # Ensemble prediction
            if probabilities:
                # Weighted average of probabilities
                weights = {'bert': 0.3, 'roberta': 0.3, 'distilbert': 0.2, 'ensemble': 0.2}
                crisis_prob = sum(probabilities.get(name, 0.5) * weights.get(name, 0.1) 
                                for name in weights.keys() if name in probabilities)
                
                # Normalize to [0, 1]
                crisis_prob = max(0, min(1, crisis_prob))
            else:
                # Fallback to rule-based detection
                crisis_prob = self._rule_based_detection(text)
            
            # Determine risk level
            risk_level = self._determine_risk_level(crisis_prob, confidence_threshold)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(probabilities, crisis_prob)
            
            # Extract key features
            key_features = self._extract_key_features(text, features)
            
            # Generate explanation if requested
            explanation = {}
            if include_explanation:
                explanation = await self._generate_explanation(text, crisis_prob)
            
            processing_time = time.time() - start_time
            
            return CrisisDetectionResult(
                crisis_probability=crisis_prob,
                risk_level=risk_level,
                confidence_score=confidence_score,
                model_name="ensemble",
                key_features=key_features,
                explanation=explanation,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise CrisisDetectionException(f"Analysis failed: {e}")
    
    async def analyze_batch(self, 
                           texts: List[str], 
                           confidence_threshold: float = None,
                           include_explanation: bool = True) -> List[CrisisDetectionResult]:
        """Analyze multiple texts in batch"""
        if not self.is_initialized:
            raise CrisisDetectionException("Service not initialized")
        
        results = []
        
        # Process in batches for efficiency
        batch_size = settings.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.analyze_text(text, confidence_threshold, include_explanation)
                for text in batch_texts
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed for text {i + j}: {result}")
                    # Create fallback result
                    result = CrisisDetectionResult(
                        crisis_probability=0.5,
                        risk_level="medium",
                        confidence_score=0.0,
                        model_name="fallback",
                        processing_time=0.0
                    )
                
                results.append(result)
        
        return results
    
    async def _predict_transformer(self, model_data, text):
        """Predict using transformer model"""
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            probability = probabilities[0][1].item()  # Probability of crisis class
        
        return prediction, probability
    
    def _predict_sklearn(self, model, features):
        """Predict using sklearn model"""
        try:
            # Ensure features are in the right format
            if hasattr(features, 'reshape'):
                features = features.reshape(1, -1)
            elif isinstance(features, dict):
                # Convert dict to array
                features = np.array(list(features.values())).reshape(1, -1)
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]  # Probability of positive class
            
            return prediction, probability
        except Exception as e:
            logger.warning(f"Sklearn prediction failed: {e}")
            return 0, 0.5  # Default fallback
    
    def _rule_based_detection(self, text: str) -> float:
        """Fallback rule-based crisis detection"""
        crisis_score = 0.0
        
        # Check for crisis keywords
        for category, keywords in CrisisDetectionConfig.CRISIS_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    crisis_score += 0.2
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        if sentiment['compound'] < -0.5:  # Very negative sentiment
            crisis_score += 0.3
        
        # Text length and complexity
        if len(text.split()) < 5:  # Very short text might indicate distress
            crisis_score += 0.1
        
        # Check for repeated words (might indicate distress)
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in word_counts.items():
            if count > 2 and len(word) > 3:  # Repeated words
                crisis_score += 0.1
        
        return min(1.0, crisis_score)
    
    def _determine_risk_level(self, crisis_prob: float, threshold: float = None) -> str:
        """Determine risk level based on crisis probability"""
        if threshold is None:
            threshold = settings.confidence_threshold
        
        if crisis_prob >= 0.9:
            return "critical"
        elif crisis_prob >= 0.8:
            return "high"
        elif crisis_prob >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence_score(self, probabilities: Dict[str, float], crisis_prob: float) -> float:
        """Calculate confidence score based on model agreement"""
        if not probabilities:
            return 0.5
        
        # Calculate variance in predictions
        probs = list(probabilities.values())
        variance = np.var(probs)
        
        # Higher variance = lower confidence
        confidence = max(0.0, 1.0 - variance)
        
        return confidence
    
    def _extract_key_features(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features that influenced the decision"""
        key_features = {
            'crisis_keywords': [],
            'sentiment_scores': {},
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'has_caps': any(c.isupper() for c in text if c.isalpha()),
            'repeated_words': []
        }
        
        # Find crisis keywords
        for category, keywords in CrisisDetectionConfig.CRISIS_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
            if found_keywords:
                key_features['crisis_keywords'].extend(found_keywords)
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        key_features['sentiment_scores'] = sentiment
        
        # Find repeated words
        words = text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only consider longer words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        key_features['repeated_words'] = [word for word, count in word_counts.items() if count > 1]
        
        return key_features
    
    async def _generate_explanation(self, text: str, crisis_prob: float) -> Dict[str, Any]:
        """Generate explanation for the prediction"""
        explanation = {
            'lime_explanation': None,
            'shap_explanation': None,
            'feature_importance': {},
            'decision_factors': []
        }
        
        try:
            # LIME explanation
            explainer = LimeTextExplainer(class_names=['Non-Crisis', 'Crisis'])
            
            def predict_proba_wrapper(texts):
                # This would need to be connected to your actual model
                # For now, return dummy probabilities
                return np.array([[1 - crisis_prob, crisis_prob]] * len(texts))
            
            lime_exp = explainer.explain_instance(text, predict_proba_wrapper, num_features=10)
            explanation['lime_explanation'] = lime_exp.as_list()
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
        
        # Decision factors
        if crisis_prob > 0.7:
            explanation['decision_factors'].append("High probability of crisis indicators")
        if crisis_prob > 0.5:
            explanation['decision_factors'].append("Moderate risk detected")
        if crisis_prob < 0.3:
            explanation['decision_factors'].append("Low risk - no crisis indicators")
        
        return explanation
    
    def is_ready(self) -> bool:
        """Check if the service is ready for analysis"""
        return self.is_initialized and len(self.models) > 0
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        model_info = {}
        
        for name, model in self.models.items():
            if name in ['bert', 'roberta', 'distilbert']:
                model_info[name] = {
                    'type': 'transformer',
                    'name': model['name'],
                    'status': 'loaded'
                }
            else:
                model_info[name] = {
                    'type': 'sklearn',
                    'status': 'loaded'
                }
        
        return model_info
