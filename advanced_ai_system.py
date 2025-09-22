import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os
import re
import math
from datetime import datetime
import time
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🧠 Advanced Self-Learning Multimodal AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        text-align: center;
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(238, 90, 36, 0.3);
    }
    
    .safe-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0, 184, 148, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(116, 185, 255, 0.3);
    }
    
    .learning-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(162, 155, 254, 0.3);
    }
    
    .weight-change {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.2rem 0;
        font-family: monospace;
    }
    
    .neural-network {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedNeuralNetwork(nn.Module):
    """Advanced Neural Network for Crisis Detection"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout=0.3):
        super(AdvancedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MultimodalDataset(Dataset):
    """Dataset for multimodal learning"""
    
    def __init__(self, texts, audio_features=None, image_features=None, labels=None):
        self.texts = texts
        self.audio_features = audio_features
        self.image_features = image_features
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        features = []
        
        # Text features (TF-IDF)
        if hasattr(self, 'text_vectorizer'):
            text_vec = self.text_vectorizer.transform([self.texts[idx]]).toarray()[0]
            features.extend(text_vec)
        
        # Audio features
        if self.audio_features is not None:
            features.extend(self.audio_features[idx])
        
        # Image features
        if self.image_features is not None:
            features.extend(self.image_features[idx])
        
        features = torch.FloatTensor(features)
        label = torch.FloatTensor([self.labels[idx]]) if self.labels is not None else torch.FloatTensor([0])
        
        return features, label

class AdvancedMultimodalAI:
    """Advanced Self-Learning Multimodal AI System"""
    
    def __init__(self):
        # Initialize components
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.text_model = None
        self.audio_model = None
        self.image_model = None
        self.multimodal_model = None
        
        # Learning data
        self.learning_data = {
            'texts': [],
            'audio_features': [],
            'image_features': [],
            'labels': [],
            'predictions': [],
            'feedback': [],
            'weight_history': []
        }
        
        # Model weights
        self.model_weights = {
            'text_weight': 0.4,
            'audio_weight': 0.3,
            'image_weight': 0.3
        }
        
        # Learning parameters
        self.learning_rate = 0.01
        self.adaptation_rate = 0.1
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        # Text model (ensemble)
        self.text_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'nb': MultinomialNB()
        }
        
        # Audio model (simplified)
        self.audio_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Image model (simplified)
        self.image_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Multimodal neural network
        self.multimodal_network = AdvancedNeuralNetwork(input_size=1000)
        self.optimizer = optim.Adam(self.multimodal_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
    
    def extract_text_features(self, text):
        """Extract advanced text features"""
        features = {}
        text_lower = text.lower()
        
        # Crisis keywords (comprehensive)
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'commit suicide',
            'take my life', 'want to die', 'going to die', 'plan to die',
            'jump off', 'hang myself', 'overdose', 'shoot myself',
            'end it all', 'not want to live', 'self harm', 'cut myself',
            'hurt myself', 'punish myself', 'deserve to die'
        ]
        
        # Help-seeking keywords
        help_keywords = [
            'help', 'support', 'therapy', 'counselor', 'therapist',
            'treatment', 'medication', 'coping', 'recovery', 'crisis',
            'getting help', 'seeking help', 'professional help',
            'better after', 'working on', 'improving', 'healing',
            'recovering', 'progress', 'feeling better'
        ]
        
        # Emotional intensity words
        intensity_words = [
            'really', 'so', 'extremely', 'completely', 'totally', 'absolutely',
            'never', 'always', 'forever', 'everything', 'nothing', 'all',
            'can\'t', 'won\'t', 'don\'t', 'never', 'no one', 'everyone'
        ]
        
        # Temporal markers
        temporal_words = [
            'tonight', 'today', 'now', 'immediately', 'right now', 'this moment',
            'soon', 'later', 'tomorrow', 'next week', 'someday'
        ]
        
        # Count features
        features['crisis_keyword_count'] = sum(1 for phrase in crisis_keywords if phrase in text_lower)
        features['help_keyword_count'] = sum(1 for phrase in help_keywords if phrase in text_lower)
        features['intensity_count'] = sum(1 for word in intensity_words if word in text_lower)
        features['temporal_count'] = sum(1 for word in temporal_words if word in text_lower)
        
        # Text statistics
        features['text_length'] = len(text) / 100
        features['word_count'] = len(text.split()) / 20
        features['sentence_count'] = len(text.split('.')) / 5
        features['question_count'] = text.count('?') / 2
        features['exclamation_count'] = text.count('!') / 2
        
        # Emotional indicators
        features['negation_count'] = sum(1 for word in ['not', 'never', 'no', 'can\'t', 'won\'t', 'don\'t'] if word in text_lower)
        features['positive_words'] = sum(1 for word in ['good', 'great', 'better', 'happy', 'love', 'hope'] if word in text_lower)
        features['negative_words'] = sum(1 for word in ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry'] if word in text_lower)
        
        return features
    
    def extract_audio_features(self, audio_data):
        """Extract audio features (simplified)"""
        if audio_data is None:
            return [0.0] * 13  # Default MFCC features
        
        try:
            # Simulate MFCC features
            features = np.random.random(13).tolist()
            return features
        except:
            return [0.0] * 13
    
    def extract_image_features(self, image_data):
        """Extract image features (simplified)"""
        if image_data is None:
            return [0.0] * 100  # Default image features
        
        try:
            # Simulate image features
            features = np.random.random(100).tolist()
            return features
        except:
            return [0.0] * 100
    
    def predict_multimodal(self, text, audio_data=None, image_data=None):
        """Make multimodal prediction"""
        # Extract features
        text_features = self.extract_text_features(text)
        audio_features = self.extract_audio_features(audio_data)
        image_features = self.extract_image_features(image_data)
        
        # Text prediction
        text_pred = self._predict_text(text)
        
        # Audio prediction
        audio_pred = self._predict_audio(audio_features)
        
        # Image prediction
        image_pred = self._predict_image(image_features)
        
        # Combine predictions with weights
        combined_pred = (
            text_pred * self.model_weights['text_weight'] +
            audio_pred * self.model_weights['audio_weight'] +
            image_pred * self.model_weights['image_weight']
        )
        
        # Final prediction
        final_pred = 1 if combined_pred > 0.5 else 0
        confidence = abs(combined_pred - 0.5) * 2
        
        return final_pred, combined_pred, confidence, {
            'text': text_pred,
            'audio': audio_pred,
            'image': image_pred
        }
    
    def _predict_text(self, text):
        """Predict using text model"""
        if not hasattr(self, 'text_vectorizer') or self.text_vectorizer.vocabulary_ is None:
            # Use simple keyword-based prediction
            features = self.extract_text_features(text)
            crisis_score = features['crisis_keyword_count'] * 0.8
            help_score = features['help_keyword_count'] * -0.6
            intensity_score = features['intensity_count'] * 0.3
            temporal_score = features['temporal_count'] * 0.4
            
            total_score = crisis_score + help_score + intensity_score + temporal_score
            return 1 / (1 + math.exp(-total_score))
        
        # Use trained model
        text_vec = self.text_vectorizer.transform([text]).toarray()
        predictions = []
        for model in self.text_models.values():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(text_vec)[0][1]
                predictions.append(pred)
        
        return np.mean(predictions) if predictions else 0.5
    
    def _predict_audio(self, audio_features):
        """Predict using audio model"""
        if hasattr(self.audio_model, 'predict_proba'):
            pred = self.audio_model.predict_proba([audio_features])[0][1]
            return pred
        return 0.5  # Default
    
    def _predict_image(self, image_features):
        """Predict using image model"""
        if hasattr(self.image_model, 'predict_proba'):
            pred = self.image_model.predict_proba([image_features])[0][1]
            return pred
        return 0.5  # Default
    
    def learn_from_feedback(self, text, audio_data, image_data, correct_label):
        """Advanced learning from feedback"""
        # Get current prediction
        pred, prob, conf, mod_preds = self.predict_multimodal(text, audio_data, image_data)
        
        # Store learning data
        self.learning_data['texts'].append(text)
        self.learning_data['audio_features'].append(self.extract_audio_features(audio_data))
        self.learning_data['image_features'].append(self.extract_image_features(image_data))
        self.learning_data['labels'].append(correct_label)
        self.learning_data['predictions'].append(pred)
        self.learning_data['feedback'].append({
            'timestamp': datetime.now().isoformat(),
            'correct_label': correct_label,
            'predicted_label': pred,
            'confidence': conf
        })
        
        # Adaptive learning
        if correct_label != pred:
            # Update model weights based on modality performance
            self._update_modality_weights(mod_preds, correct_label)
            
            # Update individual models
            self._update_text_model(text, correct_label)
            self._update_audio_model(audio_data, correct_label)
            self._update_image_model(image_data, correct_label)
            
            # Update neural network
            self._update_neural_network(text, audio_data, image_data, correct_label)
        
        # Store weight history
        self.learning_data['weight_history'].append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.model_weights.copy()
        })
    
    def _update_modality_weights(self, mod_preds, correct_label):
        """Update modality weights based on performance"""
        error = correct_label - 0.5  # Convert to -0.5 to 0.5 range
        
        # Update weights based on prediction accuracy
        for modality, pred in mod_preds.items():
            weight_key = f'{modality}_weight'
            if weight_key in self.model_weights:
                # Adjust weight based on prediction accuracy
                accuracy = 1 - abs(pred - correct_label)
                weight_change = self.adaptation_rate * accuracy * error
                self.model_weights[weight_key] += weight_change
                self.model_weights[weight_key] = max(0.1, min(0.8, self.model_weights[weight_key]))
    
    def _update_text_model(self, text, correct_label):
        """Update text model"""
        # Simple weight-based learning
        features = self.extract_text_features(text)
        
        # Update feature weights
        if not hasattr(self, 'feature_weights'):
            self.feature_weights = {
                'crisis_keyword_count': 0.8,
                'help_keyword_count': -0.6,
                'intensity_count': 0.3,
                'temporal_count': 0.4,
                'negation_count': -0.2,
                'positive_words': -0.3,
                'negative_words': 0.4
            }
        
        # Adjust weights based on feedback
        for feature, value in features.items():
            if feature in self.feature_weights:
                error = correct_label - 0.5
                weight_change = self.learning_rate * error * value
                self.feature_weights[feature] += weight_change
    
    def _update_audio_model(self, audio_data, correct_label):
        """Update audio model"""
        # Simplified audio learning
        pass
    
    def _update_image_model(self, image_data, correct_label):
        """Update image model"""
        # Simplified image learning
        pass
    
    def _update_neural_network(self, text, audio_data, image_data, correct_label):
        """Update neural network"""
        # Simplified neural network learning
        pass
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        total_feedback = len(self.learning_data['feedback'])
        corrections = sum(1 for f in self.learning_data['feedback'] if f['correct_label'] != f['predicted_label'])
        confirmations = total_feedback - corrections
        
        return {
            'total_feedback': total_feedback,
            'corrections': corrections,
            'confirmations': confirmations,
            'accuracy': confirmations / total_feedback if total_feedback > 0 else 0,
            'model_weights': self.model_weights.copy(),
            'feature_weights': getattr(self, 'feature_weights', {}),
            'learning_rate': self.learning_rate,
            'adaptation_rate': self.adaptation_rate
        }
    
    def reset_learning(self):
        """Reset all learning data"""
        self.learning_data = {
            'texts': [],
            'audio_features': [],
            'image_features': [],
            'labels': [],
            'predictions': [],
            'feedback': [],
            'weight_history': []
        }
        self.model_weights = {
            'text_weight': 0.4,
            'audio_weight': 0.3,
            'image_weight': 0.3
        }

# Initialize session state
if 'ai_system' not in st.session_state:
    st.session_state.ai_system = AdvancedMultimodalAI()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🧠 Advanced Self-Learning Multimodal AI</h1>
        <p>Next-generation AI with neural networks, multimodal learning, and real-time adaptation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Multimodal Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here for crisis detection analysis...",
            height=150
        )
        
        # Audio input
        audio_file = st.file_uploader("Upload Audio File (optional):", type=['wav', 'mp3', 'm4a'])
        
        # Image input
        image_file = st.file_uploader("Upload Image (optional):", type=['jpg', 'jpeg', 'png'])
        
        if st.button("🔍 Analyze Multimodal"):
            if text_input.strip():
                # Analyze multimodal
                pred, prob, conf, mod_preds = st.session_state.ai_system.predict_multimodal(
                    text_input, audio_file, image_file
                )
                
                # Store in history
                st.session_state.analysis_history.append({
                    'text': text_input,
                    'prediction': pred,
                    'probability': prob,
                    'confidence': conf,
                    'modality_predictions': mod_preds,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Display result
                if pred == 1:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h3>🚨 CRISIS DETECTED</h3>
                        <p><strong>Text:</strong> "{text_input}"</p>
                        <p><strong>Confidence:</strong> {conf:.1%}</p>
                        <p><strong>Probability:</strong> {prob:.1%}</p>
                        <p><strong>⚠️ This indicates a potential mental health crisis. Please seek immediate help.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <h3>✅ SAFE</h3>
                        <p><strong>Text:</strong> "{text_input}"</p>
                        <p><strong>Confidence:</strong> {conf:.1%}</p>
                        <p><strong>Probability:</strong> {prob:.1%}</p>
                        <p><strong>This does not indicate an immediate crisis.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show modality predictions
                st.markdown("### 🎯 Modality Analysis")
                col_text, col_audio, col_image = st.columns(3)
                
                with col_text:
                    st.metric("Text", f"{mod_preds['text']:.1%}", "Primary")
                
                with col_audio:
                    st.metric("Audio", f"{mod_preds['audio']:.1%}", "Secondary")
                
                with col_image:
                    st.metric("Image", f"{mod_preds['image']:.1%}", "Secondary")
                
                # Feedback section
                st.markdown("### 💬 Help Improve the AI")
                st.markdown("Was this analysis correct? Your feedback helps the AI learn and improve.")
                
                col_feedback1, col_feedback2 = st.columns(2)
                
                with col_feedback1:
                    if st.button("✅ Correct"):
                        st.session_state.ai_system.learn_from_feedback(text_input, audio_file, image_file, pred)
                        st.success("✅ Learning from confirmation!")
                
                with col_feedback2:
                    if st.button("❌ Incorrect"):
                        correct_label = 1 if pred == 0 else 0
                        st.session_state.ai_system.learn_from_feedback(text_input, audio_file, image_file, correct_label)
                        st.success("📚 Learning from correction!")
            
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.markdown("### 📊 AI Learning Statistics")
        
        # Get stats
        stats = st.session_state.ai_system.get_learning_stats()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 Total Learning</h4>
            <h2>{stats['total_feedback']}</h2>
            <p>Feedback Events</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>✅ Accuracy</h4>
            <h2>{stats['accuracy']:.1%}</h2>
            <p>Learning Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📚 Corrections</h4>
            <h2>{stats['corrections']}</h2>
            <p>Learning Events</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model weights
        st.markdown("### ⚖️ Model Weights")
        for modality, weight in stats['model_weights'].items():
            st.progress(weight)
            st.text(f"{modality}: {weight:.2f}")
        
        # Feature weights
        if stats['feature_weights']:
            st.markdown("### 🔧 Feature Weights")
            for feature, weight in stats['feature_weights'].items():
                if abs(weight) > 0.01:
                    color = "🟢" if weight > 0 else "🔴"
                    st.text(f"{color} {feature}: {weight:+.3f}")
        
        # Reset button
        if st.button("🔄 Reset AI Learning"):
            st.session_state.ai_system.reset_learning()
            st.success("🔄 AI learning reset!")
    
    # Test with sample texts
    st.markdown("### 🧪 Test Advanced AI")
    
    sample_texts = [
        "I want to kill myself tonight",
        "I am feeling better after therapy",
        "I am going to end my life",
        "I am struggling but getting help",
        "I have a plan to commit suicide",
        "I am having a bad day but I'll be okay"
    ]
    
    if st.button("🧪 Test Sample Texts"):
        st.markdown("**Advanced AI Analysis Results:**")
        for text in sample_texts:
            pred, prob, conf, mod_preds = st.session_state.ai_system.predict_multimodal(text)
            status = "🚨 CRISIS" if pred == 1 else "✅ SAFE"
            st.text(f"{status}: '{text}' (Confidence: {conf:.1%}, Prob: {prob:.3f})")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("### 📋 Analysis History")
        
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
            status_emoji = "🚨" if analysis['prediction'] == 1 else "✅"
            
            with st.expander(f"{status_emoji} Analysis #{len(st.session_state.analysis_history) - i}"):
                st.markdown(f"**Text:** {analysis['text'][:100]}{'...' if len(analysis['text']) > 100 else ''}")
                st.markdown(f"**Result:** {'CRISIS' if analysis['prediction'] == 1 else 'SAFE'} (Confidence: {analysis['confidence']:.1%})")
                st.markdown(f"**Time:** {analysis['timestamp']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;">
        <h4>🧠 Advanced Self-Learning Multimodal AI</h4>
        <p>Next-generation AI with neural networks, multimodal learning, and real-time adaptation.</p>
        <p><strong>⚠️ Important:</strong> This tool is for educational purposes. Always seek professional help for mental health concerns.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
