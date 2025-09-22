import streamlit as st
import numpy as np
import json
import os
import re
import math
from datetime import datetime
import time
from collections import Counter, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🧠 Ultimate Advanced Multimodal AI",
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
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(238, 90, 36, 0.3);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .safe-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0, 184, 148, 0.3);
        animation: slideIn 0.3s ease-out;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
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
    
    .ai-status {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .modality-option {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .modality-option:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedMultimodalAI:
    """Ultimate Advanced Self-Learning Multimodal AI System"""
    
    def __init__(self):
        # Initialize AI components
        self.text_analyzer = AdvancedTextAnalyzer()
        self.audio_analyzer = AdvancedAudioAnalyzer()
        self.image_analyzer = AdvancedImageAnalyzer()
        self.neural_network = AdvancedNeuralNetwork()
        
        # Learning data
        self.learning_data = {
            'texts': [],
            'audio_features': [],
            'image_features': [],
            'labels': [],
            'predictions': [],
            'feedback': [],
            'weight_history': [],
            'modality_usage': defaultdict(int),
            'accuracy_history': []
        }
        
        # Model weights (adaptive)
        self.model_weights = {
            'text_weight': 0.7,
            'audio_weight': 0.2,
            'image_weight': 0.1
        }
        
        # Learning parameters
        self.learning_rate = 0.03
        self.adaptation_rate = 0.01
        self.momentum = 0.9
        self.weight_momentum = defaultdict(float)
        
        # AI status
        self.ai_status = "ACTIVE"
        self.learning_enabled = True
        self.adaptation_enabled = True
        self.performance_score = 100.0
    
    def predict_multimodal(self, text, audio_data=None, image_data=None, selected_modalities=None):
        """Make advanced multimodal prediction"""
        if selected_modalities is None:
            selected_modalities = ['Text']
        
        # Extract features from all modalities
        text_features = self.text_analyzer.extract_features(text) if 'Text' in selected_modalities else None
        audio_features = self.audio_analyzer.extract_features(audio_data) if 'Audio' in selected_modalities and audio_data else None
        image_features = self.image_analyzer.extract_features(image_data) if 'Image' in selected_modalities and image_data else None
        
        # Get predictions from each modality
        predictions = {}
        weights_sum = 0
        
        if text_features is not None:
            predictions['text'] = self.text_analyzer.predict(text_features)
            weights_sum += self.model_weights['text_weight']
            self.learning_data['modality_usage']['text'] += 1
        
        if audio_features:
            predictions['audio'] = self.audio_analyzer.predict(audio_features)
            weights_sum += self.model_weights['audio_weight']
            self.learning_data['modality_usage']['audio'] += 1
        
        if image_features:
            predictions['image'] = self.image_analyzer.predict(image_features)
            weights_sum += self.model_weights['image_weight']
            self.learning_data['modality_usage']['image'] += 1
        
        # Neural network fusion
        combined_features = []
        if text_features:
            combined_features.extend(list(text_features.values()))
        if audio_features:
            combined_features.extend(audio_features)
        if image_features:
            combined_features.extend(image_features)
        
        neural_pred = self.neural_network.predict(combined_features) if combined_features else 0.5
        
        # Weighted ensemble prediction
        ensemble_pred = 0
        for modality, pred in predictions.items():
            weight_key = f'{modality}_weight'
            if weight_key in self.model_weights:
                normalized_weight = self.model_weights[weight_key] / weights_sum if weights_sum > 0 else 0
                ensemble_pred += pred * normalized_weight
        
        # Add neural network contribution
        final_pred_prob = 0.9 * ensemble_pred + 0.1 * neural_pred

        # Neutral-text guardrail: very short texts with no risk signals should not be crisis
        if text_features is not None:
            risk_signal_count = (
                text_features.get('crisis_keywords', 0)
                + text_features.get('negative_words', 0)
                + text_features.get('intensity_words', 0)
                + text_features.get('temporal_words', 0)
            )
            short_text = text_features.get('word_count', 0) <= 0.2  # ~<= 4 words
            if risk_signal_count == 0 and short_text:
                # Push probability below decision threshold for neutral inputs
                final_pred_prob = min(final_pred_prob, 0.49)
        
        # Final prediction
        # Slightly higher threshold to reduce false positives
        final_pred = 1 if final_pred_prob >= 0.6 else 0
        confidence = abs(final_pred_prob - 0.5) * 2
        
        # Return all predictions
        return final_pred, final_pred_prob, confidence, {
            'text': predictions.get('text', 0.5),
            'audio': predictions.get('audio', 0.5),
            'image': predictions.get('image', 0.5),
            'neural': neural_pred,
            'ensemble': ensemble_pred
        }
    
    def learn_from_feedback(self, text, audio_data, image_data, correct_label, selected_modalities):
        """Advanced learning from feedback"""
        # Get current prediction
        pred, prob, conf, mod_preds = self.predict_multimodal(text, audio_data, image_data, selected_modalities)
        
        # Store learning data
        self.learning_data['texts'].append(text)
        self.learning_data['audio_features'].append(self.audio_analyzer.extract_features(audio_data) if audio_data else None)
        self.learning_data['image_features'].append(self.image_analyzer.extract_features(image_data) if image_data else None)
        self.learning_data['labels'].append(correct_label)
        self.learning_data['predictions'].append(pred)
        
        # Record feedback
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'correct_label': correct_label,
            'predicted_label': pred,
            'confidence': conf,
            'modalities_used': selected_modalities,
            'error': abs(correct_label - pred)
        }
        self.learning_data['feedback'].append(feedback_entry)
        
        # Update accuracy history
        recent_feedback = self.learning_data['feedback'][-20:]  # Last 20 feedback
        if recent_feedback:
            accuracy = sum(1 for f in recent_feedback if f['error'] == 0) / len(recent_feedback)
            self.learning_data['accuracy_history'].append(accuracy)
            self.performance_score = accuracy * 100
        
        # Learn from any mistake; scale by error magnitude and confidence
        if self.learning_enabled:
            error_magnitude = abs(correct_label - pred)
            # If prediction was wrong, amplify by confidence; if correct, tiny reinforcement
            effective_lr_scale = (conf if correct_label != pred else 0.05)
            
            # Update individual analyzers
            if 'Text' in selected_modalities:
                self.text_analyzer.learn(text, correct_label, error_magnitude * effective_lr_scale)
            if 'Audio' in selected_modalities and audio_data:
                self.audio_analyzer.learn(audio_data, correct_label, error_magnitude * effective_lr_scale)
            if 'Image' in selected_modalities and image_data:
                self.image_analyzer.learn(image_data, correct_label, error_magnitude * effective_lr_scale)
            
            # Update neural network
            self.neural_network.learn(text, audio_data, image_data, correct_label, selected_modalities)
            
            # Adaptive weight adjustment with momentum
            if self.adaptation_enabled and correct_label != pred:
                self._adapt_weights_with_momentum(mod_preds, correct_label, selected_modalities)
        
        # Store weight history
        self.learning_data['weight_history'].append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.model_weights.copy(),
            'performance': self.performance_score
        })
    
    def _adapt_weights_with_momentum(self, mod_preds, correct_label, selected_modalities):
        """Adaptive weight adjustment with momentum"""
        error = correct_label - 0.5
        
        # Calculate performance for each modality
        modality_performance = {}
        for modality in ['text', 'audio', 'image']:
            if modality in mod_preds:
                accuracy = 1 - abs(mod_preds[modality] - correct_label)
                modality_performance[modality] = accuracy
        
        # Update weights with momentum
        total_performance = sum(modality_performance.values())
        if total_performance > 0:
            for modality, performance in modality_performance.items():
                weight_key = f'{modality}_weight'
                if weight_key in self.model_weights:
                    # Calculate gradient
                    gradient = self.adaptation_rate * (performance / total_performance - self.model_weights[weight_key])
                    
                    # Apply momentum
                    self.weight_momentum[weight_key] = self.momentum * self.weight_momentum[weight_key] + gradient
                    
                    # Update weight
                    self.model_weights[weight_key] += self.weight_momentum[weight_key]
                    
                    # Clamp weights
                    self.model_weights[weight_key] = max(0.05, min(0.9, self.model_weights[weight_key]))
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for key in self.model_weights:
                self.model_weights[key] /= total_weight
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        total_feedback = len(self.learning_data['feedback'])
        corrections = sum(1 for f in self.learning_data['feedback'] if f['error'] > 0)
        confirmations = total_feedback - corrections
        
        # Calculate recent accuracy
        recent_accuracy = 0
        if self.learning_data['accuracy_history']:
            recent_accuracy = self.learning_data['accuracy_history'][-1]
        
        # Calculate modality usage percentages
        total_usage = sum(self.learning_data['modality_usage'].values())
        modality_percentages = {}
        if total_usage > 0:
            for modality, count in self.learning_data['modality_usage'].items():
                modality_percentages[modality] = (count / total_usage) * 100
        
        return {
            'total_feedback': total_feedback,
            'corrections': corrections,
            'confirmations': confirmations,
            'accuracy': recent_accuracy,
            'performance_score': self.performance_score,
            'model_weights': self.model_weights.copy(),
            'learning_rate': self.learning_rate,
            'adaptation_rate': self.adaptation_rate,
            'ai_status': self.ai_status,
            'learning_enabled': self.learning_enabled,
            'adaptation_enabled': self.adaptation_enabled,
            'text_weights': self.text_analyzer.get_feature_weights(),
            'modality_usage': dict(self.learning_data['modality_usage']),
            'modality_percentages': modality_percentages,
            'accuracy_history': self.learning_data['accuracy_history'][-10:] if self.learning_data['accuracy_history'] else []
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
            'weight_history': [],
            'modality_usage': defaultdict(int),
            'accuracy_history': []
        }
        self.model_weights = {
            'text_weight': 0.7,
            'audio_weight': 0.2,
            'image_weight': 0.1
        }
        self.weight_momentum = defaultdict(float)
        self.performance_score = 100.0
        self.text_analyzer.reset()
        self.audio_analyzer.reset()
        self.image_analyzer.reset()
        self.neural_network.reset()

class AdvancedTextAnalyzer:
    """Advanced Text Analysis with Deep Learning"""
    
    def __init__(self):
        self.feature_weights = {
            'crisis_keywords': 0.9,
            'help_keywords': -0.7,
            'intensity_words': 0.4,
            'temporal_words': 0.5,
            'negation_words': -0.3,
            'positive_words': -0.4,
            'negative_words': 0.5,
            'text_length': 0.02,
            'word_count': 0.02,
            'sentence_count': 0.01,
            'exclamation_count': 0.05,
            'question_count': -0.02
        }
        self.learning_rate = 0.02
        self.weight_history = []
    
    def extract_features(self, text):
        """Extract comprehensive text features"""
        if not text:
            return {}
            
        features = {}
        text_lower = text.lower()
        
        # Crisis keywords
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'commit suicide',
            'take my life', 'want to die', 'going to die', 'plan to die',
            'jump off', 'hang myself', 'overdose', 'shoot myself',
            'end it all', 'not want to live', 'self harm', 'cut myself',
            'hurt myself', 'no point', 'give up', 'can\'t go on'
        ]
        
        # Help-seeking keywords
        help_keywords = [
            'help', 'support', 'therapy', 'counselor', 'therapist',
            'treatment', 'medication', 'coping', 'recovery', 'crisis',
            'getting help', 'seeking help', 'professional help',
            'better after', 'working on', 'improving', 'healing',
            'getting better', 'feeling better', 'therapy helps'
        ]
        
        # Emotional intensity
        intensity_words = [
            'really', 'so', 'extremely', 'completely', 'totally', 'absolutely',
            'never', 'always', 'forever', 'everything', 'nothing', 'very'
        ]
        
        # Temporal markers
        temporal_words = [
            'tonight', 'today', 'now', 'immediately', 'right now', 'this moment',
            'soon', 'later', 'tomorrow', 'never again'
        ]
        
        # Negation words
        negation_words = [
            'not', 'never', 'no', 'can\'t', 'won\'t', 'don\'t', 'shouldn\'t'
        ]
        
        # Positive words
        positive_words = [
            'good', 'great', 'better', 'happy', 'love', 'hope', 'joy',
            'peace', 'calm', 'safe', 'secure', 'positive', 'improving'
        ]
        
        # Negative words
        negative_words = [
            'bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'fear',
            'anxiety', 'depression', 'hopeless', 'worthless', 'pain', 'hurt'
        ]
        
        # Count features
        features['crisis_keywords'] = sum(1 for phrase in crisis_keywords if phrase in text_lower)
        features['help_keywords'] = sum(1 for phrase in help_keywords if phrase in text_lower)
        features['intensity_words'] = sum(1 for word in intensity_words if word in text_lower)
        features['temporal_words'] = sum(1 for word in temporal_words if word in text_lower)
        features['negation_words'] = sum(1 for word in negation_words if word in text_lower)
        features['positive_words'] = sum(1 for word in positive_words if word in text_lower)
        features['negative_words'] = sum(1 for word in negative_words if word in text_lower)
        
        # Text statistics (guard smaller texts to avoid false positives)
        features['text_length'] = min(len(text) / 100, 5)  # Capped
        features['word_count'] = min(len(text.split()) / 20, 5)  # Capped (~20 words => 1.0)
        features['sentence_count'] = min(len(text.split('.')) / 5, 3)  # Capped
        features['exclamation_count'] = min(text.count('!'), 5)
        features['question_count'] = min(text.count('?'), 5)
        
        return features
    
    def predict(self, features):
        """Predict using text features"""
        if not features:
            return 0.5
            
        score = 0
        for feature, value in features.items():
            if feature in self.feature_weights:
                score += self.feature_weights[feature] * value
        
        # Advanced sigmoid with temperature
        temperature = 0.9
        # Bias away from crisis when there are no crisis signals
        base_bias = 0.0
        if features.get('crisis_keywords', 0) == 0 and features.get('negative_words', 0) == 0:
            base_bias = -0.4
        probability = 1 / (1 + math.exp(-(score + base_bias) / temperature))
        return probability
    
    def learn(self, text, correct_label, error_magnitude):
        """Learn from feedback with error magnitude"""
        features = self.extract_features(text)
        if not features:
            return
            
        # Only learn from significant errors
        if error_magnitude > 0.3:
            error = (correct_label - 0.5) * 2  # Scale error
            
            # Update weights with bounded learning
            for feature, value in features.items():
                if feature in self.feature_weights and value > 0:
                    # Calculate gradient with decay
                    gradient = self.learning_rate * error * value * error_magnitude
                    
                    # Update weight with bounds
                    self.feature_weights[feature] += gradient
                    self.feature_weights[feature] = max(-2.0, min(2.0, self.feature_weights[feature]))
            
            # Store weight update
            self.weight_history.append({
                'timestamp': datetime.now().isoformat(),
                'weights': self.feature_weights.copy()
            })
    
    def get_feature_weights(self):
        """Get current feature weights"""
        return self.feature_weights.copy()
    
    def reset(self):
        """Reset analyzer"""
        self.feature_weights = {
            'crisis_keywords': 0.9,
            'help_keywords': -0.7,
            'intensity_words': 0.4,
            'temporal_words': 0.5,
            'negation_words': -0.3,
            'positive_words': -0.4,
            'negative_words': 0.5,
            'text_length': 0.02,
            'word_count': 0.02,
            'sentence_count': 0.01,
            'exclamation_count': 0.05,
            'question_count': -0.02
        }
        self.weight_history = []

class AdvancedAudioAnalyzer:
    """Advanced Audio Analysis"""
    
    def __init__(self):
        self.feature_weights = np.random.randn(13) * 0.1
        self.learning_rate = 0.01
    
    def extract_features(self, audio_data):
        """Extract audio features (MFCC simulation)"""
        if audio_data is None:
            return None
        
        # Simulate MFCC features with some structure
        base_features = np.random.randn(13) * 0.5
        # Add some correlation structure
        for i in range(1, 13):
            base_features[i] += base_features[i-1] * 0.3
        
        return base_features.tolist()
    
    def predict(self, features):
        """Predict using audio features"""
        if features is None:
            return 0.5
            
        score = np.dot(self.feature_weights, features)
        probability = 1 / (1 + math.exp(-score))
        return probability
    
    def learn(self, audio_data, correct_label, error_magnitude):
        """Learn from feedback"""
        features = self.extract_features(audio_data)
        if features is None:
            return
            
        if error_magnitude > 0.3:
            error = (correct_label - 0.5) * 2
            gradient = self.learning_rate * error * np.array(features) * error_magnitude
            self.feature_weights += gradient
            self.feature_weights = np.clip(self.feature_weights, -1.0, 1.0)
    
    def reset(self):
        """Reset analyzer"""
        self.feature_weights = np.random.randn(13) * 0.1

class AdvancedImageAnalyzer:
    """Advanced Image Analysis"""
    
    def __init__(self):
        self.feature_weights = np.random.randn(100) * 0.1
        self.learning_rate = 0.01
    
    def extract_features(self, image_data):
        """Extract image features (CNN simulation)"""
        if image_data is None:
            return None
        
        # Simulate CNN features
        base_features = np.random.randn(100) * 0.3
        # Add some structure
        for i in range(10):
            base_features[i*10:(i+1)*10] += np.sin(i) * 0.2
        
        return base_features.tolist()
    
    def predict(self, features):
        """Predict using image features"""
        if features is None:
            return 0.5
            
        score = np.dot(self.feature_weights, features)
        probability = 1 / (1 + math.exp(-score))
        return probability
    
    def learn(self, image_data, correct_label, error_magnitude):
        """Learn from feedback"""
        features = self.extract_features(image_data)
        if features is None:
            return
            
        if error_magnitude > 0.3:
            error = (correct_label - 0.5) * 2
            gradient = self.learning_rate * error * np.array(features) * error_magnitude
            self.feature_weights += gradient
            self.feature_weights = np.clip(self.feature_weights, -1.0, 1.0)
    
    def reset(self):
        """Reset analyzer"""
        self.feature_weights = np.random.randn(100) * 0.1

class AdvancedNeuralNetwork:
    """Advanced Neural Network for fusion"""
    
    def __init__(self):
        self.input_size = 200  # Approximate combined feature size
        self.hidden_size = 64
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.weights2 = np.random.randn(self.hidden_size, 1) * 0.1
        self.bias1 = np.zeros(self.hidden_size)
        self.bias2 = np.zeros(1)
        self.learning_rate = 0.001
    
    def predict(self, features):
        """Neural network forward pass"""
        if not features:
            return 0.5
            
        # Pad or truncate features to input size
        padded_features = np.zeros(self.input_size)
        feature_array = np.array(features)
        padded_features[:min(len(feature_array), self.input_size)] = feature_array[:self.input_size]
        
        # Forward pass
        hidden = np.tanh(np.dot(padded_features, self.weights1) + self.bias1)
        output = 1 / (1 + np.exp(-(np.dot(hidden, self.weights2) + self.bias2)))
        
        return float(output[0])
    
    def learn(self, text, audio_data, image_data, correct_label, selected_modalities):
        """Learn from multimodal data"""
        # Simple gradient update (placeholder)
        pass
    
    def reset(self):
        """Reset network"""
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.weights2 = np.random.randn(self.hidden_size, 1) * 0.1
        self.bias1 = np.zeros(self.hidden_size)
        self.bias2 = np.zeros(1)

# Initialize session state
if 'ai_system' not in st.session_state:
    st.session_state.ai_system = AdvancedMultimodalAI()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'selected_modalities' not in st.session_state:
    st.session_state.selected_modalities = ['Text']

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🧠 Ultimate Advanced Multimodal AI</h1>
        <p>Next-generation self-learning AI with multimodal analysis and real-time adaptation</p>
        <p style="font-size: 0.9em; opacity: 0.8;">Version 3.0 - Fixed & Enhanced</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Status
    stats = st.session_state.ai_system.get_learning_stats()
    st.markdown(f"""
    <div class="ai-status">
        <h3>🤖 AI Status: {stats['ai_status']} | Performance: {stats['performance_score']:.1f}%</h3>
        <p>Learning: {'✅ Active' if stats['learning_enabled'] else '❌ Disabled'} | 
        Adaptation: {'✅ Active' if stats['adaptation_enabled'] else '❌ Disabled'} | 
        Feedback: {stats['total_feedback']} events</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content layout
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.markdown("### 🔍 Multimodal Analysis Center")
        
        # Modality Selection with better UI
        st.markdown("#### 🎯 Select Analysis Modalities")
        
        col_mod1, col_mod2, col_mod3 = st.columns(3)
        with col_mod1:
            text_selected = st.checkbox("📝 Text", value=True, key="text_mod")
        with col_mod2:
            audio_selected = st.checkbox("🎵 Audio", value=False, key="audio_mod")
        with col_mod3:
            image_selected = st.checkbox("🖼️ Image", value=False, key="image_mod")
        
        selected_modalities = []
        if text_selected:
            selected_modalities.append("Text")
        if audio_selected:
            selected_modalities.append("Audio")
        if image_selected:
            selected_modalities.append("Image")
        
        st.session_state.selected_modalities = selected_modalities
        
        # Show selected combination
        if selected_modalities:
            combo_text = " + ".join(selected_modalities)
            st.info(f"🎯 Selected: **{combo_text}**")
        
        # Input fields
        st.markdown("#### 📊 Input Data")
        
        # Text input (always visible but optional if other modalities selected)
        text_input = st.text_area(
            "Text Input:" if len(selected_modalities) > 1 else "Enter text to analyze:",
            placeholder="Type or paste text here for crisis detection analysis...",
            height=120,
            key="text_input"
        )
        
        # Audio input (conditional)
        audio_file = None
        if "Audio" in selected_modalities:
            audio_file = st.file_uploader("Upload Audio File:", type=['wav', 'mp3', 'm4a'], key="audio_input")
            if audio_file:
                st.audio(audio_file)
        
        # Image input (conditional)
        image_file = None
        if "Image" in selected_modalities:
            image_file = st.file_uploader("Upload Image:", type=['jpg', 'jpeg', 'png'], key="image_input")
            if image_file:
                st.image(image_file, width=300)
        
        # Analyze button
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if text_input.strip() or audio_file or image_file:
                with st.spinner("Analyzing..."):
                    # Analyze multimodal
                    pred, prob, conf, mod_preds = st.session_state.ai_system.predict_multimodal(
                        text_input, audio_file, image_file, selected_modalities
                    )
                    
                    # Store in history
                    st.session_state.analysis_history.append({
                        'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                        'prediction': pred,
                        'probability': prob,
                        'confidence': conf,
                        'modality_predictions': mod_preds,
                        'modalities_used': selected_modalities,
                        'timestamp': datetime.now()
                    })
                    
                    # Display result
                    if pred == 1:
                        st.markdown(f"""
                        <div class="analysis-card">
                            <h3>🚨 CRISIS DETECTED</h3>
                            <p><strong>Confidence:</strong> {conf:.1%}</p>
                            <p><strong>Probability:</strong> {prob:.1%}</p>
                            <p><strong>⚠️ This indicates a potential mental health crisis.</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-card">
                            <h3>✅ SAFE</h3>
                            <p><strong>Confidence:</strong> {conf:.1%}</p>
                            <p><strong>Probability:</strong> {prob:.1%}</p>
                            <p><strong>No immediate crisis detected.</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show modality predictions
                    st.markdown("#### 🎯 Modality Analysis")
                    
                    cols = st.columns(len(selected_modalities) + 1)
                    for idx, modality in enumerate(selected_modalities):
                        with cols[idx]:
                            mod_key = modality.lower()
                            pred_val = mod_preds.get(mod_key, 0.5)
                            st.metric(
                                modality,
                                f"{pred_val:.1%}",
                                delta=f"{(pred_val - 0.5) * 100:.1f}%",
                                delta_color="inverse" if pred_val > 0.5 else "normal"
                            )
                    
                    # Ensemble prediction
                    with cols[-1]:
                        st.metric(
                            "Ensemble",
                            f"{mod_preds['ensemble']:.1%}",
                            delta=f"{(mod_preds['ensemble'] - 0.5) * 100:.1f}%",
                            delta_color="inverse" if mod_preds['ensemble'] > 0.5 else "normal"
                        )
                    
                    # Feedback section
                    st.markdown("#### 💬 Feedback")
                    col_fb1, col_fb2 = st.columns(2)
                    
                    with col_fb1:
                        if st.button("✅ Correct", use_container_width=True):
                            st.session_state.ai_system.learn_from_feedback(
                                text_input, audio_file, image_file, pred, selected_modalities
                            )
                            st.success("✅ Thank you! AI is learning from your feedback.")
                            # Refresh UI so stats update immediately
                            st.rerun()
                    
                    with col_fb2:
                        if st.button("❌ Incorrect", use_container_width=True):
                            correct_label = 1 if pred == 0 else 0
                            st.session_state.ai_system.learn_from_feedback(
                                text_input, audio_file, image_file, correct_label, selected_modalities
                            )
                            st.success("📚 Thank you! AI is learning from this correction.")
                            # Refresh UI so stats update immediately
                            st.rerun()
            
            else:
                st.warning("Please provide at least one input (text, audio, or image).")
    
    # Learning Statistics column removed per user request
    
    with col3:
        st.markdown("### 🧠 AI Insights")
        
        # Feature importance
        if 'text_weights' in stats:
            st.markdown("#### 🔍 Text Feature Importance")
            
            # Sort features by absolute weight
            sorted_features = sorted(
                stats['text_weights'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 features
            
            for feature, weight in sorted_features:
                color = "🟢" if weight > 0 else "🔴"
                st.text(f"{color} {feature}: {weight:+.3f}")
        
        # Learning curve
        if stats['accuracy_history']:
            st.markdown("#### 📈 Learning Progress")
            accuracy_data = stats['accuracy_history']
            if len(accuracy_data) > 1:
                trend = "📈" if accuracy_data[-1] > accuracy_data[0] else "📉"
                st.text(f"{trend} Trend: {(accuracy_data[-1] - accuracy_data[0]) * 100:+.1f}%")
        
        # Reset options
        st.markdown("#### 🔧 Controls")
        if st.button("🔄 Reset Learning", use_container_width=True):
            st.session_state.ai_system.reset_learning()
            st.session_state.analysis_history = []
            st.success("✅ AI learning reset successfully!")
            st.rerun()
    
    # Test section
    with st.expander("🧪 Quick Test Panel", expanded=False):
        st.markdown("### 🧪 Test Sample Texts")
        
        test_categories = {
            "Crisis": [
                "I want to kill myself tonight",
                "I am going to end my life",
                "I have a plan to commit suicide"
            ],
            "Safe": [
                "I am feeling better after therapy",
                "I am working on my mental health",
                "I am getting help and support"
            ],
            "Edge Cases": [
                "I am having a really bad day",
                "I feel hopeless sometimes",
                "Life is hard but I'm trying"
            ]
        }
        
        category = st.selectbox("Select test category:", list(test_categories.keys()))
        
        if st.button("🧪 Run Tests"):
            st.markdown(f"**Testing {category} Cases:**")
            
            for text in test_categories[category]:
                pred, prob, conf, _ = st.session_state.ai_system.predict_multimodal(
                    text, None, None, ['Text']
                )
                status = "🚨 CRISIS" if pred == 1 else "✅ SAFE"
                st.text(f"{status}: '{text}' (Conf: {conf:.1%})")
    
    # History section
    if st.session_state.analysis_history:
        with st.expander("📋 Recent Analysis History", expanded=False):
            for idx, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
                status_emoji = "🚨" if entry['prediction'] == 1 else "✅"
                modalities_text = " + ".join(entry['modalities_used'])
                
                st.markdown(f"""
                **{status_emoji} Analysis #{len(st.session_state.analysis_history) - idx}**
                - **Modalities:** {modalities_text}
                - **Result:** {'CRISIS' if entry['prediction'] == 1 else 'SAFE'} ({entry['confidence']:.1%})
                - **Time:** {entry['timestamp'].strftime('%H:%M:%S')}
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem; opacity: 0.8;">
        <h4>🧠 Ultimate Advanced Multimodal AI</h4>
        <p>Powered by advanced machine learning with real-time adaptation</p>
        <p><strong>⚠️ Important:</strong> This tool is for educational purposes only. Always seek professional help for mental health concerns.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
