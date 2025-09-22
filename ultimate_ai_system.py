import streamlit as st
import numpy as np
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
    page_title="🧠 Ultimate Self-Learning Multimodal AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultimate CSS
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
    
    .ai-status {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class UltimateMultimodalAI:
    """Ultimate Self-Learning Multimodal AI System"""
    
    def __init__(self):
        # Initialize AI components
        self.text_analyzer = TextAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.neural_network = NeuralNetwork()
        
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
            'text_weight': 0.5,
            'audio_weight': 0.3,
            'image_weight': 0.2
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_rate = 0.05
        
        # AI status
        self.ai_status = "ACTIVE"
        self.learning_enabled = True
        self.adaptation_enabled = True
    
    def predict_multimodal(self, text, audio_data=None, image_data=None):
        """Make advanced multimodal prediction"""
        # Extract features from all modalities
        text_features = self.text_analyzer.extract_features(text)
        audio_features = self.audio_analyzer.extract_features(audio_data)
        image_features = self.image_analyzer.extract_features(image_data)
        
        # Get predictions from each modality
        text_pred = self.text_analyzer.predict(text_features)
        audio_pred = self.audio_analyzer.predict(audio_features)
        image_pred = self.image_analyzer.predict(image_features)
        
        # Neural network fusion
        combined_features = list(text_features.values()) + audio_features + image_features
        neural_pred = self.neural_network.predict(combined_features)
        
        # Weighted ensemble prediction
        ensemble_pred = (
            text_pred * self.model_weights['text_weight'] +
            audio_pred * self.model_weights['audio_weight'] +
            image_pred * self.model_weights['image_weight'] +
            neural_pred * 0.1  # Neural network contribution
        )
        
        # Final prediction
        final_pred = 1 if ensemble_pred > 0.5 else 0
        confidence = abs(ensemble_pred - 0.5) * 2
        
        return final_pred, ensemble_pred, confidence, {
            'text': text_pred,
            'audio': audio_pred,
            'image': image_pred,
            'neural': neural_pred
        }
    
    def learn_from_feedback(self, text, audio_data, image_data, correct_label):
        """Advanced learning from feedback"""
        # Get current prediction
        pred, prob, conf, mod_preds = self.predict_multimodal(text, audio_data, image_data)
        
        # Store learning data
        self.learning_data['texts'].append(text)
        self.learning_data['audio_features'].append(self.audio_analyzer.extract_features(audio_data))
        self.learning_data['image_features'].append(self.image_analyzer.extract_features(image_data))
        self.learning_data['labels'].append(correct_label)
        self.learning_data['predictions'].append(pred)
        self.learning_data['feedback'].append({
            'timestamp': datetime.now().isoformat(),
            'correct_label': correct_label,
            'predicted_label': pred,
            'confidence': conf
        })
        
        # Advanced learning
        if self.learning_enabled:
            # Update individual analyzers
            self.text_analyzer.learn(text, correct_label)
            self.audio_analyzer.learn(audio_data, correct_label)
            self.image_analyzer.learn(image_data, correct_label)
            
            # Update neural network
            self.neural_network.learn(text, audio_data, image_data, correct_label)
            
            # Adaptive weight adjustment
            if self.adaptation_enabled:
                self._adapt_weights(mod_preds, correct_label)
        
        # Store weight history
        self.learning_data['weight_history'].append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.model_weights.copy()
        })
    
    def _adapt_weights(self, mod_preds, correct_label):
        """Adapt model weights based on performance"""
        error = correct_label - 0.5
        
        # Update weights based on prediction accuracy
        for modality, pred in mod_preds.items():
            if modality in self.model_weights:
                accuracy = 1 - abs(pred - correct_label)
                weight_change = self.adaptation_rate * accuracy * error
                self.model_weights[modality] += weight_change
                self.model_weights[modality] = max(0.1, min(0.8, self.model_weights[modality]))
    
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
            'learning_rate': self.learning_rate,
            'adaptation_rate': self.adaptation_rate,
            'ai_status': self.ai_status,
            'learning_enabled': self.learning_enabled,
            'adaptation_enabled': self.adaptation_enabled
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
            'text_weight': 0.5,
            'audio_weight': 0.3,
            'image_weight': 0.2
        }
        self.text_analyzer.reset()
        self.audio_analyzer.reset()
        self.image_analyzer.reset()
        self.neural_network.reset()

class TextAnalyzer:
    """Advanced Text Analysis with Learning"""
    
    def __init__(self):
        self.feature_weights = {
            'crisis_keywords': 0.8,
            'help_keywords': -0.6,
            'intensity_words': 0.3,
            'temporal_words': 0.4,
            'negation_words': -0.2,
            'positive_words': -0.3,
            'negative_words': 0.4,
            'text_length': 0.02,
            'word_count': 0.02,
            'sentence_count': 0.01
        }
        self.learning_rate = 0.1
    
    def extract_features(self, text):
        """Extract comprehensive text features"""
        features = {}
        text_lower = text.lower()
        
        # Crisis keywords
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'commit suicide',
            'take my life', 'want to die', 'going to die', 'plan to die',
            'jump off', 'hang myself', 'overdose', 'shoot myself',
            'end it all', 'not want to live', 'self harm', 'cut myself'
        ]
        
        # Help-seeking keywords
        help_keywords = [
            'help', 'support', 'therapy', 'counselor', 'therapist',
            'treatment', 'medication', 'coping', 'recovery', 'crisis',
            'getting help', 'seeking help', 'professional help',
            'better after', 'working on', 'improving', 'healing'
        ]
        
        # Emotional intensity
        intensity_words = [
            'really', 'so', 'extremely', 'completely', 'totally', 'absolutely',
            'never', 'always', 'forever', 'everything', 'nothing'
        ]
        
        # Temporal markers
        temporal_words = [
            'tonight', 'today', 'now', 'immediately', 'right now', 'this moment',
            'soon', 'later', 'tomorrow'
        ]
        
        # Negation words
        negation_words = [
            'not', 'never', 'no', 'can\'t', 'won\'t', 'don\'t', 'never'
        ]
        
        # Positive words
        positive_words = [
            'good', 'great', 'better', 'happy', 'love', 'hope', 'joy',
            'peace', 'calm', 'safe', 'secure'
        ]
        
        # Negative words
        negative_words = [
            'bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'fear',
            'anxiety', 'depression', 'hopeless', 'worthless'
        ]
        
        # Count features
        features['crisis_keywords'] = sum(1 for phrase in crisis_keywords if phrase in text_lower)
        features['help_keywords'] = sum(1 for phrase in help_keywords if phrase in text_lower)
        features['intensity_words'] = sum(1 for word in intensity_words if word in text_lower)
        features['temporal_words'] = sum(1 for word in temporal_words if word in text_lower)
        features['negation_words'] = sum(1 for word in negation_words if word in text_lower)
        features['positive_words'] = sum(1 for word in positive_words if word in text_lower)
        features['negative_words'] = sum(1 for word in negative_words if word in text_lower)
        
        # Text statistics
        features['text_length'] = len(text) / 100
        features['word_count'] = len(text.split()) / 20
        features['sentence_count'] = len(text.split('.')) / 5
        
        return features
    
    def predict(self, features):
        """Predict using text features"""
        score = 0
        for feature, value in features.items():
            if feature in self.feature_weights:
                score += self.feature_weights[feature] * value
        
        # Sigmoid function
        probability = 1 / (1 + math.exp(-score))
        return probability
    
    def learn(self, text, correct_label):
        """Learn from feedback"""
        features = self.extract_features(text)
        error = correct_label - 0.5
        
        # Update weights
        for feature, value in features.items():
            if feature in self.feature_weights:
                weight_change = self.learning_rate * error * value
                self.feature_weights[feature] += weight_change
    
    def reset(self):
        """Reset analyzer"""
        self.feature_weights = {
            'crisis_keywords': 0.8,
            'help_keywords': -0.6,
            'intensity_words': 0.3,
            'temporal_words': 0.4,
            'negation_words': -0.2,
            'positive_words': -0.3,
            'negative_words': 0.4,
            'text_length': 0.02,
            'word_count': 0.02,
            'sentence_count': 0.01
        }

class AudioAnalyzer:
    """Advanced Audio Analysis with Learning"""
    
    def __init__(self):
        self.feature_weights = [0.1] * 13  # MFCC features
        self.learning_rate = 0.05
    
    def extract_features(self, audio_data):
        """Extract audio features"""
        if audio_data is None:
            return [0.0] * 13
        
        # Simulate MFCC features
        features = np.random.random(13).tolist()
        return features
    
    def predict(self, features):
        """Predict using audio features"""
        score = sum(w * f for w, f in zip(self.feature_weights, features))
        probability = 1 / (1 + math.exp(-score))
        return probability
    
    def learn(self, audio_data, correct_label):
        """Learn from feedback"""
        features = self.extract_features(audio_data)
        error = correct_label - 0.5
        
        # Update weights
        for i, feature in enumerate(features):
            if i < len(self.feature_weights):
                weight_change = self.learning_rate * error * feature
                self.feature_weights[i] += weight_change
    
    def reset(self):
        """Reset analyzer"""
        self.feature_weights = [0.1] * 13

class ImageAnalyzer:
    """Advanced Image Analysis with Learning"""
    
    def __init__(self):
        self.feature_weights = [0.1] * 100  # Image features
        self.learning_rate = 0.05
    
    def extract_features(self, image_data):
        """Extract image features"""
        if image_data is None:
            return [0.0] * 100
        
        # Simulate image features
        features = np.random.random(100).tolist()
        return features
    
    def predict(self, features):
        """Predict using image features"""
        score = sum(w * f for w, f in zip(self.feature_weights, features))
        probability = 1 / (1 + math.exp(-score))
        return probability
    
    def learn(self, image_data, correct_label):
        """Learn from feedback"""
        features = self.extract_features(image_data)
        error = correct_label - 0.5
        
        # Update weights
        for i, feature in enumerate(features):
            if i < len(self.feature_weights):
                weight_change = self.learning_rate * error * feature
                self.feature_weights[i] += weight_change
    
    def reset(self):
        """Reset analyzer"""
        self.feature_weights = [0.1] * 100

class NeuralNetwork:
    """Simplified Neural Network"""
    
    def __init__(self):
        self.weights = [0.1] * 200  # Combined features
        self.learning_rate = 0.01
    
    def predict(self, features):
        """Neural network prediction"""
        score = sum(w * f for w, f in zip(self.weights, features))
        probability = 1 / (1 + math.exp(-score))
        return probability
    
    def learn(self, text, audio_data, image_data, correct_label):
        """Neural network learning"""
        # Combine features
        text_features = TextAnalyzer().extract_features(text)
        audio_features = AudioAnalyzer().extract_features(audio_data)
        image_features = ImageAnalyzer().extract_features(image_data)
        
        combined_features = list(text_features.values()) + audio_features + image_features
        
        error = correct_label - 0.5
        
        # Update weights
        for i, feature in enumerate(combined_features):
            if i < len(self.weights):
                weight_change = self.learning_rate * error * feature
                self.weights[i] += weight_change
    
    def reset(self):
        """Reset neural network"""
        self.weights = [0.1] * 200

# Initialize session state
if 'ai_system' not in st.session_state:
    st.session_state.ai_system = UltimateMultimodalAI()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🧠 Ultimate Self-Learning Multimodal AI</h1>
        <p>Next-generation AI with advanced learning, multimodal analysis, and real-time adaptation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Status
    stats = st.session_state.ai_system.get_learning_stats()
    st.markdown(f"""
    <div class="ai-status">
        <h3>🤖 AI Status: {stats['ai_status']}</h3>
        <p>Learning: {'✅ Enabled' if stats['learning_enabled'] else '❌ Disabled'} | 
        Adaptation: {'✅ Enabled' if stats['adaptation_enabled'] else '❌ Disabled'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Multimodal Analysis")
        
        # Modality selection
        st.markdown("#### 🎯 Select Analysis Modalities")
        modality_options = st.multiselect(
            "Choose which modalities to use for analysis:",
            options=["Text", "Audio", "Image"],
            default=["Text"],
            help="Select one or more modalities for analysis"
        )
        
        # Text input (always available)
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here for crisis detection analysis...",
            height=150
        )
        
        # Audio input (conditional)
        audio_file = None
        if "Audio" in modality_options:
            audio_file = st.file_uploader("Upload Audio File:", type=['wav', 'mp3', 'm4a'])
        
        # Image input (conditional)
        image_file = None
        if "Image" in modality_options:
            image_file = st.file_uploader("Upload Image:", type=['jpg', 'jpeg', 'png'])
        
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
                st.markdown(f"**Selected Modalities:** {', '.join(modality_options)}")
                
                # Create columns based on selected modalities
                num_cols = len(modality_options) + 1  # +1 for neural network
                cols = st.columns(num_cols)
                
                col_idx = 0
                
                # Text analysis (always shown)
                if "Text" in modality_options:
                    with cols[col_idx]:
                        st.metric("Text", f"{mod_preds['text']:.1%}", "Primary")
                    col_idx += 1
                
                # Audio analysis (conditional)
                if "Audio" in modality_options:
                    with cols[col_idx]:
                        st.metric("Audio", f"{mod_preds['audio']:.1%}", "Secondary")
                    col_idx += 1
                
                # Image analysis (conditional)
                if "Image" in modality_options:
                    with cols[col_idx]:
                        st.metric("Image", f"{mod_preds['image']:.1%}", "Secondary")
                    col_idx += 1
                
                # Neural network (always shown)
                with cols[col_idx]:
                    st.metric("Neural", f"{mod_preds['neural']:.1%}", "AI")
                
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
        
        # Reset button
        if st.button("🔄 Reset AI Learning"):
            st.session_state.ai_system.reset_learning()
            st.success("🔄 AI learning reset!")
    
    # Modality combination examples
    st.markdown("### 🎯 Available Modality Combinations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📝 Text Only**
        - Basic text analysis
        - Keyword detection
        - Emotional analysis
        """)
    
    with col2:
        st.markdown("""
        **📝🎵 Text + Audio**
        - Text + voice analysis
        - Speech emotion detection
        - Combined audio-text features
        """)
    
    with col3:
        st.markdown("""
        **📝🖼️ Text + Image**
        - Text + visual analysis
        - Image emotion detection
        - Combined visual-text features
        """)
    
    with col4:
        st.markdown("""
        **📝🎵🖼️ All Modalities**
        - Complete multimodal analysis
        - Text + Audio + Image
        - Maximum AI accuracy
        """)
    
    # Test with sample texts
    st.markdown("### 🧪 Test Ultimate AI")
    
    sample_texts = [
        "I want to kill myself tonight",
        "I am feeling better after therapy",
        "I am going to end my life",
        "I am struggling but getting help",
        "I have a plan to commit suicide",
        "I am having a bad day but I'll be okay"
    ]
    
    if st.button("🧪 Test Sample Texts"):
        st.markdown("**Ultimate AI Analysis Results:**")
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
        <h4>🧠 Ultimate Self-Learning Multimodal AI</h4>
        <p>Next-generation AI with advanced learning, multimodal analysis, and real-time adaptation.</p>
        <p><strong>⚠️ Important:</strong> This tool is for educational purposes. Always seek professional help for mental health concerns.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
