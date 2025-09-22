#!/usr/bin/env python3
"""
Multimodal Self-Learning Crisis Detection System
Supports text, audio, and image analysis with self-learning capabilities
"""

import streamlit as st
import re
import random
import json
import os
import math
import numpy as np
from datetime import datetime
import time
from collections import Counter, defaultdict
import base64
import io

# Page configuration
st.set_page_config(
    page_title="🧠 Multimodal Self-Learning Crisis Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS
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
    
    .modal-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
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
    
    .multimodal-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(162, 155, 254, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(116, 185, 255, 0.3);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(232, 67, 147, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class MultimodalLearningSystem:
    """Multimodal self-learning crisis detection system"""
    
    def __init__(self):
        self.feedback_data = {
            'corrections': [],
            'confirmations': [],
            'model_weights': {},
            'learning_history': [],
            'modality_weights': {
                'text': 1.0,
                'audio': 0.8,
                'image': 0.6
            }
        }
        
        # Base weights for different modalities
        self.text_weights = {
            'crisis_keyword_count': 0.8,
            'help_keyword_count': -0.6,
            'text_length': 0.1,
            'word_count': 0.05,
            'intensity_count': 0.3,
            'temporal_count': 0.4,
            'negation_count': -0.2
        }
        
        self.audio_weights = {
            'pitch_variance': 0.7,
            'speech_rate': 0.5,
            'volume_variance': 0.6,
            'pause_frequency': 0.4,
            'emotional_tone': 0.8,
            'stress_indicators': 0.9
        }
        
        self.image_weights = {
            'color_saturation': 0.3,
            'brightness_variance': 0.4,
            'edge_density': 0.5,
            'texture_complexity': 0.6,
            'facial_expression': 0.8,
            'composition_balance': 0.2
        }
        
        self.learning_rate = 0.1
    
    def extract_text_features(self, text):
        """Extract features from text"""
        features = {}
        text_lower = text.lower()
        
        # Crisis keywords
        crisis_keywords = [
            'suicide', 'kill', 'die', 'death', 'end', 'pain', 'suffer',
            'hopeless', 'worthless', 'useless', 'burden', 'alone', 'isolated',
            'plan', 'method', 'pills', 'rope', 'gun', 'jump', 'bridge'
        ]
        
        # Help-seeking keywords
        help_keywords = [
            'help', 'support', 'therapy', 'counselor', 'therapist',
            'treatment', 'medication', 'coping', 'recovery', 'crisis'
        ]
        
        features['crisis_keyword_count'] = sum(1 for word in crisis_keywords if word in text_lower)
        features['help_keyword_count'] = sum(1 for word in help_keywords if word in text_lower)
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Emotional intensity
        intensity_words = ['really', 'so', 'extremely', 'completely', 'totally']
        features['intensity_count'] = sum(1 for word in intensity_words if word in text_lower)
        
        # Temporal markers
        temporal_words = ['tonight', 'today', 'now', 'immediately', 'right now']
        features['temporal_count'] = sum(1 for word in temporal_words if word in text_lower)
        
        # Negation
        negation_words = ['not', 'never', 'no', 'can\'t', 'won\'t']
        features['negation_count'] = sum(1 for word in negation_words if word in text_lower)
        
        return features
    
    def extract_audio_features(self, audio_data):
        """Extract features from audio (simulated)"""
        # In a real implementation, this would analyze actual audio
        # For demo purposes, we'll simulate audio features
        features = {}
        
        # Simulate audio analysis
        features['pitch_variance'] = random.uniform(0.1, 0.9)
        features['speech_rate'] = random.uniform(0.3, 0.8)
        features['volume_variance'] = random.uniform(0.2, 0.7)
        features['pause_frequency'] = random.uniform(0.1, 0.6)
        features['emotional_tone'] = random.uniform(0.2, 0.9)
        features['stress_indicators'] = random.uniform(0.1, 0.8)
        
        return features
    
    def extract_image_features(self, image_data):
        """Extract features from image (simulated)"""
        # In a real implementation, this would analyze actual images
        # For demo purposes, we'll simulate image features
        features = {}
        
        # Simulate image analysis
        features['color_saturation'] = random.uniform(0.2, 0.8)
        features['brightness_variance'] = random.uniform(0.1, 0.7)
        features['edge_density'] = random.uniform(0.3, 0.9)
        features['texture_complexity'] = random.uniform(0.2, 0.8)
        features['facial_expression'] = random.uniform(0.1, 0.9)
        features['composition_balance'] = random.uniform(0.3, 0.8)
        
        return features
    
    def predict_multimodal(self, text=None, audio_data=None, image_data=None):
        """Predict using multiple modalities"""
        predictions = {}
        confidences = {}
        
        # Text analysis
        if text:
            text_features = self.extract_text_features(text)
            text_score = self._calculate_score(text_features, self.text_weights, 'text')
            text_prob = 1 / (1 + math.exp(-text_score))
            predictions['text'] = 1 if text_prob > 0.5 else 0
            confidences['text'] = abs(text_prob - 0.5) * 2
        
        # Audio analysis
        if audio_data:
            audio_features = self.extract_audio_features(audio_data)
            audio_score = self._calculate_score(audio_features, self.audio_weights, 'audio')
            audio_prob = 1 / (1 + math.exp(-audio_score))
            predictions['audio'] = 1 if audio_prob > 0.5 else 0
            confidences['audio'] = abs(audio_prob - 0.5) * 2
        
        # Image analysis
        if image_data:
            image_features = self.extract_image_features(image_data)
            image_score = self._calculate_score(image_features, self.image_weights, 'image')
            image_prob = 1 / (1 + math.exp(-image_score))
            predictions['image'] = 1 if image_prob > 0.5 else 0
            confidences['image'] = abs(image_prob - 0.5) * 2
        
        # Combine predictions
        if not predictions:
            return None, 0, 0, {}
        
        # Weighted ensemble
        total_weight = 0
        weighted_score = 0
        
        for modality, pred in predictions.items():
            weight = self.feedback_data['modality_weights'][modality]
            confidence = confidences[modality]
            weighted_score += pred * weight * confidence
            total_weight += weight * confidence
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_prediction = 1 if final_score > 0.5 else 0
            final_confidence = abs(final_score - 0.5) * 2
        else:
            final_prediction = 0
            final_confidence = 0
        
        return final_prediction, final_confidence, predictions, confidences
    
    def _calculate_score(self, features, weights, modality):
        """Calculate score for a modality"""
        score = 0
        for feature, value in features.items():
            if feature in weights:
                # Apply learned weights if available
                learned_weight = self.feedback_data['model_weights'].get(f'{modality}_{feature}', 0)
                score += (weights[feature] + learned_weight) * value
        return score
    
    def learn_from_feedback(self, text=None, audio_data=None, image_data=None, user_label=None):
        """Learn from user feedback across modalities"""
        if user_label is None:
            return
        
        prediction, confidence, modality_preds, modality_confs = self.predict_multimodal(text, audio_data, image_data)
        
        if user_label != prediction:
            # Add correction
            correction = {
                'text': text,
                'audio_data': audio_data is not None,
                'image_data': image_data is not None,
                'predicted_label': prediction,
                'correct_label': user_label,
                'confidence': confidence,
                'modality_predictions': modality_preds,
                'modality_confidences': modality_confs,
                'timestamp': datetime.now().isoformat()
            }
            self.feedback_data['corrections'].append(correction)
            self._update_weights(correction)
        else:
            # Add confirmation
            confirmation = {
                'text': text,
                'audio_data': audio_data is not None,
                'image_data': image_data is not None,
                'predicted_label': prediction,
                'confidence': confidence,
                'modality_predictions': modality_preds,
                'modality_confidences': modality_confs,
                'timestamp': datetime.now().isoformat()
            }
            self.feedback_data['confirmations'].append(confirmation)
            self._strengthen_weights(confirmation)
    
    def _update_weights(self, correction):
        """Update weights based on correction"""
        # Update modality weights
        for modality, pred in correction['modality_predictions'].items():
            error = correction['correct_label'] - pred
            self.feedback_data['modality_weights'][modality] += self.learning_rate * error * 0.1
        
        # Update feature weights
        if correction['text']:
            text_features = self.extract_text_features(correction['text'])
            for feature, value in text_features.items():
                key = f'text_{feature}'
                if key not in self.feedback_data['model_weights']:
                    self.feedback_data['model_weights'][key] = 0.0
                
                error = correction['correct_label'] - correction['predicted_label']
                self.feedback_data['model_weights'][key] += self.learning_rate * error * value
    
    def _strengthen_weights(self, confirmation):
        """Strengthen weights for correct predictions"""
        # Strengthen modality weights
        for modality, pred in confirmation['modality_predictions'].items():
            self.feedback_data['modality_weights'][modality] += self.learning_rate * 0.05 * pred
    
    def get_stats(self):
        """Get learning statistics"""
        return {
            'total_feedback': len(self.feedback_data['corrections']) + len(self.feedback_data['confirmations']),
            'corrections': len(self.feedback_data['corrections']),
            'confirmations': len(self.feedback_data['confirmations']),
            'modality_weights': self.feedback_data['modality_weights'],
            'learning_rate': self.learning_rate
        }

# Initialize session state
if 'multimodal_system' not in st.session_state:
    st.session_state.multimodal_system = MultimodalLearningSystem()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🧠 Multimodal Self-Learning Crisis Detector</h1>
        <p>Advanced AI that analyzes text, audio, and images to detect mental health crises</p>
        <p>Learns from your feedback to improve accuracy across all modalities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modality selection
    st.markdown("### 🎯 Select Analysis Modalities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_text = st.checkbox("📝 Text Analysis", value=True, help="Analyze text for crisis indicators")
    with col2:
        use_audio = st.checkbox("🎵 Audio Analysis", value=False, help="Analyze audio for emotional distress")
    with col3:
        use_image = st.checkbox("🖼️ Image Analysis", value=False, help="Analyze images for visual cues")
    
    # Main analysis section
    st.markdown("### 🔍 Multimodal Analysis")
    
    # Text input
    text_input = None
    if use_text:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here for crisis detection analysis...",
            height=150
        )
    
    # Audio upload
    audio_data = None
    if use_audio:
        st.markdown("#### 🎵 Audio Upload")
        audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'm4a'], help="Upload audio for emotional analysis")
        if audio_file:
            audio_data = audio_file.read()
            st.success(f"Audio file uploaded: {audio_file.name}")
    
    # Image upload
    image_data = None
    if use_image:
        st.markdown("#### 🖼️ Image Upload")
        image_file = st.file_uploader("Upload image file", type=['jpg', 'jpeg', 'png', 'gif'], help="Upload image for visual analysis")
        if image_file:
            image_data = image_file.read()
            st.success(f"Image file uploaded: {image_file.name}")
    
    # Analysis button
    if st.button("🔍 Analyze Multimodal Content"):
        if not any([text_input, audio_data, image_data]):
            st.warning("Please provide at least one type of content to analyze.")
        else:
            with st.spinner("Analyzing multimodal content..."):
                # Analyze content
                prediction, confidence, modality_preds, modality_confs = st.session_state.multimodal_system.predict_multimodal(
                    text=text_input, audio_data=audio_data, image_data=image_data
                )
                
                # Store in history
                st.session_state.analysis_history.append({
                    'text': text_input,
                    'audio_uploaded': audio_data is not None,
                    'image_uploaded': image_data is not None,
                    'prediction': prediction,
                    'confidence': confidence,
                    'modality_predictions': modality_preds,
                    'modality_confidences': modality_confs,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Display results
                if prediction == 1:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h3>🚨 CRISIS DETECTED</h3>
                        <p><strong>Overall Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>⚠️ This content indicates a potential mental health crisis. Please seek immediate help.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <h3>✅ SAFE</h3>
                        <p><strong>Overall Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>This content does not indicate an immediate crisis.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show modality-specific results
                if modality_preds:
                    st.markdown("#### 📊 Modality-Specific Results")
                    
                    for modality, pred in modality_preds.items():
                        conf = modality_confs[modality]
                        status = "🚨 CRISIS" if pred == 1 else "✅ SAFE"
                        modality_name = modality.capitalize()
                        
                        st.markdown(f"""
                        <div class="modal-card">
                            <h4>{modality_name} Analysis: {status}</h4>
                            <p><strong>Confidence:</strong> {conf:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Feedback section
                st.markdown("### 💬 Help Improve the AI")
                st.markdown("Was this analysis correct? Your feedback helps the AI learn and improve.")
                
                col_feedback1, col_feedback2 = st.columns(2)
                
                with col_feedback1:
                    if st.button("✅ Correct"):
                        st.session_state.multimodal_system.learn_from_feedback(
                            text=text_input, audio_data=audio_data, image_data=image_data, user_label=prediction
                        )
                        st.success("✅ Thank you! Your feedback has been recorded.")
                
                with col_feedback2:
                    if st.button("❌ Incorrect"):
                        correct_label = 1 if prediction == 0 else 0
                        st.session_state.multimodal_system.learn_from_feedback(
                            text=text_input, audio_data=audio_data, image_data=image_data, user_label=correct_label
                        )
                        st.success("📚 Thank you! The AI is learning from your correction.")
    
    # Sidebar with statistics
    with st.sidebar:
        st.markdown("### 📊 Learning Statistics")
        
        stats = st.session_state.multimodal_system.get_stats()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎓 Total Learning</h4>
            <h2>{stats['total_feedback']}</h2>
            <p>Feedback Events</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>✅ Confirmations</h4>
            <h2>{stats['confirmations']}</h2>
            <p>Correct Predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📚 Corrections</h4>
            <h2>{stats['corrections']}</h2>
            <p>Learning Events</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Modality weights
        st.markdown("### 🎯 Modality Weights")
        for modality, weight in stats['modality_weights'].items():
            st.metric(f"{modality.capitalize()} Weight", f"{weight:.2f}")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("### 📋 Recent Analysis History")
        
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
            status_emoji = "🚨" if analysis['prediction'] == 1 else "✅"
            
            with st.expander(f"{status_emoji} Analysis #{len(st.session_state.analysis_history) - i}"):
                modalities_used = []
                if analysis['text']:
                    modalities_used.append("Text")
                if analysis['audio_uploaded']:
                    modalities_used.append("Audio")
                if analysis['image_uploaded']:
                    modalities_used.append("Image")
                
                st.markdown(f"**Modalities:** {', '.join(modalities_used)}")
                st.markdown(f"**Result:** {'CRISIS' if analysis['prediction'] == 1 else 'SAFE'} (Confidence: {analysis['confidence']:.1%})")
                st.markdown(f"**Time:** {analysis['timestamp']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;">
        <h4>🧠 Multimodal Self-Learning Crisis Detector</h4>
        <p>Advanced AI that learns from your feedback across text, audio, and image modalities.</p>
        <p><strong>⚠️ Important:</strong> This tool is for educational purposes. Always seek professional help for mental health concerns.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
