import streamlit as st
import re
import random
import json
import os
import math
from datetime import datetime
import time
from collections import Counter, defaultdict

# Page configuration
st.set_page_config(
    page_title="🧠 Balanced Self-Learning Crisis Detector",
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
</style>
""", unsafe_allow_html=True)

class BalancedLearningSystem:
    """Balanced self-learning crisis detection system"""
    
    def __init__(self):
        self.feedback_data = {
            'corrections': [],
            'confirmations': [],
            'model_weights': {},
            'learning_history': []
        }
        
        # BALANCED weights that work properly
        self.base_weights = {
            'crisis_keyword_count': 1.2,      # Strong indicator
            'help_keyword_count': -0.8,       # Strong negative indicator
            'text_length': 0.02,              # Weak indicator
            'word_count': 0.02,               # Weak indicator
            'intensity_count': 0.4,           # Medium indicator
            'temporal_count': 0.6,           # Medium-strong indicator
            'negation_count': -0.3            # Medium negative indicator
        }
        
        self.learning_rate = 0.1
        
    def extract_features(self, text):
        """Extract features from text"""
        features = {}
        text_lower = text.lower()
        
        # Crisis keywords (specific phrases)
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'commit suicide',
            'take my life', 'want to die', 'going to die', 'plan to die',
            'jump off', 'hang myself', 'overdose', 'shoot myself',
            'kill myself', 'end it all', 'not want to live'
        ]
        
        # Help-seeking keywords
        help_keywords = [
            'help', 'support', 'therapy', 'counselor', 'therapist',
            'treatment', 'medication', 'coping', 'recovery', 'crisis',
            'getting help', 'seeking help', 'professional help',
            'better after', 'working on', 'improving'
        ]
        
        # Count keywords
        features['crisis_keyword_count'] = sum(1 for phrase in crisis_keywords if phrase in text_lower)
        features['help_keyword_count'] = sum(1 for phrase in help_keywords if phrase in text_lower)
        
        # Text features (normalized)
        features['text_length'] = len(text) / 100
        features['word_count'] = len(text.split()) / 20
        
        # Emotional intensity
        intensity_words = ['really', 'so', 'extremely', 'completely', 'totally', 'absolutely']
        features['intensity_count'] = sum(1 for word in intensity_words if word in text_lower)
        
        # Temporal markers
        temporal_words = ['tonight', 'today', 'now', 'immediately', 'right now', 'this moment']
        features['temporal_count'] = sum(1 for word in temporal_words if word in text_lower)
        
        # Negation (positive indicator)
        negation_words = ['not', 'never', 'no', 'can\'t', 'won\'t', 'don\'t']
        features['negation_count'] = sum(1 for word in negation_words if word in text_lower)
        
        return features
    
    def predict_with_confidence(self, text):
        """Predict with confidence score - BALANCED"""
        features = self.extract_features(text)
        
        # Combine base weights with learned weights
        combined_weights = self.base_weights.copy()
        for feature, weight in self.feedback_data['model_weights'].items():
            if feature in combined_weights:
                combined_weights[feature] += weight
        
        # Calculate score
        score = 0
        for feature, value in features.items():
            if feature in combined_weights:
                score += combined_weights[feature] * value
        
        # Sigmoid function with proper threshold
        probability = 1 / (1 + math.exp(-score))
        
        # Determine prediction and confidence
        prediction = 1 if probability > 0.5 else 0
        confidence = abs(probability - 0.5) * 2
        
        return prediction, probability, confidence
    
    def learn_from_feedback(self, text, user_label):
        """Learn from user feedback"""
        prediction, probability, confidence = self.predict_with_confidence(text)
        
        if user_label != prediction:
            # Add correction
            correction = {
                'text': text,
                'predicted_label': prediction,
                'correct_label': user_label,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'features': self.extract_features(text)
            }
            self.feedback_data['corrections'].append(correction)
            self._update_weights(correction)
            
            # Show learning feedback
            st.success(f"📚 Learning from correction: '{text[:50]}...'")
            st.info(f"Model predicted: {prediction} → User corrected to: {user_label}")
        else:
            # Add confirmation
            confirmation = {
                'text': text,
                'predicted_label': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'features': self.extract_features(text)
            }
            self.feedback_data['confirmations'].append(confirmation)
            self._strengthen_weights(confirmation)
            
            # Show confirmation feedback
            st.success(f"✅ Confirmed prediction: '{text[:50]}...'")
    
    def _update_weights(self, correction):
        """Update weights based on correction"""
        features = correction['features']
        predicted = correction['predicted_label']
        correct = correction['correct_label']
        
        # Calculate error
        error = correct - predicted
        
        # Update weights
        for feature, value in features.items():
            if feature not in self.feedback_data['model_weights']:
                self.feedback_data['model_weights'][feature] = 0.0
            
            weight_change = self.learning_rate * error * value
            self.feedback_data['model_weights'][feature] += weight_change
    
    def _strengthen_weights(self, confirmation):
        """Strengthen weights for correct predictions"""
        features = confirmation['features']
        predicted = confirmation['predicted_label']
        
        for feature, value in features.items():
            if feature not in self.feedback_data['model_weights']:
                self.feedback_data['model_weights'][feature] = 0.0
            
            self.feedback_data['model_weights'][feature] += self.learning_rate * 0.1 * value * predicted
    
    def get_stats(self):
        """Get learning statistics"""
        return {
            'total_feedback': len(self.feedback_data['corrections']) + len(self.feedback_data['confirmations']),
            'corrections': len(self.feedback_data['corrections']),
            'confirmations': len(self.feedback_data['confirmations']),
            'learning_rate': self.learning_rate,
            'current_weights': self.feedback_data['model_weights']
        }
    
    def reset_learning(self):
        """Reset learning data"""
        self.feedback_data = {
            'corrections': [],
            'confirmations': [],
            'model_weights': {},
            'learning_history': []
        }
        st.success("🔄 Learning data reset!")

# Initialize session state
if 'learning_system' not in st.session_state:
    st.session_state.learning_system = BalancedLearningSystem()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🧠 Balanced Self-Learning Crisis Detector</h1>
        <p>Properly balanced AI with working self-learning capabilities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste text here for crisis detection analysis...",
            height=150
        )
        
        if st.button("🔍 Analyze Text"):
            if text_input.strip():
                # Analyze text
                prediction, probability, confidence = st.session_state.learning_system.predict_with_confidence(text_input)
                
                # Store in history
                st.session_state.analysis_history.append({
                    'text': text_input,
                    'prediction': prediction,
                    'probability': probability,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Display result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h3>🚨 CRISIS DETECTED</h3>
                        <p><strong>Text:</strong> "{text_input}"</p>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                        <p><strong>⚠️ This text indicates a potential mental health crisis. Please seek immediate help.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <h3>✅ SAFE</h3>
                        <p><strong>Text:</strong> "{text_input}"</p>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                        <p><strong>This text does not indicate an immediate crisis.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feedback section
                st.markdown("### 💬 Help Improve the AI")
                st.markdown("Was this analysis correct? Your feedback helps the AI learn and improve.")
                
                col_feedback1, col_feedback2 = st.columns(2)
                
                with col_feedback1:
                    if st.button("✅ Correct"):
                        st.session_state.learning_system.learn_from_feedback(text_input, prediction)
                
                with col_feedback2:
                    if st.button("❌ Incorrect"):
                        correct_label = 1 if prediction == 0 else 0
                        st.session_state.learning_system.learn_from_feedback(text_input, correct_label)
            
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.markdown("### 📊 Learning Statistics")
        
        # Get stats
        stats = st.session_state.learning_system.get_stats()
        
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
        
        # Learning insights
        if stats['total_feedback'] > 0:
            st.markdown("### 🧠 Learning Insights")
            st.markdown("**Current Weight Changes:**")
            for feature, weight in stats['current_weights'].items():
                if abs(weight) > 0.01:  # Only show significant changes
                    st.text(f"{feature}: {weight:+.4f}")
        
        # Reset button
        if st.button("🔄 Reset Learning"):
            st.session_state.learning_system.reset_learning()
    
    # Test with sample texts
    st.markdown("### 🧪 Test with Sample Texts")
    
    sample_texts = [
        "I want to kill myself tonight",
        "I am feeling better after therapy",
        "I am going to end my life",
        "I am struggling but getting help",
        "I have a plan to commit suicide",
        "I am having a bad day but I'll be okay"
    ]
    
    if st.button("🧪 Test Sample Texts"):
        st.markdown("**Sample Analysis Results:**")
        for text in sample_texts:
            pred, prob, conf = st.session_state.learning_system.predict_with_confidence(text)
            status = "🚨 CRISIS" if pred == 1 else "✅ SAFE"
            st.text(f"{status}: '{text}' (Confidence: {conf:.1%}, Prob: {prob:.3f})")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("### 📋 Recent Analysis History")
        
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
        <h4>🧠 Balanced Self-Learning Crisis Detector</h4>
        <p>Properly balanced AI with working self-learning capabilities.</p>
        <p><strong>⚠️ Important:</strong> This tool is for educational purposes. Always seek professional help for mental health concerns.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
