#!/usr/bin/env python3
"""
Self-Learning Crisis Detection System
Learns and improves from user feedback
"""

import csv
import json
import os
import random
import math
from datetime import datetime
from collections import Counter, defaultdict
import re

class FeedbackLearner:
    """Learns from user feedback to improve predictions"""
    
    def __init__(self, feedback_file="data/user_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self.load_feedback()
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
    def load_feedback(self):
        """Load existing feedback data"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'corrections': [],
            'confirmations': [],
            'model_weights': {},
            'feature_importance': {},
            'learning_history': []
        }
    
    def save_feedback(self):
        """Save feedback data"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def add_correction(self, text, predicted_label, correct_label, confidence):
        """Add a correction to the feedback system"""
        correction = {
            'text': text,
            'predicted_label': predicted_label,
            'correct_label': correct_label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'features': self.extract_features(text)
        }
        self.feedback_data['corrections'].append(correction)
        self.update_model_weights(correction)
        self.save_feedback()
    
    def add_confirmation(self, text, predicted_label, confidence):
        """Add a confirmation to strengthen correct predictions"""
        confirmation = {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'features': self.extract_features(text)
        }
        self.feedback_data['confirmations'].append(confirmation)
        self.strengthen_model_weights(confirmation)
        self.save_feedback()
    
    def extract_features(self, text):
        """Extract features from text for learning"""
        features = {}
        
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
        
        text_lower = text.lower()
        
        # Count crisis keywords
        features['crisis_keyword_count'] = sum(1 for word in crisis_keywords if word in text_lower)
        
        # Count help-seeking keywords
        features['help_keyword_count'] = sum(1 for word in help_keywords if word in text_lower)
        
        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Emotional intensity
        intensity_words = ['really', 'so', 'extremely', 'completely', 'totally']
        features['intensity_count'] = sum(1 for word in intensity_words if word in text_lower)
        
        # Temporal markers
        temporal_words = ['tonight', 'today', 'now', 'immediately', 'right now']
        features['temporal_count'] = sum(1 for word in temporal_words if word in text_lower)
        
        # Negation patterns
        negation_patterns = ['not', 'never', 'no', 'can\'t', 'won\'t']
        features['negation_count'] = sum(1 for word in negation_patterns if word in text_lower)
        
        return features
    
    def update_model_weights(self, correction):
        """Update model weights based on correction"""
        if 'model_weights' not in self.feedback_data:
            self.feedback_data['model_weights'] = {}
        
        features = correction['features']
        predicted = correction['predicted_label']
        correct = correction['correct_label']
        
        # Adjust weights based on error
        for feature, value in features.items():
            if feature not in self.feedback_data['model_weights']:
                self.feedback_data['model_weights'][feature] = 0.0
            
            # Calculate error
            error = correct - predicted
            
            # Update weight
            self.feedback_data['model_weights'][feature] += self.learning_rate * error * value
        
        # Record learning event
        learning_event = {
            'type': 'correction',
            'timestamp': datetime.now().isoformat(),
            'error': abs(correct - predicted),
            'features_updated': len(features)
        }
        self.feedback_data['learning_history'].append(learning_event)
    
    def strengthen_model_weights(self, confirmation):
        """Strengthen weights for correct predictions"""
        if 'model_weights' not in self.feedback_data:
            self.feedback_data['model_weights'] = {}
        
        features = confirmation['features']
        predicted = confirmation['predicted_label']
        
        # Strengthen weights for correct predictions
        for feature, value in features.items():
            if feature not in self.feedback_data['model_weights']:
                self.feedback_data['model_weights'][feature] = 0.0
            
            # Small positive adjustment
            self.feedback_data['model_weights'][feature] += self.learning_rate * 0.1 * value * predicted
    
    def get_adaptive_weights(self):
        """Get current adaptive weights"""
        return self.feedback_data.get('model_weights', {})
    
    def get_learning_stats(self):
        """Get learning statistics"""
        corrections = len(self.feedback_data.get('corrections', []))
        confirmations = len(self.feedback_data.get('confirmations', []))
        total_feedback = corrections + confirmations
        
        return {
            'total_corrections': corrections,
            'total_confirmations': confirmations,
            'total_feedback': total_feedback,
            'learning_rate': self.learning_rate
        }

class SelfLearningCrisisDetector:
    """Crisis detector that learns from user feedback"""
    
    def __init__(self):
        self.feedback_learner = FeedbackLearner()
        self.base_weights = self.load_base_weights()
        self.adaptive_weights = self.feedback_learner.get_adaptive_weights()
        
    def load_base_weights(self):
        """Load base model weights"""
        return {
            'crisis_keyword_count': 0.8,
            'help_keyword_count': -0.6,
            'text_length': 0.1,
            'word_count': 0.05,
            'intensity_count': 0.3,
            'temporal_count': 0.4,
            'negation_count': -0.2
        }
    
    def predict_with_confidence(self, text):
        """Predict with confidence score"""
        features = self.feedback_learner.extract_features(text)
        
        # Combine base weights with adaptive weights
        combined_weights = self.base_weights.copy()
        for feature, weight in self.adaptive_weights.items():
            if feature in combined_weights:
                combined_weights[feature] += weight
        
        # Calculate score
        score = 0
        for feature, value in features.items():
            if feature in combined_weights:
                score += combined_weights[feature] * value
        
        # Convert to probability
        probability = 1 / (1 + math.exp(-score))
        
        # Determine prediction and confidence
        prediction = 1 if probability > 0.5 else 0
        confidence = abs(probability - 0.5) * 2  # Scale to 0-1
        
        return prediction, probability, confidence
    
    def learn_from_feedback(self, text, user_label, user_confidence=None):
        """Learn from user feedback"""
        prediction, probability, confidence = self.predict_with_confidence(text)
        
        # Check if user disagrees with prediction
        if user_label != prediction:
            # Add correction
            self.feedback_learner.add_correction(text, prediction, user_label, confidence)
            print(f"📚 Learning from correction: '{text[:50]}...'")
            print(f"   Model predicted: {prediction} (confidence: {confidence:.3f})")
            print(f"   User corrected to: {user_label}")
        else:
            # Add confirmation
            self.feedback_learner.add_confirmation(text, prediction, confidence)
            print(f"✅ Confirmed prediction: '{text[:50]}...'")
            print(f"   Both model and user agree: {prediction} (confidence: {confidence:.3f})")
        
        # Update adaptive weights
        self.adaptive_weights = self.feedback_learner.get_adaptive_weights()
    
    def get_learning_insights(self):
        """Get insights about what the model has learned"""
        stats = self.feedback_learner.get_learning_stats()
        
        insights = {
            'total_feedback': stats['total_feedback'],
            'corrections': stats['total_corrections'],
            'confirmations': stats['total_confirmations'],
            'learning_rate': stats['learning_rate']
        }
        
        # Analyze feature importance changes
        base_weights = self.base_weights
        adaptive_weights = self.adaptive_weights
        
        feature_changes = {}
        for feature in base_weights:
            original = base_weights[feature]
            current = original + adaptive_weights.get(feature, 0)
            change = current - original
            feature_changes[feature] = {
                'original': original,
                'current': current,
                'change': change,
                'change_percent': (change / abs(original)) * 100 if original != 0 else 0
            }
        
        insights['feature_changes'] = feature_changes
        return insights

class InteractiveLearningInterface:
    """Interactive interface for learning from user feedback"""
    
    def __init__(self):
        self.detector = SelfLearningCrisisDetector()
        self.session_feedback = []
    
    def analyze_text(self, text):
        """Analyze text and return prediction with confidence"""
        prediction, probability, confidence = self.detector.predict_with_confidence(text)
        
        result = {
            'text': text,
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'status': 'CRISIS' if prediction == 1 else 'SAFE',
            'confidence_level': self.get_confidence_level(confidence)
        }
        
        return result
    
    def get_confidence_level(self, confidence):
        """Get confidence level description"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def provide_feedback(self, text, user_label, user_confidence=None):
        """Provide feedback to improve the model"""
        self.detector.learn_from_feedback(text, user_label, user_confidence)
        
        # Track session feedback
        feedback_entry = {
            'text': text,
            'user_label': user_label,
            'timestamp': datetime.now().isoformat()
        }
        self.session_feedback.append(feedback_entry)
    
    def get_session_stats(self):
        """Get statistics for current session"""
        return {
            'total_analyzed': len(self.session_feedback),
            'session_feedback': self.session_feedback
        }
    
    def get_learning_insights(self):
        """Get learning insights"""
        return self.detector.get_learning_insights()
    
    def export_learning_data(self, filename="data/learning_export.json"):
        """Export learning data for analysis"""
        data = {
            'session_stats': self.get_session_stats(),
            'learning_insights': self.get_learning_insights(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Learning data exported to {filename}")

def demo_self_learning():
    """Demonstrate the self-learning system"""
    print("🧠 Self-Learning Crisis Detection System Demo")
    print("=" * 50)
    
    interface = InteractiveLearningInterface()
    
    # Demo texts
    demo_texts = [
        "I want to kill myself tonight",
        "I'm feeling better after therapy",
        "I'm going to end my life",
        "I'm struggling but I'm getting help",
        "I have a plan to commit suicide",
        "I'm having a bad day but I'll be okay",
        "I'm going to jump off the bridge",
        "I'm working on my mental health"
    ]
    
    print("\n📊 Initial Analysis:")
    for text in demo_texts:
        result = interface.analyze_text(text)
        print(f"Text: '{text}'")
        print(f"Prediction: {result['status']} (Confidence: {result['confidence']:.3f})")
        print()
    
    print("\n🎓 Learning from User Feedback:")
    
    # Simulate user feedback
    feedback_cases = [
        ("I want to kill myself tonight", 1, "User confirms crisis"),
        ("I'm feeling better after therapy", 0, "User confirms safe"),
        ("I'm going to end my life", 1, "User confirms crisis"),
        ("I'm struggling but I'm getting help", 0, "User confirms safe"),
        ("I have a plan to commit suicide", 1, "User confirms crisis"),
        ("I'm having a bad day but I'll be okay", 0, "User confirms safe"),
        ("I'm going to jump off the bridge", 1, "User confirms crisis"),
        ("I'm working on my mental health", 0, "User confirms safe")
    ]
    
    for text, user_label, description in feedback_cases:
        print(f"Feedback: {description}")
        interface.provide_feedback(text, user_label)
        print()
    
    print("\n📈 Learning Insights:")
    insights = interface.get_learning_insights()
    print(f"Total feedback received: {insights['total_feedback']}")
    print(f"Corrections: {insights['corrections']}")
    print(f"Confirmations: {insights['confirmations']}")
    
    print("\n🔍 Feature Changes:")
    for feature, changes in insights['feature_changes'].items():
        if abs(changes['change']) > 0.01:  # Only show significant changes
            print(f"{feature}: {changes['original']:.3f} → {changes['current']:.3f} ({changes['change']:+.3f})")
    
    print("\n📊 Re-analysis after Learning:")
    for text in demo_texts[:4]:  # Re-analyze first 4 texts
        result = interface.analyze_text(text)
        print(f"Text: '{text}'")
        print(f"Prediction: {result['status']} (Confidence: {result['confidence']:.3f})")
        print()
    
    # Export learning data
    interface.export_learning_data()
    
    print("✅ Self-learning demo completed!")

if __name__ == "__main__":
    demo_self_learning()
