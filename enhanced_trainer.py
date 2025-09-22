#!/usr/bin/env python3
"""
Enhanced Crisis Detection Trainer
Uses the comprehensive dataset to train advanced models
"""

import csv
import random
import os
import re
from collections import Counter
import math

class SimpleTextPreprocessor:
    """Simple text preprocessor without external dependencies"""
    
    def __init__(self):
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def preprocess(self, text):
        """Preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

class SimpleTFIDF:
    """Simple TF-IDF implementation"""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}
        self.feature_names = []
    
    def fit_transform(self, texts):
        """Fit and transform texts"""
        # Build vocabulary
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Get top features
        top_words = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        self.feature_names = [word for word, _ in top_words]
        
        # Calculate IDF
        n_docs = len(texts)
        for word in self.vocabulary:
            doc_count = sum(1 for text in texts if word in text.split())
            self.idf[word] = math.log(n_docs / (1 + doc_count))
        
        # Transform texts
        tfidf_matrix = []
        for text in texts:
            words = text.split()
            word_freq = Counter(words)
            tfidf_vector = []
            
            for word in self.feature_names:
                if word in word_freq:
                    tf = word_freq[word] / len(words)
                    idf = self.idf[word]
                    tfidf_vector.append(tf * idf)
                else:
                    tfidf_vector.append(0.0)
            
            tfidf_matrix.append(tfidf_vector)
        
        return tfidf_matrix
    
    def transform(self, texts):
        """Transform new texts"""
        tfidf_matrix = []
        for text in texts:
            words = text.split()
            word_freq = Counter(words)
            tfidf_vector = []
            
            for word in self.feature_names:
                if word in word_freq:
                    tf = word_freq[word] / len(words)
                    idf = self.idf.get(word, 0)
                    tfidf_vector.append(tf * idf)
                else:
                    tfidf_vector.append(0.0)
            
            tfidf_matrix.append(tfidf_vector)
        
        return tfidf_matrix

class SimpleNaiveBayes:
    """Simple Naive Bayes implementation"""
    
    def __init__(self):
        self.class_priors = {}
        self.word_probs = {}
        self.vocabulary = set()
    
    def fit(self, X, y):
        """Fit the model"""
        # Calculate class priors
        n_samples = len(y)
        class_counts = Counter(y)
        for class_label in class_counts:
            self.class_priors[class_label] = class_counts[class_label] / n_samples
        
        # Calculate word probabilities for each class
        for class_label in class_counts:
            self.word_probs[class_label] = {}
            
            # Get all words in this class
            class_words = []
            for i, label in enumerate(y):
                if label == class_label:
                    class_words.extend(X[i].split())
            
            # Calculate word frequencies
            word_counts = Counter(class_words)
            total_words = len(class_words)
            
            # Add smoothing
            for word in word_counts:
                self.word_probs[class_label][word] = (word_counts[word] + 1) / (total_words + len(self.vocabulary))
                self.vocabulary.add(word)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        predictions = []
        
        for text in X:
            words = text.split()
            class_scores = {}
            
            for class_label in self.class_priors:
                # Start with log prior
                score = math.log(self.class_priors[class_label])
                
                # Add log likelihood for each word
                for word in words:
                    if word in self.word_probs[class_label]:
                        score += math.log(self.word_probs[class_label][word])
                    else:
                        # Smoothing for unseen words
                        score += math.log(1 / (sum(len(text.split()) for text in X) + len(self.vocabulary)))
                
                class_scores[class_label] = score
            
            # Convert to probabilities
            max_score = max(class_scores.values())
            exp_scores = {k: math.exp(v - max_score) for k, v in class_scores.items()}
            total = sum(exp_scores.values())
            probabilities = [exp_scores.get(0, 0) / total, exp_scores.get(1, 0) / total]
            predictions.append(probabilities)
        
        return predictions

class SimpleLogisticRegression:
    """Simple Logistic Regression implementation"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0
    
    def sigmoid(self, z):
        """Sigmoid function"""
        return 1 / (1 + math.exp(-z))
    
    def fit(self, X, y):
        """Fit the model"""
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        
        for iteration in range(self.max_iter):
            for i, features in enumerate(X):
                # Calculate prediction
                z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
                prediction = self.sigmoid(z)
                
                # Calculate error
                error = y[i] - prediction
                
                # Update weights
                for j in range(n_features):
                    self.weights[j] += self.learning_rate * error * features[j]
                self.bias += self.learning_rate * error
    
    def predict_proba(self, X):
        """Predict probabilities"""
        predictions = []
        for features in X:
            z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
            prob = self.sigmoid(z)
            predictions.append([1 - prob, prob])
        return predictions

class EnhancedCrisisDetector:
    """Enhanced crisis detection system"""
    
    def __init__(self):
        self.preprocessor = SimpleTextPreprocessor()
        self.vectorizer = SimpleTFIDF(max_features=5000)
        self.models = {}
        self.is_fitted = False
    
    def fit(self, texts, labels):
        """Fit the model"""
        print("Preprocessing texts...")
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        print("Extracting features...")
        X = self.vectorizer.fit_transform(processed_texts)
        
        print("Training models...")
        
        # Train Naive Bayes
        print("Training Naive Bayes...")
        self.models['nb'] = SimpleNaiveBayes()
        self.models['nb'].fit(processed_texts, labels)
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        self.models['lr'] = SimpleLogisticRegression()
        self.models['lr'].fit(X, labels)
        
        self.is_fitted = True
        print("Training completed!")
    
    def predict_proba(self, texts):
        """Predict probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        
        # Get predictions from both models
        nb_probs = self.models['nb'].predict_proba(processed_texts)
        lr_probs = self.models['lr'].predict_proba(X)
        
        # Average the predictions
        ensemble_probs = []
        for nb_prob, lr_prob in zip(nb_probs, lr_probs):
            avg_prob = [(nb_prob[0] + lr_prob[0]) / 2, (nb_prob[1] + lr_prob[1]) / 2]
            ensemble_probs.append(avg_prob)
        
        return ensemble_probs
    
    def predict(self, texts):
        """Predict labels"""
        probs = self.predict_proba(texts)
        return [1 if prob[1] > 0.5 else 0 for prob in probs]

def load_dataset(filename):
    """Load dataset from CSV"""
    texts = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label']))
    
    return texts, labels

def evaluate_model(y_true, y_pred, y_proba=None):
    """Evaluate model performance"""
    # Calculate basic metrics
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    
    # Calculate precision, recall, F1
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if y_proba:
        # Calculate AUC (simplified)
        sorted_data = sorted(zip(y_proba, y_true), key=lambda x: x[0][1], reverse=True)
        n_positive = sum(y_true)
        n_negative = len(y_true) - n_positive
        
        auc = 0
        rank = 1
        for prob, label in sorted_data:
            if label == 1:
                auc += rank
            rank += 1
        
        auc = (auc - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
        metrics['auc'] = auc
    
    return metrics

def main():
    """Main training function"""
    print("Starting Enhanced Crisis Detection Training...")
    
    # Load datasets
    print("Loading datasets...")
    try:
        train_texts, train_labels = load_dataset("data/comprehensive_crisis_dataset.csv")
        val_texts, val_labels = load_dataset("data/validation_enhanced.csv")
        print(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples")
    except FileNotFoundError:
        print("Dataset files not found. Please run simple_data_generator.py first.")
        return
    
    # Train model
    print("Training enhanced crisis detection model...")
    detector = EnhancedCrisisDetector()
    detector.fit(train_texts, train_labels)
    
    # Evaluate model
    print("Evaluating model...")
    val_predictions = detector.predict(val_texts)
    val_probabilities = detector.predict_proba(val_texts)
    val_proba_scores = [prob[1] for prob in val_probabilities]
    
    metrics = evaluate_model(val_labels, val_predictions, val_probabilities)
    
    print("\nModel Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test on some examples
    print("\nTesting on sample texts:")
    test_texts = [
        "I want to kill myself tonight",
        "I'm feeling better after therapy",
        "I'm going to end my life",
        "I'm struggling but I'm getting help",
        "I have a plan to commit suicide"
    ]
    
    test_predictions = detector.predict(test_texts)
    test_probabilities = detector.predict_proba(test_texts)
    
    for text, pred, prob in zip(test_texts, test_predictions, test_probabilities):
        crisis_prob = prob[1]
        status = "CRISIS" if pred == 1 else "SAFE"
        print(f"Text: '{text}'")
        print(f"Prediction: {status} (Crisis probability: {crisis_prob:.3f})")
        print()
    
    print("Enhanced Crisis Detection Training completed successfully!")

if __name__ == "__main__":
    main()
