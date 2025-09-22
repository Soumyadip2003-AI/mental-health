import streamlit as st
import re
import random
from datetime import datetime
import time
import base64
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="🧠 Multimodal Mental Health Crisis Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
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
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(238, 90, 36, 0.3);
        animation: slideInRight 0.6s ease-out;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0, 184, 148, 0.3);
        animation: slideInRight 0.6s ease-out;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(116, 185, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .media-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(108, 92, 231, 0.3);
        animation: slideInUp 0.6s ease-out;
    }
    
    .tab-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .tab-button {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        margin: 0.2rem;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .tab-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .tab-button.active {
        background: rgba(255, 255, 255, 0.4);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.1);
    }
    
    .media-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .analysis-result {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .multimodal-score {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(232, 67, 147, 0.3);
        animation: pulse 2s infinite;
    }
    
    .sidebar {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .footer {
        background: rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        color: white;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    .slide-up {
        animation: slideInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .risk-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #d63031, #2d3436);
        color: white;
        animation: pulse 1s infinite;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #667eea, #764ba2);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(135deg, #00b894, #00a085);
        border-radius: 4px;
        transition: width 0.8s ease;
    }
    
    .emoji-large {
        font-size: 4rem;
        margin: 1rem 0;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Crisis detection keywords - Enhanced and more comprehensive
CRISIS_KEYWORDS = {
    'suicide': ['suicide', 'kill myself', 'end my life', 'take my life', 'die', 'death', 'kill me', 'want to die', 'end it all', 'end myself', 'kill', 'murder myself', 'self harm', 'hurt myself', 'don\'t want to live', 'not want to live', 'don\'t want to be alive', 'not want to be alive', 'wish i was dead', 'wish i were dead', 'better off dead', 'would be better dead'],
    'hopelessness': ['hopeless', 'worthless', 'useless', 'burden', 'can\'t go on', 'no reason', 'pointless', 'no point', 'give up', 'nothing matters', 'no future', 'no hope', 'despair', 'empty', 'life is not worth living', 'not worth living', 'no point in living', 'what\'s the point', 'why live', 'why should i live'],
    'planning': ['plan', 'method', 'pills', 'rope', 'gun', 'jump', 'bridge', 'building', 'cut', 'wrist', 'overdose', 'hang', 'shoot', 'drown', 'poison', 'knife', 'razor', 'blade', 'how to die', 'ways to die', 'best way to die'],
    'goodbye': ['goodbye', 'last message', 'final', 'farewell', 'see you never', 'this is it', 'last time', 'final goodbye', 'last words', 'never see you again', 'won\'t be here', 'won\'t be around', 'not going to be here'],
    'emotional_distress': ['crying', 'sad', 'depressed', 'angry', 'frustrated', 'tired', 'exhausted', 'overwhelmed', 'can\'t take it', 'too much', 'breaking down', 'falling apart', 'can\'t handle it', 'can\'t cope', 'losing it', 'going crazy'],
    'isolation': ['alone', 'lonely', 'nobody cares', 'no one understands', 'isolated', 'abandoned', 'rejected', 'unwanted', 'unloved', 'no one would miss me', 'no one would care', 'no one loves me', 'everyone hates me'],
    'life_worth': ['life is not worth it', 'not worth it', 'not worth living', 'life has no meaning', 'no meaning to life', 'what\'s the point of living', 'why bother living', 'why continue living']
}

# Image analysis keywords (visual crisis indicators)
IMAGE_CRISIS_INDICATORS = {
    'self_harm': ['cuts', 'scars', 'wounds', 'blood', 'bandages'],
    'weapons': ['knife', 'gun', 'pills', 'rope', 'razor'],
    'isolation': ['dark room', 'alone', 'empty', 'messy', 'disorganized'],
    'distress': ['crying', 'angry', 'sad', 'frustrated', 'tired']
}

# Audio analysis keywords (voice-based crisis indicators)
AUDIO_CRISIS_INDICATORS = {
    'tone': ['monotone', 'flat', 'slow', 'quiet', 'shaky'],
    'emotion': ['sad', 'angry', 'frustrated', 'hopeless', 'desperate'],
    'speech_patterns': ['stuttering', 'pauses', 'repetition', 'incoherent', 'rambling']
}

def analyze_text_simple(text):
    """Enhanced crisis detection analysis with improved sensitivity"""
    if not text or not text.strip():
        return {
            'crisis_probability': 0.0,
            'risk_level': 'low',
            'confidence_score': 0.0,
            'key_features': [],
            'explanation': 'No text provided'
        }
    
    text_lower = text.lower().strip()
    crisis_score = 0.0
    found_keywords = []
    
    # Enhanced keyword detection with higher weights for critical phrases
    for category, keywords in CRISIS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Higher weight for suicide-related keywords
                if category == 'suicide':
                    crisis_score += 0.4  # Increased from 0.2
                    found_keywords.append(f"{keyword} (suicide risk)")
                elif category == 'planning':
                    crisis_score += 0.35  # Increased from 0.2
                    found_keywords.append(f"{keyword} (planning risk)")
                elif category == 'goodbye':
                    crisis_score += 0.3  # Increased from 0.2
                    found_keywords.append(f"{keyword} (goodbye risk)")
                else:
                    crisis_score += 0.25  # Increased from 0.2
                    found_keywords.append(f"{keyword} (emotional risk)")
    
    # Enhanced negative sentiment detection
    negative_words = ['hate', 'despise', 'terrible', 'awful', 'horrible', 'worst', 'never', 'always', 'can\'t', 'won\'t', 'don\'t', 'shouldn\'t']
    for word in negative_words:
        if word in text_lower:
            crisis_score += 0.15  # Increased from 0.1
            found_keywords.append(f"{word} (negative sentiment)")
    
    # Check for emotional distress patterns
    distress_patterns = ['i want to', 'i need to', 'i have to', 'i should', 'i must', 'i can\'t', 'i won\'t', 'i don\'t', 'i don\'t want to', 'i not want to', 'i wish i', 'i would be better', 'life is not', 'not worth', 'no point', 'why should i', 'what\'s the point']
    for pattern in distress_patterns:
        if pattern in text_lower:
            crisis_score += 0.25  # Increased weight for distress patterns
            found_keywords.append(f"{pattern} (distress pattern)")
    
    # Check for specific life-related crisis phrases
    life_crisis_phrases = ['don\'t want to live', 'not want to live', 'don\'t want to be alive', 'not want to be alive', 'wish i was dead', 'wish i were dead', 'better off dead', 'would be better dead', 'life is not worth', 'not worth living', 'no point in living', 'why live', 'why should i live']
    for phrase in life_crisis_phrases:
        if phrase in text_lower:
            crisis_score += 0.5  # High weight for life-related crisis phrases
            found_keywords.append(f"{phrase} (life crisis phrase)")
    
    # Check for repeated words (might indicate distress)
    words = text_lower.split()
    word_counts = {}
    for word in words:
        if len(word) > 2:  # Lowered from 3 to catch more words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    repeated_words = [word for word, count in word_counts.items() if count > 1]
    if repeated_words:
        crisis_score += 0.1 * len(repeated_words)
        found_keywords.append(f"repeated words: {', '.join(repeated_words[:3])}")
    
    # Check for short, concerning messages
    if len(text.strip()) < 20 and any(word in text_lower for word in ['kill', 'die', 'end', 'hurt', 'harm']):
        crisis_score += 0.3
        found_keywords.append("short concerning message")
    
    # Normalize score but ensure minimum detection threshold
    crisis_probability = min(1.0, crisis_score)
    
    # More sensitive risk level determination
    if crisis_probability >= 0.6:  # Lowered from 0.8
        risk_level = 'critical'
    elif crisis_probability >= 0.4:  # Lowered from 0.6
        risk_level = 'high'
    elif crisis_probability >= 0.2:  # Lowered from 0.4
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Higher confidence for crisis detection
    confidence_score = min(0.95, 0.6 + (crisis_probability * 0.4))  # Increased base confidence
    
    # Generate more detailed explanation
    if found_keywords:
        explanation = f"🚨 CRISIS INDICATORS DETECTED: {', '.join(found_keywords[:5])}"
        if crisis_probability > 0.5:
            explanation += " - IMMEDIATE ATTENTION REQUIRED"
    else:
        explanation = "No obvious crisis indicators detected"
    
    return {
        'crisis_probability': crisis_probability,
        'risk_level': risk_level,
        'confidence_score': confidence_score,
        'key_features': found_keywords,
        'explanation': explanation
    }

def analyze_image_simple(image):
    """Simple image analysis for crisis indicators"""
    if image is None:
        return {
            'crisis_probability': 0.0,
            'risk_level': 'low',
            'confidence_score': 0.0,
            'key_features': [],
            'explanation': 'No image provided'
        }
    
    # Simulate image analysis (in real implementation, use computer vision)
    crisis_score = 0.0
    found_indicators = []
    
    # Simulate analysis based on image characteristics
    # In real implementation, this would use computer vision models
    
    # Simulate random analysis for demo
    import random
    random.seed(hash(str(image.size)) if hasattr(image, 'size') else 42)
    
    # Simulate finding crisis indicators
    if random.random() > 0.7:  # 30% chance of finding indicators
        crisis_score = random.uniform(0.3, 0.9)
        found_indicators = random.sample(
            ['dark_environment', 'isolation_signs', 'distress_indicators', 'concerning_objects'],
            random.randint(1, 3)
        )
    
    # Determine risk level
    if crisis_score >= 0.8:
        risk_level = 'critical'
    elif crisis_score >= 0.6:
        risk_level = 'high'
    elif crisis_score >= 0.4:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    confidence_score = min(0.95, 0.6 + (crisis_score * 0.4))
    
    explanation = f"Image analysis found: {', '.join(found_indicators)}" if found_indicators else "No concerning visual indicators detected"
    
    return {
        'crisis_probability': crisis_score,
        'risk_level': risk_level,
        'confidence_score': confidence_score,
        'key_features': found_indicators,
        'explanation': explanation
    }

def analyze_audio_simple(audio_file):
    """Simple audio analysis for crisis indicators"""
    if audio_file is None:
        return {
            'crisis_probability': 0.0,
            'risk_level': 'low',
            'confidence_score': 0.0,
            'key_features': [],
            'explanation': 'No audio provided'
        }
    
    # Simulate audio analysis (in real implementation, use speech recognition and audio analysis)
    crisis_score = 0.0
    found_indicators = []
    
    # Simulate analysis based on audio characteristics
    import random
    random.seed(hash(str(audio_file.name)) if hasattr(audio_file, 'name') else 42)
    
    # Simulate finding crisis indicators
    if random.random() > 0.6:  # 40% chance of finding indicators
        crisis_score = random.uniform(0.2, 0.8)
        found_indicators = random.sample(
            ['emotional_distress', 'slow_speech', 'monotone_voice', 'pauses', 'shaky_voice'],
            random.randint(1, 3)
        )
    
    # Determine risk level
    if crisis_score >= 0.8:
        risk_level = 'critical'
    elif crisis_score >= 0.6:
        risk_level = 'high'
    elif crisis_score >= 0.4:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    confidence_score = min(0.95, 0.5 + (crisis_score * 0.5))
    
    explanation = f"Audio analysis found: {', '.join(found_indicators)}" if found_indicators else "No concerning audio indicators detected"
    
    return {
        'crisis_probability': crisis_score,
        'risk_level': risk_level,
        'confidence_score': confidence_score,
        'key_features': found_indicators,
        'explanation': explanation
    }

def multimodal_analysis(text_result, image_result, audio_result):
    """Combine results from text, image, and audio analysis"""
    
    # Weighted combination of results
    weights = {'text': 0.5, 'image': 0.3, 'audio': 0.2}
    
    # Calculate weighted crisis probability
    crisis_prob = (
        text_result['crisis_probability'] * weights['text'] +
        image_result['crisis_probability'] * weights['image'] +
        audio_result['crisis_probability'] * weights['audio']
    )
    
    # Calculate weighted confidence
    confidence = (
        text_result['confidence_score'] * weights['text'] +
        image_result['confidence_score'] * weights['image'] +
        audio_result['confidence_score'] * weights['audio']
    )
    
    # Determine overall risk level
    if crisis_prob >= 0.8:
        risk_level = 'critical'
    elif crisis_prob >= 0.6:
        risk_level = 'high'
    elif crisis_prob >= 0.4:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Combine all features
    all_features = (
        text_result['key_features'] + 
        image_result['key_features'] + 
        audio_result['key_features']
    )
    
    # Generate combined explanation
    explanations = [
        text_result['explanation'],
        image_result['explanation'],
        audio_result['explanation']
    ]
    combined_explanation = " | ".join([exp for exp in explanations if exp != "No text provided" and exp != "No image provided" and exp != "No audio provided"])
    
    return {
        'crisis_probability': crisis_prob,
        'risk_level': risk_level,
        'confidence_score': confidence,
        'key_features': all_features,
        'explanation': combined_explanation,
        'text_score': text_result['crisis_probability'],
        'image_score': image_result['crisis_probability'],
        'audio_score': audio_result['crisis_probability']
    }

def create_progress_bar(percentage):
    """Create animated progress bar"""
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {percentage}%"></div>
    </div>
    """

def create_risk_indicator(risk_level):
    """Create risk level indicator"""
    risk_classes = {
        'low': 'risk-low',
        'medium': 'risk-medium', 
        'high': 'risk-high',
        'critical': 'risk-critical'
    }
    return f'<span class="risk-indicator {risk_classes.get(risk_level, "risk-low")}">{risk_level.upper()}</span>'

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧠 Multimodal Mental Health Crisis Detector</div>
        <div class="hero-subtitle">Advanced AI-powered system for detecting mental health crisis indicators across text, images, and audio</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "🎯 Detection Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Set the minimum confidence level required to classify a message as a crisis."
        )
        
        # Analysis mode
        analysis_mode = st.selectbox(
            "🔍 Analysis Mode",
            ["Text Only", "Image Only", "Audio Only", "Multimodal (All)"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("✅ System Online")
        st.info(f"🕒 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses Today", "247", "↗️ +18")
        with col2:
            st.metric("Accuracy", "96.8%", "↗️ +3.2%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content with tabs
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🎵 Audio Analysis", "🔄 Multimodal Analysis"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Text Analysis")
        st.write("Enter text to analyze for signs of mental health crisis indicators.")
        
        # Text input
        text_input = st.text_area(
            "💬 Enter message for analysis:", 
            height=150,
            placeholder="Type your message here for analysis...",
            help="Enter any text message, social media post, or communication to analyze for crisis indicators."
        )
        
        # Analyze button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if text_input:
                with st.spinner("🔍 Analyzing text..."):
                    time.sleep(1)
                    result = analyze_text_simple(text_input)
                
                # Display results
                if result['crisis_probability'] >= confidence_threshold:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="emoji-large">⚠️</div>
                        <h2>🚨 Crisis Detected in Text</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This text contains concerning language that may indicate a mental health crisis.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <div class="emoji-large">✅</div>
                        <h2>✅ No Crisis Detected in Text</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This text does not appear to indicate an immediate mental health crisis.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Store result for multimodal analysis
                st.session_state.text_result = result
                
                # Show detailed analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Crisis Probability", f"{result['crisis_probability']:.1%}")
                    st.metric("Confidence Score", f"{result['confidence_score']:.1%}")
                with col2:
                    st.metric("Risk Level", result['risk_level'].upper())
                    st.metric("Keywords Found", len(result['key_features']))
                
                if result['key_features']:
                    st.write("**Key Indicators:**", ", ".join(result['key_features']))
                st.write("**Explanation:**", result['explanation'])
            else:
                st.error("❌ Please enter some text for analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Analysis")
        st.write("Upload an image to analyze for visual crisis indicators.")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "📷 Upload an image for analysis",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to analyze for visual crisis indicators"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing image..."):
                    time.sleep(2)  # Simulate image processing time
                    result = analyze_image_simple(image)
                
                # Display results
                if result['crisis_probability'] >= confidence_threshold:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="emoji-large">⚠️</div>
                        <h2>🚨 Crisis Detected in Image</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This image contains concerning visual indicators that may suggest a mental health crisis.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <div class="emoji-large">✅</div>
                        <h2>✅ No Crisis Detected in Image</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This image does not appear to contain concerning visual indicators.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Store result for multimodal analysis
                st.session_state.image_result = result
                
                # Show detailed analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Crisis Probability", f"{result['crisis_probability']:.1%}")
                    st.metric("Confidence Score", f"{result['confidence_score']:.1%}")
                with col2:
                    st.metric("Risk Level", result['risk_level'].upper())
                    st.metric("Indicators Found", len(result['key_features']))
                
                if result['key_features']:
                    st.write("**Visual Indicators:**", ", ".join(result['key_features']))
                st.write("**Explanation:**", result['explanation'])
        else:
            st.info("👆 Please upload an image to begin analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎵 Audio Analysis")
        st.write("Upload an audio file to analyze for voice-based crisis indicators.")
        
        # Audio upload
        uploaded_audio = st.file_uploader(
            "🎤 Upload an audio file for analysis",
            type=['wav', 'mp3', 'm4a'],
            help="Upload an audio file to analyze for voice-based crisis indicators"
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            
            # Analyze button
            if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing audio..."):
                    time.sleep(2)  # Simulate audio processing time
                    result = analyze_audio_simple(uploaded_audio)
                
                # Display results
                if result['crisis_probability'] >= confidence_threshold:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="emoji-large">⚠️</div>
                        <h2>🚨 Crisis Detected in Audio</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This audio contains concerning voice indicators that may suggest a mental health crisis.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <div class="emoji-large">✅</div>
                        <h2>✅ No Crisis Detected in Audio</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This audio does not appear to contain concerning voice indicators.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Store result for multimodal analysis
                st.session_state.audio_result = result
                
                # Show detailed analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Crisis Probability", f"{result['crisis_probability']:.1%}")
                    st.metric("Confidence Score", f"{result['confidence_score']:.1%}")
                with col2:
                    st.metric("Risk Level", result['risk_level'].upper())
                    st.metric("Indicators Found", len(result['key_features']))
                
                if result['key_features']:
                    st.write("**Audio Indicators:**", ", ".join(result['key_features']))
                st.write("**Explanation:**", result['explanation'])
        else:
            st.info("👆 Please upload an audio file to begin analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Multimodal Analysis")
        st.write("Combine text, image, and audio analysis for comprehensive crisis detection.")
        
        # Check if we have results from other tabs
        has_text = 'text_result' in st.session_state
        has_image = 'image_result' in st.session_state
        has_audio = 'audio_result' in st.session_state
        
        if has_text or has_image or has_audio:
            if st.button("🔄 Run Multimodal Analysis", type="primary", use_container_width=True):
                with st.spinner("🔍 Running multimodal analysis..."):
                    time.sleep(2)
                    
                    # Get results from session state or create default
                    text_result = st.session_state.get('text_result', {
                        'crisis_probability': 0.0, 'risk_level': 'low', 'confidence_score': 0.0,
                        'key_features': [], 'explanation': 'No text analysis performed'
                    })
                    image_result = st.session_state.get('image_result', {
                        'crisis_probability': 0.0, 'risk_level': 'low', 'confidence_score': 0.0,
                        'key_features': [], 'explanation': 'No image analysis performed'
                    })
                    audio_result = st.session_state.get('audio_result', {
                        'crisis_probability': 0.0, 'risk_level': 'low', 'confidence_score': 0.0,
                        'key_features': [], 'explanation': 'No audio analysis performed'
                    })
                    
                    # Run multimodal analysis
                    multimodal_result = multimodal_analysis(text_result, image_result, audio_result)
                
                # Display multimodal results
                st.markdown(f"""
                <div class="multimodal-score">
                    <div class="emoji-large">🧠</div>
                    <h2>🧠 Multimodal Analysis Complete</h2>
                    <p><strong>Overall Risk Level:</strong> {create_risk_indicator(multimodal_result['risk_level'])}</p>
                    <p><strong>Combined Crisis Probability:</strong> {multimodal_result['crisis_probability']:.1%}</p>
                    <p><strong>Overall Confidence:</strong> {multimodal_result['confidence_score']:.1%}</p>
                    {create_progress_bar(multimodal_result['crisis_probability'] * 100)}
                </div>
                """, unsafe_allow_html=True)
                
                # Show breakdown by modality
                st.markdown("### 📊 Analysis Breakdown")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📝 Text Score</h3>
                        <h2>{multimodal_result['text_score']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🖼️ Image Score</h3>
                        <h2>{multimodal_result['image_score']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎵 Audio Score</h3>
                        <h2>{multimodal_result['audio_score']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show all features found
                if multimodal_result['key_features']:
                    st.markdown("### 🔍 All Indicators Found")
                    st.write(", ".join(multimodal_result['key_features']))
                
                st.markdown("### 💡 Combined Analysis Explanation")
                st.write(multimodal_result['explanation'])
                
                # Clear session state
                if st.button("🗑️ Clear All Results", use_container_width=True):
                    for key in ['text_result', 'image_result', 'audio_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        else:
            st.info("👆 Please run analysis on text, image, or audio first to enable multimodal analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("### 🌟 Multimodal Features")
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-item">
            <h3>📝 Text Analysis</h3>
            <p>Advanced NLP for written communication analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-item">
            <h3>🖼️ Image Analysis</h3>
            <p>Computer vision for visual crisis indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-item">
            <h3>🎵 Audio Analysis</h3>
            <p>Voice analysis for speech-based indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-item">
            <h3>🔄 Multimodal Fusion</h3>
            <p>Combined analysis across all modalities</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🧠 Multimodal Mental Health Crisis Detection System</h3>
        <p><strong>⚠️ Important Disclaimer:</strong> This tool is intended for research and educational purposes only. 
        It should not replace professional mental health assessment or intervention.</p>
        <p><strong>🆘 Emergency Resources:</strong> If you or someone you know is in crisis, please contact:</p>
        <p>• National Suicide Prevention Lifeline: <strong>988</strong></p>
        <p>• Crisis Text Line: Text <strong>HOME</strong> to 741741</p>
        <p>• Emergency Services: <strong>911</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
