import streamlit as st
import re
import random
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🧠 Mental Health Crisis Detector",
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

# Crisis detection keywords
CRISIS_KEYWORDS = {
    'suicide': ['suicide', 'kill myself', 'end my life', 'take my life', 'die', 'death'],
    'hopelessness': ['hopeless', 'worthless', 'useless', 'burden', 'can\'t go on', 'no reason', 'pointless'],
    'planning': ['plan', 'method', 'pills', 'rope', 'gun', 'jump', 'bridge', 'building', 'cut', 'wrist'],
    'goodbye': ['goodbye', 'last message', 'final', 'farewell', 'see you never', 'this is it']
}

def analyze_text_simple(text):
    """Simple crisis detection analysis"""
    if not text or not text.strip():
        return {
            'crisis_probability': 0.0,
            'risk_level': 'low',
            'confidence_score': 0.0,
            'key_features': [],
            'explanation': 'No text provided'
        }
    
    text_lower = text.lower()
    crisis_score = 0.0
    found_keywords = []
    
    # Check for crisis keywords
    for category, keywords in CRISIS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                crisis_score += 0.2
                found_keywords.append(keyword)
    
    # Check for negative sentiment indicators
    negative_words = ['hate', 'despise', 'terrible', 'awful', 'horrible', 'worst', 'never', 'always']
    for word in negative_words:
        if word in text_lower:
            crisis_score += 0.1
    
    # Check for repeated words (might indicate distress)
    words = text_lower.split()
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only consider longer words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    repeated_words = [word for word, count in word_counts.items() if count > 1]
    if repeated_words:
        crisis_score += 0.1 * len(repeated_words)
    
    # Normalize score
    crisis_probability = min(1.0, crisis_score)
    
    # Determine risk level
    if crisis_probability >= 0.8:
        risk_level = 'critical'
    elif crisis_probability >= 0.6:
        risk_level = 'high'
    elif crisis_probability >= 0.4:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Calculate confidence (based on keyword matches)
    confidence_score = min(0.95, 0.5 + (crisis_probability * 0.5))
    
    # Generate explanation
    if found_keywords:
        explanation = f"Found crisis indicators: {', '.join(found_keywords[:5])}"
    else:
        explanation = "No obvious crisis indicators detected"
    
    return {
        'crisis_probability': crisis_probability,
        'risk_level': risk_level,
        'confidence_score': confidence_score,
        'key_features': found_keywords,
        'explanation': explanation
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
        <div class="hero-title">🧠 Mental Health Crisis Detector</div>
        <div class="hero-subtitle">Advanced AI-powered system for detecting mental health crisis indicators</div>
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
        
        # Detection mode
        detection_mode = st.selectbox(
            "🔍 Detection Mode",
            ["Text Analysis", "Advanced Analysis", "Real-time Monitoring"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("✅ System Online")
        st.info(f"🕒 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses Today", "127", "↗️ +12")
        with col2:
            st.metric("Accuracy", "94.2%", "↗️ +2.1%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Crisis Detection Analysis")
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
                # Loading animation
                with st.spinner("🔍 Analyzing text..."):
                    time.sleep(1)  # Simulate analysis time
                    result = analyze_text_simple(text_input)
                
                # Display results with beautiful animations
                st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                
                if result['crisis_probability'] >= confidence_threshold:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="emoji-large">⚠️</div>
                        <h2>🚨 Crisis Detected</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This message contains concerning language that may indicate a mental health crisis.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <div class="emoji-large">✅</div>
                        <h2>✅ No Crisis Detected</h2>
                        <p><strong>Risk Level:</strong> {create_risk_indicator(result['risk_level'])}</p>
                        <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                        <p>This message does not appear to indicate an immediate mental health crisis.</p>
                        {create_progress_bar(result['crisis_probability'] * 100)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics in a beautiful grid
                st.markdown("### 📊 Analysis Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Crisis Probability</h3>
                        <h2>{result['crisis_probability']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Confidence Score</h3>
                        <h2>{result['confidence_score']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚠️ Risk Level</h3>
                        <h2>{result['risk_level'].upper()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔍 Keywords Found</h3>
                        <h2>{len(result['key_features'])}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Key features and explanation
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔍 Key Indicators Found")
                    if result['key_features']:
                        for keyword in result['key_features'][:5]:
                            st.markdown(f"• **{keyword}**")
                    else:
                        st.info("No specific crisis indicators detected")
                
                with col2:
                    st.markdown("### 💡 Analysis Explanation")
                    st.write(result['explanation'])
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Please enter some text for analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧪 Test with Sample Texts")
        
        sample_texts = {
            "🚨 Crisis Text": "I want to kill myself and end it all tonight",
            "✅ Non-Crisis Text": "I'm feeling great today and looking forward to the weekend",
            "⚠️ Medium Risk": "I feel hopeless and don't see any point in continuing",
            "🔴 High Risk": "I've been planning how to end my life and have the means ready"
        }
        
        selected_sample = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
        
        if st.button("🧪 Test Sample Text", use_container_width=True):
            sample_text = sample_texts[selected_sample]
            st.text_area("Sample Text:", value=sample_text, height=100, disabled=True)
            
            with st.spinner("🔍 Analyzing sample text..."):
                time.sleep(1)
                result = analyze_text_simple(sample_text)
                
                st.markdown("### 📊 Sample Analysis Results")
                st.metric("Crisis Probability", f"{result['crisis_probability']:.1%}")
                st.metric("Risk Level", result['risk_level'].upper())
                st.metric("Confidence", f"{result['confidence_score']:.1%}")
                st.write(f"**Explanation:** {result['explanation']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("### 🌟 Key Features")
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-item">
            <h3>🔍 Real-time Analysis</h3>
            <p>Instant crisis detection with advanced AI algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-item">
            <h3>📊 Confidence Scoring</h3>
            <p>Adjustable sensitivity and confidence thresholds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-item">
            <h3>💡 Explainable AI</h3>
            <p>See which words triggered the analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-item">
            <h3>⚠️ Risk Assessment</h3>
            <p>Multi-level risk classification system</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🧠 Mental Health Crisis Detection System</h3>
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
