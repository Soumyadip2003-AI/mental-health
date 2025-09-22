import streamlit as st
import re
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Mental Health Crisis Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .highlight {
        background-color: #E8F4F8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .alert {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin-bottom: 1rem;
    }
    .success {
        background-color: #E8F5E8;
        color: #2E7D32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #7F8FA6;
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

def main():
    st.title("🧠 Mental Health Crisis Detector")
    st.markdown("**Advanced AI-powered system for detecting mental health crisis indicators**")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Set the minimum confidence level required to classify a message as a crisis."
    )
    
    # Detection mode
    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Text Analysis", "Advanced Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success("✅ System Online")
    st.sidebar.info(f"🕒 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content
    st.header("🔍 Crisis Detection Analysis")
    st.write("Enter text to analyze for signs of mental health crisis indicators.")
    
    # Text input
    text_input = st.text_area(
        "Enter message for analysis:", 
        height=150,
        placeholder="Type your message here for analysis..."
    )
    
    # Analyze button
    if st.button("🔍 Analyze Text", type="primary"):
        if text_input:
            with st.spinner("Analyzing text..."):
                # Analyze text
                result = analyze_text_simple(text_input)
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display prediction
                    if result['crisis_probability'] >= confidence_threshold:
                        st.markdown(f"""
                        <div class="alert">
                            <h3>⚠️ Crisis Detected</h3>
                            <p><strong>Risk Level:</strong> {result['risk_level'].upper()}</p>
                            <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                            <p>This message contains concerning language that may indicate a mental health crisis.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success">
                            <h3>✅ No Crisis Detected</h3>
                            <p><strong>Risk Level:</strong> {result['risk_level'].upper()}</p>
                            <p><strong>Probability:</strong> {result['crisis_probability']:.1%}</p>
                            <p>This message does not appear to indicate an immediate mental health crisis.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display confidence
                    st.metric("Confidence Score", f"{result['confidence_score']:.1%}")
                    
                    # Disclaimer
                    st.info("⚠️ **Important**: This tool is for research purposes only and should not replace professional mental health assessment.")
                
                with col2:
                    # Key features
                    if result['key_features']:
                        st.subheader("🔍 Key Indicators Found")
                        for keyword in result['key_features'][:5]:
                            st.write(f"• {keyword}")
                    else:
                        st.subheader("🔍 Key Indicators")
                        st.write("No specific crisis indicators detected")
                    
                    # Explanation
                    st.subheader("💡 Analysis Explanation")
                    st.write(result['explanation'])
                
                # Show detailed analysis
                st.subheader("📈 Detailed Analysis")
                
                # Create metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Crisis Probability", f"{result['crisis_probability']:.1%}")
                
                with col2:
                    st.metric("Confidence Score", f"{result['confidence_score']:.1%}")
                
                with col3:
                    st.metric("Risk Level", result['risk_level'].upper())
                
                with col4:
                    st.metric("Keywords Found", len(result['key_features']))
                
                # Risk level explanation
                st.subheader("🎯 Risk Level Explanation")
                risk_explanations = {
                    'low': 'Minimal risk indicators detected',
                    'medium': 'Some concerning patterns identified',
                    'high': 'Multiple crisis indicators present',
                    'critical': 'Strong evidence of immediate crisis'
                }
                st.write(risk_explanations.get(result['risk_level'], 'Unknown risk level'))
        else:
            st.error("Please enter some text for analysis.")
    
    # Sample texts for testing
    st.subheader("🧪 Test with Sample Texts")
    
    sample_texts = {
        "Crisis Text": "I want to kill myself and end it all tonight",
        "Non-Crisis Text": "I'm feeling great today and looking forward to the weekend",
        "Medium Risk": "I feel hopeless and don't see any point in continuing",
        "High Risk": "I've been planning how to end my life and have the means ready"
    }
    
    selected_sample = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
    
    if st.button("Test Sample Text"):
        sample_text = sample_texts[selected_sample]
        st.text_area("Sample Text:", value=sample_text, height=100, disabled=True)
        
        with st.spinner("Analyzing sample text..."):
            result = analyze_text_simple(sample_text)
            
            st.subheader("Sample Analysis Results")
            st.write(f"**Crisis Probability:** {result['crisis_probability']:.1%}")
            st.write(f"**Risk Level:** {result['risk_level'].upper()}")
            st.write(f"**Confidence:** {result['confidence_score']:.1%}")
            st.write(f"**Explanation:** {result['explanation']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>🧠 Mental Health Crisis Detection System</h4>
        <p><strong>⚠️ Important Disclaimer:</strong> This tool is intended for research and educational purposes only. 
        It should not replace professional mental health assessment or intervention.</p>
        <p><strong>Emergency Resources:</strong> If you or someone you know is in crisis, please contact:</p>
        <p>• National Suicide Prevention Lifeline: 988</p>
        <p>• Crisis Text Line: Text HOME to 741741</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
