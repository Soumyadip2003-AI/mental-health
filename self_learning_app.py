import streamlit as st
import re
import random
import json
import os
from datetime import datetime
import time

# Import our self-learning system
from self_learning_detector import InteractiveLearningInterface

# Page configuration
st.set_page_config(
    page_title="🧠 Self-Learning Mental Health Crisis Detector",
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
    
    .learning-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(162, 155, 254, 0.3);
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
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(116, 185, 255, 0.4);
    }
    
    .feedback-section {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(232, 67, 147, 0.3);
    }
    
    .insights-section {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(225, 112, 85, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .feedback-button {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%) !important;
        margin: 0.5rem;
    }
    
    .correction-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        margin: 0.5rem;
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
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'learning_interface' not in st.session_state:
    st.session_state.learning_interface = InteractiveLearningInterface()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🧠 Self-Learning Mental Health Crisis Detector</h1>
        <p class="hero-subtitle">Advanced AI that learns from your feedback to improve crisis detection</p>
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
            height=150,
            help="The AI will analyze the text for potential mental health crisis indicators"
        )
        
        if st.button("🔍 Analyze Text", key="analyze_btn"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Analyze text
                    result = st.session_state.learning_interface.analyze_text(text_input)
                    
                    # Store in history
                    st.session_state.analysis_history.append({
                        'text': text_input,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Display result
                    if result['prediction'] == 1:
                        st.markdown(f"""
                        <div class="analysis-card">
                            <h3>🚨 CRISIS DETECTED</h3>
                            <p><strong>Text:</strong> "{result['text']}"</p>
                            <p><strong>Confidence:</strong> {result['confidence']:.1%} ({result['confidence_level']})</p>
                            <p><strong>Probability:</strong> {result['probability']:.1%}</p>
                            <p><strong>⚠️ This text indicates a potential mental health crisis. Please seek immediate help.</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-card">
                            <h3>✅ SAFE</h3>
                            <p><strong>Text:</strong> "{result['text']}"</p>
                            <p><strong>Confidence:</strong> {result['confidence']:.1%} ({result['confidence_level']})</p>
                            <p><strong>Probability:</strong> {result['probability']:.1%}</p>
                            <p><strong>This text does not indicate an immediate crisis.</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Feedback section
                    st.markdown("### 💬 Help Improve the AI")
                    st.markdown("Was this analysis correct? Your feedback helps the AI learn and improve.")
                    
                    col_feedback1, col_feedback2 = st.columns(2)
                    
                    with col_feedback1:
                        if st.button("✅ Correct", key="correct_btn", help="The AI's analysis was correct"):
                            st.session_state.learning_interface.provide_feedback(
                                text_input, result['prediction']
                            )
                            st.session_state.feedback_count += 1
                            st.success("✅ Thank you! Your feedback has been recorded.")
                    
                    with col_feedback2:
                        if st.button("❌ Incorrect", key="incorrect_btn", help="The AI's analysis was incorrect"):
                            # Get correct label
                            correct_label = 1 if result['prediction'] == 0 else 0
                            st.session_state.learning_interface.provide_feedback(
                                text_input, correct_label
                            )
                            st.session_state.feedback_count += 1
                            st.success("📚 Thank you! The AI is learning from your correction.")
            
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.markdown("### 📊 Learning Statistics")
        
        # Get learning insights
        insights = st.session_state.learning_interface.get_learning_insights()
        session_stats = st.session_state.learning_interface.get_session_stats()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎓 Total Learning</h4>
            <h2>{insights['total_feedback']}</h2>
            <p>Feedback Events</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>✅ Confirmations</h4>
            <h2>{insights['confirmations']}</h2>
            <p>Correct Predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📚 Corrections</h4>
            <h2>{insights['corrections']}</h2>
            <p>Learning Events</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Session</h4>
            <h2>{session_stats['total_analyzed']}</h2>
            <p>Texts Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Learning Insights Section
    if insights['total_feedback'] > 0:
        st.markdown("### 🧠 Learning Insights")
        
        st.markdown("""
        <div class="insights-section">
            <h3>🎯 What the AI Has Learned</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show feature changes
        feature_changes = insights['feature_changes']
        significant_changes = {k: v for k, v in feature_changes.items() if abs(v['change']) > 0.01}
        
        if significant_changes:
            st.markdown("**Feature Importance Changes:**")
            for feature, changes in significant_changes.items():
                change_percent = changes['change_percent']
                change_emoji = "📈" if changes['change'] > 0 else "📉"
                st.markdown(f"- **{feature}**: {change_emoji} {change_percent:+.1f}%")
        else:
            st.info("The AI is still learning. Provide more feedback to see learning insights!")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("### 📋 Recent Analysis History")
        
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
            result = analysis['result']
            status_emoji = "🚨" if result['prediction'] == 1 else "✅"
            status_color = "🔴" if result['prediction'] == 1 else "🟢"
            
            with st.expander(f"{status_emoji} Analysis #{len(st.session_state.analysis_history) - i}"):
                st.markdown(f"**Text:** {analysis['text'][:100]}{'...' if len(analysis['text']) > 100 else ''}")
                st.markdown(f"**Result:** {result['status']} (Confidence: {result['confidence']:.1%})")
                st.markdown(f"**Time:** {analysis['timestamp']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;">
        <h4>🧠 Self-Learning Mental Health Crisis Detector</h4>
        <p>This AI learns from your feedback to improve crisis detection accuracy.</p>
        <p><strong>⚠️ Important:</strong> This tool is for educational purposes. Always seek professional help for mental health concerns.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
