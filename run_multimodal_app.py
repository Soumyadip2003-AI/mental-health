#!/usr/bin/env python3
"""
Launcher for Multimodal Self-Learning Mental Health Crisis Detector
"""

import subprocess
import sys
import os

def main():
    """Launch the multimodal self-learning app"""
    print("🚀 Starting Multimodal Self-Learning Mental Health Crisis Detector...")
    print("=" * 70)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    print("\n🧠 Multimodal Self-Learning Features:")
    print("• Text Analysis - Crisis keyword detection")
    print("• Audio Analysis - Emotional tone and stress indicators")
    print("• Image Analysis - Visual cues and facial expressions")
    print("• Cross-modal Learning - Learns from all modalities")
    print("• Adaptive Weights - Modality importance adjustment")
    print("• Real-time Feedback - Continuous improvement")
    
    print("\n🎯 Supported Modalities:")
    print("• 📝 Text: Crisis keywords, emotional intensity, temporal markers")
    print("• 🎵 Audio: Pitch variance, speech rate, emotional tone")
    print("• 🖼️ Image: Color analysis, facial expressions, composition")
    
    print("\n🌐 Starting web application...")
    print("The app will open in your browser at http://localhost:8502")
    print("Press Ctrl+C to stop the server")
    
    # Run the streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "multimodal_self_learning.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Multimodal app stopped. Thank you for using it!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
