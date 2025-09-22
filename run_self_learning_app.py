#!/usr/bin/env python3
"""
Launcher for Self-Learning Mental Health Crisis Detector
"""

import subprocess
import sys
import os

def main():
    """Launch the self-learning app"""
    print("🚀 Starting Self-Learning Mental Health Crisis Detector...")
    print("=" * 60)
    
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
    
    print("\n🧠 Self-Learning Features:")
    print("• Learns from user feedback")
    print("• Adapts model weights in real-time")
    print("• Tracks learning statistics")
    print("• Exports learning data")
    print("• Improves accuracy over time")
    
    print("\n🌐 Starting web application...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    # Run the streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "self_learning_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Self-Learning app stopped. Thank you for using it!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
