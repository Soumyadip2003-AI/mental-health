#!/usr/bin/env python3
"""
Ultimate Self-Learning Multimodal AI Launcher
"""

import subprocess
import sys
import os

def main():
    print("🧠 Ultimate Self-Learning Multimodal AI")
    print("=" * 50)
    print("🚀 Starting Ultimate AI System...")
    print("✅ Advanced self-learning with neural networks")
    print("✅ Multimodal analysis (text, audio, image)")
    print("✅ Real-time learning and adaptation")
    print("✅ Comprehensive AI monitoring")
    print("🌐 Starting web application...")
    print("The app will open in your browser at http://localhost:8507")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run the ultimate AI system
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ultimate_ai_system.py", 
            "--server.port", "8507", 
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Ultimate AI System stopped!")
    except Exception as e:
        print(f"❌ Error starting Ultimate AI System: {e}")

if __name__ == "__main__":
    main()
