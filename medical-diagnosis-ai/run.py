#!/usr/bin/env python3
"""
Convenience script to run the AI Medical Diagnosis System
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def run_frontend():
    """Run the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    # Wait a bit for backend to start
    time.sleep(10)
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.serverAddress", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    package_to_module = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'streamlit': 'streamlit',
        'tensorflow': 'tensorflow',
        'opencv-python': 'cv2',
        'transformers': 'transformers',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'plotly': 'plotly',
        'requests': 'requests',
        'spacy': 'spacy',
        'scispacy': 'scispacy'
    }

    missing_packages = []
    optional_packages = []
    
    for package, module in package_to_module.items():
        try:
            __import__(module)
        except ImportError:
            if package in ['spacy', 'scispacy']:
                optional_packages.append(package)
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    if optional_packages:
        print("⚠️ Optional packages not found (enhanced NLP features will be limited):")
        for package in optional_packages:
            print(f"   - {package}")
        print("\n💡 For enhanced medical NLP, install with:")
        print("   pip install spacy scispacy")
        print("   python -m spacy download en_core_web_sm")
    
    print("✅ All core dependencies are installed")
    
    # Check spaCy models
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_sci_sm")
            print("✅ scispaCy medical model loaded successfully")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
                print("✅ Standard spaCy model loaded (install scispaCy for medical entities)")
            except OSError:
                print("⚠️ No spaCy models found. Install with:")
                print("   python -m spacy download en_core_web_sm")
    except ImportError:
        pass
    
    return True

def main():
    """Main function to run the application"""
    print("🏥 AI Medical Diagnosis System")
    print("Enhanced with spaCy & Adaptive Recommendations")
    print("=" * 55)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create app directory if it doesn't exist
    app_dir = Path("app")
    app_dir.mkdir(exist_ok=True)
    
    if not Path("app/main.py").exists():
        print("❌ FastAPI backend file not found: app/main.py")
        sys.exit(1)
    
    if not Path("streamlit_app.py").exists():
        print("❌ Streamlit frontend file not found: streamlit_app.py")
        sys.exit(1)
    
    print("🎯 Starting both servers...")
    print("   Backend:  http://localhost:8000")
    print("   Frontend: http://localhost:8501")
    print("   API Docs: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop both servers")
    print("-" * 55)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    try:
        # Run frontend in main thread
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main() 