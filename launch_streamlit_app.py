"""
Streamlit App Launcher with Error Handling
=========================================

This script launches the Streamlit crop yield predictor app with
proper error handling and troubleshooting.
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"[ERROR] {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'preprocessed_agricultural_data.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} found")
        else:
            missing_files.append(file)
            print(f"[ERROR] {file} missing")
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        print("Please run the preprocessing pipeline first:")
        print("python agricultural_data_preprocessing.py")
        return False
    
    return True

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return None

def launch_app():
    """Launch the Streamlit app with error handling"""
    print("Crop Yield Predictor - App Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Check data files
    print("\n2. Checking data files...")
    if not check_data_files():
        print("\n❌ Data file check failed. Please ensure data files exist.")
        return False
    
    # Find available port
    print("\n3. Finding available port...")
    port = find_available_port()
    if port is None:
        print("❌ No available ports found")
        return False
    
    print(f"[OK] Using port {port}")
    
    # Try different app versions
    app_files = [
        'streamlit_crop_yield_app_fixed.py',
        'streamlit_crop_yield_app.py',
        'test_streamlit_app.py'
    ]
    
    for app_file in app_files:
        if os.path.exists(app_file):
            print(f"\n4. Launching {app_file}...")
            print(f"📱 The app will open in your default web browser")
            print(f"🔗 URL: http://localhost:{port}")
            print(f"⏹️  Press Ctrl+C to stop the app")
            
            try:
                # Launch the app
                cmd = [
                    sys.executable, "-m", "streamlit", "run", app_file,
                    "--server.port", str(port),
                    "--server.address", "localhost",
                    "--server.headless", "true"
                ]
                
                subprocess.run(cmd)
                return True
                
            except KeyboardInterrupt:
                print("\n👋 App stopped by user")
                return True
            except Exception as e:
                print(f"[ERROR] Error launching {app_file}: {e}")
                continue
    
    print("[ERROR] No working app version found")
    return False

def main():
    """Main function"""
    try:
        success = launch_app()
        if not success:
            print("\nTroubleshooting Tips:")
            print("1. Ensure all dependencies are installed: pip install streamlit plotly pandas numpy scikit-learn")
            print("2. Run the preprocessing pipeline: python agricultural_data_preprocessing.py")
            print("3. Check that you're in the correct directory")
            print("4. Try running the test app: python test_streamlit_app.py")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()
