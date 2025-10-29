
import sys
import subprocess
import importlib.util
import os

# Create models cache directory
MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)
    print(f"Created models cache directory: {MODELS_CACHE_DIR}")

# --- Package Installation ---
# ADDED 'pandas' for detailed analysis tab functionality.
# REMOVED 'encodec' to complete the requested cleanup.
REQUIRED_PACKAGES = [
    'PyQt5', 'torch', 'torchaudio', 'numpy', 'pyaudio', 
    'scipy', 'librosa', 'matplotlib', 'pystoi', 'pesq', 'soundfile', 
    'dac', 'einops', 'pandas'
]

def check_and_install_packages():
    """Checks and installs required packages."""
    print("Checking required packages...")
    
    # Install required packages
    for package in REQUIRED_PACKAGES:
        module_name = package
        # Map package names to their importable module names where different
        if package == 'PyQt5': module_name = 'PyQt5'
        if package == 'pystoi': module_name = 'pystoi'
        if package == 'pesq': module_name = 'pesq'
        if package == 'dac': module_name = 'dac'
        if package == 'pandas': module_name = 'pandas' # Added pandas check
        
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"Package '{package}' not found. Attempting to install...")
            try:
                if package == 'pesq':
                    # Special installation for pesq dependencies
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pesq[speechmetrics]"])
                elif package == 'dac':
                    # Special installation for DAC
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "descript-audio-codec"])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed '{package}'.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to install '{package}'. Error: {e}")
                if package not in ['einops']: # Some packages are optional
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: An unexpected error occurred during installation of '{package}'. Error: {e}")
                if package not in ['einops']:
                    sys.exit(1)
        else:
            print(f"{package} is already installed.")
    

# Run the package check
print("="*70)
print("Audio Codec Suite - Automatic Setup")
print("="*70)
check_and_install_packages()
print("="*70)
print("Setup complete! Starting application...")
print("="*70 + "\n")

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget
    from streaming_tab import StreamingTab
    from evaluation_tab import EvaluationTab
    from detailed_analysis_tab import DetailedAnalysisTab # Added new tab import
except ImportError as e:
    print(f"Failed to import a required module: {e}")
    print("Please ensure all packages from REQUIRED_PACKAGES are installed.")
    sys.exit(1)


class MainWindow(QMainWindow):
    """The main application window which holds the tabbed interface."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultra Low-Latency Audio Codec Suite")
        self.setGeometry(100, 100, 1050, 800) # Increased size for detailed analysis tab

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.streaming_tab = StreamingTab()
        self.evaluation_tab = EvaluationTab()
        self.detailed_analysis_tab = DetailedAnalysisTab() # New tab instance

        self.tabs.addTab(self.streaming_tab, "Real-Time Streaming")
        self.tabs.addTab(self.evaluation_tab, "Model Evaluation")
        self.tabs.addTab(self.detailed_analysis_tab, "Detailed Comparative Analysis") # New tab added
        
        # Show available codecs in status bar
        self.statusBar().showMessage(self._get_codec_status())
        
    def _get_codec_status(self):
        """Check which codecs are available"""
        status_parts = []
        
        try:
            import dac
            status_parts.append("DAC ✓")
        except:
            status_parts.append("DAC ✗")
            
        try:
            import pandas
            status_parts.append("Pandas ✓")
        except:
            status_parts.append("Pandas ✗")
            
        return "System Status: " + " | ".join(status_parts)
        
    def closeEvent(self, event):
        """Ensures background threads are terminated when the application is closed."""
        print("Closing application...")
        self.streaming_tab.stop_streaming()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
