import os 
import numpy as np
import librosa
import soundfile as sf
import torch
import pandas as pd
import time
import math
from pesq import pesq
from pystoi import stoi

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QComboBox, QGroupBox, QProgressBar, QTextEdit, 
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model import (
    MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec, 
    HOP_SIZE, DAC_AVAILABLE, SR
)

# --- Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

# --- Utility Functions ---

# Bitrate calculation (aligned with NEW ~9.33kbps trainer)
DOWN_FACTOR_TINY = 24
NUM_CODEBOOKS_TINY = 2
CODEBOOK_SIZE_TINY = 128 # UPDATED from 64
BITRATE_TINY = (SR / DOWN_FACTOR_TINY * math.log2(CODEBOOK_SIZE_TINY) * NUM_CODEBOOKS_TINY) / 1000

# Function to get model instance and metadata
def get_model_and_meta(model_name, model_path):
    model = None
    bitrate = "N/A"
    latency = "N/A"
    
    try:
        if "Uncompressed" in model_name:
            bitrate = "256.00 kbps"
            latency = "0ms"
            # model remains None
        elif "μ-Law" in model_name: 
            model = MuLawCodec()
            bitrate = "128.00 kbps"
            latency = "15ms"
        elif "A-Law" in model_name: 
            model = ALawCodec()
            bitrate = "128.00 kbps"
            latency = "15ms"
        elif "AMR-WB" in model_name: 
            model = AMRWBCodec()
            bitrate = "~12.65 kbps"
            latency = "20ms"
        elif "DAC" in model_name and DAC_AVAILABLE:
            model = DACCodec(model_type="16khz")
            bitrate = "~8-12 kbps"
            latency = "20ms"
        elif "Tiny Transformer" in model_name and os.path.exists(model_path):
            model = TinyTransformerCodec.load_model(model_path) 
            bitrate = f"{BITRATE_TINY:.2f} kbps" # Will now calculate ~9.33
            latency = "15ms"
        
        return model, bitrate, latency, None
    except Exception as e:
        return None, "Error", "Error", str(e)


# --- MODIFIED Analysis Worker Thread ---
class AnalysisWorker(QObject):
    finished = pyqtSignal(list)
    progress_update = pyqtSignal(int, str) # Signal for progress bar and status updates
    
    def __init__(self, audio_file_path, tiny_model_path):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.tiny_model_path = tiny_model_path
        # Use CPU for stability in multi-threaded analysis
        self.device = torch.device("cpu")
        
        self.models_to_test = [
            "Uncompressed",
            "μ-Law Codec (Baseline)", 
            "A-Law Codec (Baseline)",
            "AMR-WB (Simulated)", 
            "DAC Codec (16kHz)",
            "Tiny Transformer Codec (Custom)",
        ]

    def run(self):
        all_results = []
        sr = SR
        original_wav = None

        try:
            # Load original audio once
            original_wav, _ = librosa.load(self.audio_file_path, sr=sr, mono=True)
            original_wav = original_wav.astype(np.float32)
        except Exception as e:
            self.progress_update.emit(0, f"FATAL ERROR: Could not load audio file. {e}")
            self.finished.emit(all_results)
            return

        num_models = len(self.models_to_test)
        
        # Define window and hop sizes from training script
        HOP_SAMPLES = int(15 * sr / 1000) # 240 samples
        WINDOW_SAMPLES = int(30 * sr / 1000) # 480 samples
        
        for i, model_name in enumerate(self.models_to_test):
            model_results = {'name': model_name, 'error': None}
            self.progress_update.emit(int(((i + 1) / num_models) * 100), f"Analyzing: {model_name}...")
            
            # Load model and get metadata
            model, bitrate, latency, error_msg = get_model_and_meta(model_name, self.tiny_model_path)
            model_results.update({'bitrate': bitrate, 'latency': latency})

            if error_msg:
                model_results['error'] = f"Model Load Error: {error_msg}"
                all_results.append(model_results)
                continue
            
            if model_name == "DAC Codec (16kHz)" and not DAC_AVAILABLE:
                model_results['error'] = "DAC not installed/available."
                all_results.append(model_results)
                continue
            
            try:
                start_time = time.time()
                reconstructed_wav = np.copy(original_wav) # Default to passthrough

                with torch.no_grad():
                    if model is not None:
                        # --- MODIFICATION: Use Overlap-Save (OaS) streaming evaluation ---
                        reconstructed_chunks = []
                        
                        # Iterate over the audio with a *hop* of 240 samples
                        for chunk_i in range(0, len(original_wav), HOP_SAMPLES):
                            # Get a *window* of 480 samples
                            chunk = original_wav[chunk_i : chunk_i + WINDOW_SAMPLES]
                            
                            # Pad the final chunk if it's too short
                            if len(chunk) < WINDOW_SAMPLES:
                                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)), 'constant')
                            
                            audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device, dtype=torch.float32)

                            if isinstance(model, (DACCodec, TinyTransformerCodec)):
                                if audio_tensor.dim() == 2:
                                    audio_tensor = audio_tensor.unsqueeze(1) # (1, 1, L_chunk)
                                
                                if isinstance(model, TinyTransformerCodec):
                                    # 1. Encode the 480-sample window
                                    codes, _, orig_len, _ = model.encode(audio_tensor)
                                    # 2. Decode *without* skip connections (HONEST streaming test)
                                    reconstructed_tensor = model.decode(codes, orig_len, encoder_outputs=None)
                                else: # DACCodec
                                    codes, orig_len = model.encode(audio_tensor)
                                    reconstructed_tensor = model.decode(codes, orig_len)
                                
                                decoded_audio = reconstructed_tensor.squeeze().detach().cpu().numpy()
                                
                                # 3. OaS: Keep only the *last* 240 samples (the valid new audio)
                                new_audio = decoded_audio[-HOP_SAMPLES:]
                                reconstructed_chunks.append(new_audio)
                                
                            else:
                                # Traditional codecs (Mu-Law, A-Law, AMR-WB)
                                # These are stateless, so we process hop-by-hop
                                hop_chunk = original_wav[chunk_i : chunk_i + HOP_SAMPLES]
                                audio_tensor = torch.from_numpy(hop_chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                                encoded = model.encode(audio_tensor)
                                reconstructed_tensor = model.decode(encoded)
                                reconstructed_chunks.append(reconstructed_tensor.squeeze().detach().cpu().numpy())
                        
                        if reconstructed_chunks:
                            reconstructed_wav = np.concatenate(reconstructed_chunks)
                    
                    # If model is None ("Uncompressed"), reconstructed_wav stays as the copy.
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    audio_duration = len(original_wav) / sr
                    real_time_factor = processing_time / audio_duration

                    # Ensure lengths match for metrics
                    min_len = min(len(original_wav), len(reconstructed_wav))
                    original_chunk, reconstructed_chunk = original_wav[:min_len], reconstructed_wav[:min_len]

                    # Metrics
                    pesq_score = pesq(int(sr), original_chunk, reconstructed_chunk, 'wb') 
                    stoi_score = stoi(original_chunk, reconstructed_chunk, int(sr), extended=False)
                    
                    model_results.update({
                        'pesq': pesq_score,
                        'stoi': stoi_score,
                        'rtf': real_time_factor,
                        'processing_time': processing_time,
                        'audio_duration': audio_duration
                    })

            except Exception as e:
                model_results['error'] = f"Evaluation Error: {e}"
                import traceback
                print(f"Error during {model_name} evaluation: {traceback.format_exc()}")
            
            all_results.append(model_results)

        self.progress_update.emit(100, "Analysis complete!")
        self.finished.emit(all_results)


# --- Detailed Analysis Tab UI ---
class DetailedAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.results_data = []
        self._setup_ui()
        self.eval_thread = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Config Group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        file_layout = QHBoxLayout()
        self.audio_file_edit = QLineEdit()
        self.audio_file_edit.setPlaceholderText("Path to a representative audio file (.wav) for analysis...")
        self.browse_audio_button = QPushButton("Browse Audio...")
        self.browse_audio_button.clicked.connect(self.browse_audio)
        file_layout.addWidget(QLabel("Audio File:"))
        file_layout.addWidget(self.audio_file_edit)
        file_layout.addWidget(self.browse_audio_button)
        config_layout.addLayout(file_layout)
        
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to Tiny Transformer Codec (.pt) file.")
        self.browse_model_button = QPushButton("Browse Tiny Codec...")
        self.browse_model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(QLabel("Tiny Codec Model:"))
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.browse_model_button)
        config_layout.addLayout(model_layout)
        
        self.run_analysis_button = QPushButton("RUN ALL-MODEL COMPARATIVE ANALYSIS")
        self.run_analysis_button.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        config_layout.addWidget(self.run_analysis_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        config_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Status: Ready to analyze all available codecs.")
        config_layout.addWidget(self.status_label)

        layout.addWidget(config_group)
        
        # Results Group
        results_group = QGroupBox("Comparative Results")
        results_layout = QVBoxLayout(results_group)
        
        # Table for numeric data
        self.results_table = QTableWidget(0, 7) # Rows: dynamic, Cols: 7 (Name, Bitrate, Latency, PESQ, STOI, RTF, Error)
        self.results_table.setHorizontalHeaderLabels(["Codec Name", "Bitrate (kbps)", "Latency (ms)", "PESQ (wb)", "STOI", "RTF (Time Factor)", "Status/Error"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setMinimumHeight(250)
        results_layout.addWidget(self.results_table)
        
        # Plots area
        plot_control_layout = QHBoxLayout()
        self.canvas_metrics = MplCanvas(self, height=3)
        self.canvas_rtf = MplCanvas(self, height=3)
        plot_control_layout.addWidget(self.canvas_metrics)
        plot_control_layout.addWidget(self.canvas_rtf)
        results_layout.addLayout(plot_control_layout)
        
        # Data Export
        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Results (.txt)")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        export_layout.addStretch()
        export_layout.addWidget(self.export_button)
        results_layout.addLayout(export_layout)

        layout.addWidget(results_group)

    def browse_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
        if filepath:
            self.audio_file_edit.setText(filepath)

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
        if filepath:
            self.model_path_edit.setText(filepath)

    def run_analysis(self):
        audio_path = self.audio_file_edit.text()
        tiny_model_path = self.model_path_edit.text()
        
        if not os.path.exists(audio_path) or not os.path.isfile(audio_path):
            self.status_label.setText("Status: ERROR - Please select a valid audio file.")
            return

        if not os.path.exists(tiny_model_path) or not os.path.isfile(tiny_model_path):
             self.status_label.setText("Status: WARNING - Tiny Codec path is invalid. It will be skipped.")
             # We proceed with analysis, but the Tiny Codec will fail gracefully in the worker

        self.run_analysis_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Starting comparative analysis...")

        self.eval_thread = QThread()
        self.worker = AnalysisWorker(audio_path, tiny_model_path)
        self.worker.moveToThread(self.eval_thread)
        
        self.worker.progress_update.connect(self.update_progress, Qt.QueuedConnection)
        self.worker.finished.connect(self.on_analysis_complete, Qt.QueuedConnection)
        self.eval_thread.started.connect(self.worker.run)
        self.eval_thread.finished.connect(self.eval_thread.deleteLater) # Clean up thread
        self.worker.finished.connect(self.worker.deleteLater) # Clean up worker
        self.eval_thread.start()

    def update_progress(self, percentage, message):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(f"Status: {message}")

    def on_analysis_complete(self, results):
        self.results_data = results
        self.display_results(results)
        self.plot_results(results)
        
        self.run_analysis_button.setEnabled(True)
        self.export_button.setEnabled(True)
        if self.eval_thread:
            self.eval_thread.quit()
            self.eval_thread.wait()
            self.eval_thread = None
        self.status_label.setText("Status: Analysis complete. Ready to export.")

    def display_results(self, results):
        self.results_table.setRowCount(len(results))
        for row, data in enumerate(results):
            name = data.get('name', 'N/A')
            bitrate = data.get('bitrate', 'N/A')
            latency = data.get('latency', 'N/A')
            pesq = f"{data.get('pesq', 0.0):.4f}" if data.get('pesq') is not None else "--"
            stoi = f"{data.get('stoi', 0.0):.4f}" if data.get('stoi') is not None else "--"
            rtf = f"{data.get('rtf', 0.0):.3f}" if data.get('rtf') is not None else "--"
            error = data.get('error', "Success")
            
            # Format RTF/Status text
            if rtf != "--":
                 rtf_text = f"{rtf} (Real-time: {'Yes' if float(rtf) < 1.0 else 'No'})"
            else:
                 rtf_text = rtf

            self.results_table.setItem(row, 0, QTableWidgetItem(name))
            self.results_table.setItem(row, 1, QTableWidgetItem(bitrate.split(' ')[0]))
            self.results_table.setItem(row, 2, QTableWidgetItem(latency.split('m')[0].strip('~')))
            self.results_table.setItem(row, 3, QTableWidgetItem(pesq))
            self.results_table.setItem(row, 4, QTableWidgetItem(stoi))
            self.results_table.setItem(row, 5, QTableWidgetItem(rtf_text))
            self.results_table.setItem(row, 6, QTableWidgetItem(error))

    def plot_results(self, results):
        df = pd.DataFrame([
            {
                'name': d['name'], 
                'pesq': d.get('pesq', 0.0), 
                'stoi': d.get('stoi', 0.0), 
                'rtf': d.get('rtf', 0.001), 
                'error': d.get('error')
            } for d in results if d.get('error') is None
        ])
        
        if df.empty:
            self.canvas_metrics.axes.cla(); self.canvas_metrics.axes.set_title("No Data to Plot"); self.canvas_metrics.draw()
            self.canvas_rtf.axes.cla(); self.canvas_rtf.axes.set_title("No Data to Plot"); self.canvas_rtf.draw()
            return

        # Plot 1: PESQ vs STOI
        self.canvas_metrics.axes.cla()
        bar_width = 0.35
        r1 = np.arange(len(df['name']))
        r2 = [x + bar_width for x in r1]
        
        self.canvas_metrics.axes.bar(r1, df['pesq'], color='#007bff', width=bar_width, edgecolor='grey', label='PESQ')
        self.canvas_metrics.axes.bar(r2, df['stoi'] * 4, color='#28a745', width=bar_width, edgecolor='grey', label='STOI (Scaled x4)')
        
        self.canvas_metrics.axes.set_xlabel('Codec Model')
        self.canvas_metrics.axes.set_xticks([r + bar_width/2 for r in r1])
        self.canvas_metrics.axes.set_xticklabels(df['name'], rotation=45, ha="right")
        self.canvas_metrics.axes.set_title('PESQ and STOI Performance (Honest Streaming Mode)')
        self.canvas_metrics.axes.legend()
        self.canvas_metrics.fig.tight_layout()
        self.canvas_metrics.draw()

        # Plot 2: Real-Time Factor (RTF)
        self.canvas_rtf.axes.cla()
        # Ensure RTF values are not zero for log scaling, use log scale for clear visualization
        rtf_values = df['rtf'].replace(0, 0.001) 
        colors = ['green' if r < 1.0 else 'red' for r in rtf_values]

        self.canvas_rtf.axes.bar(df['name'], rtf_values, color=colors)
        self.canvas_rtf.axes.axhline(y=1.0, color='blue', linestyle='--', label='Real-time boundary (RTF=1.0)')
        self.canvas_rtf.axes.set_yscale('log')
        self.canvas_rtf.axes.set_ylim(bottom=0.01)

        self.canvas_rtf.axes.set_xlabel('Codec Model')
        self.canvas_rtf.axes.set_ylabel('Real-Time Factor (RTF, Log Scale)')
        self.canvas_rtf.axes.set_title('Codec Processing Speed (Lower is Better)')
        self.canvas_rtf.axes.set_xticklabels(df['name'], rotation=45, ha="right")
        self.canvas_rtf.axes.legend()
        self.canvas_rtf.fig.tight_layout()
        self.canvas_rtf.draw()


    def export_results(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Results", "audio_codec_analysis.txt", "Text Files (*.txt)")
        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                f.write("--- Audio Codec Comparative Analysis ---\n")
                f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Original Audio File: {self.audio_file_edit.text()}\n")
                f.write(f"Tiny Codec Model: {self.model_path_edit.text()}\n")
                f.write("-" * 40 + "\n\n")

                headers = ["Codec Name", "Bitrate (kbps)", "Latency (ms)", "PESQ (wb)", "STOI", "RTF (Time Factor)", "Status/Error"]
                f.write("{:<30} {:<15} {:<15} {:<10} {:<10} {:<20} {}\n".format(*headers))
                f.write("-" * 120 + "\n")

                for data in self.results_data:
                    name = data.get('name', 'N/A')
                    bitrate = data.get('bitrate', 'N/A')
                    latency = data.get('latency', 'N/A')
                    pesq = f"{data.get('pesq', 0.0):.4f}" if data.get('pesq') is not None else "--"
                    stoi = f"{data.get('stoi', 0.0):.4f}" if data.get('stoi') is not None else "--"
                    rtf = f"{data.get('rtf', 0.0):.3f}" if data.get('rtf') is not None else "--"
                    error = data.get('error', "Success")

                    f.write("{:<30} {:<15} {:<15} {:<10} {:<10} {:<20} {}\n".format(
                        name, bitrate, latency, pesq, stoi, rtf, error
                    ))
                f.write("\n\nNote: All metrics (PESQ, STOI, RTF) are calculated in a streaming-compatible,")
                f.write("\nOverlap-Save (OaS) mode *without* skip connections for an honest comparison.")
            
            self.status_label.setText(f"Status: Results successfully exported to {filepath}")

        except Exception as e:
            self.status_label.setText(f"Status: ERROR - Failed to export results: {e}")
