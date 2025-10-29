
# import os
# import threading
# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
#     QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox,
#     QGroupBox, QCheckBox, QListWidget
# )
# from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt

# import socket
# import pyaudio
# import numpy as np
# import torch
# import time
# import librosa
# import struct

# from model import (
#     MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec,
#     HOP_SIZE, DAC_AVAILABLE
# )

# # --- Configuration ---
# BROADCAST_PORT = 37020
# STREAM_PORT = 37021
# DEVICE_ID = "NeuralCodecPC"
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = HOP_SIZE  # 15ms = 240 samples


# # --- Network Discovery Worker ---
# class DiscoveryWorker(QObject):
#     peer_found = pyqtSignal(str, str)

#     def __init__(self):
#         super().__init__()
#         self._running = True

#     def run(self):
#         with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             try:
#                 s.bind(('', BROADCAST_PORT))
#             except OSError as e:
#                 print(f"Discovery: Could not bind to port {BROADCAST_PORT}. {e}")
#                 return
#             s.settimeout(1.0)
#             while self._running:
#                 try:
#                     data, addr = s.recvfrom(1024)
#                     message = data.decode()
#                     if message.startswith(DEVICE_ID):
#                         self.peer_found.emit(message.split(':')[1], addr[0])
#                 except socket.timeout:
#                     continue
#                 except Exception as e:
#                     print(f"Discovery error: {e}")

#     def stop(self):
#         self._running = False


# # --- Audio Streaming Worker ---
# class StreamerWorker(QObject):
#     log_message = pyqtSignal(str)

#     def __init__(self, target_ip, model, model_type_str):
#         super().__init__()
#         self.target_ip = target_ip
#         self.model = model
#         self.model_type_str = model_type_str
#         self.p = pyaudio.PyAudio()
#         self._running = True
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.is_muted = False
#         self.file_playback_path = None
#         self.file_playback_event = threading.Event()

#         # Codec-specific buffering
#         self.send_buffer = []
#         self.recv_buffer = []

#         if isinstance(self.model, DACCodec):
#             self.chunk_size = 320  # DAC uses 20ms chunks
#         elif isinstance(self.model, TinyTransformerCodec):
#             # CRITICAL FIX: Match the training script's CHUNK_DURATION (0.03s * 16000 = 480 samples)
#             self.chunk_size = int(0.03 * RATE)  # 30ms = 480 samples (Synchronized with training script)
#         else:
#             self.chunk_size = CHUNK

#     def run(self):
#         self.sender_thread = threading.Thread(target=self.send_audio, daemon=True)
#         self.receiver_thread = threading.Thread(target=self.receive_audio, daemon=True)
#         self.sender_thread.start()
#         self.receiver_thread.start()

#     def set_mute(self, muted):
#         self.is_muted = muted
#         self.log_message.emit(f"Microphone {'muted' if muted else 'unmuted'}.")

#     def start_file_playback(self, filepath):
#         self.file_playback_path = filepath
#         self.file_playback_event.set()
#         self.log_message.emit(f"Queued file for playback: {filepath}")

#     def stop_file_playback(self, from_thread=False):
#         self.file_playback_path = None
#         if not from_thread:
#             self.file_playback_event.clear()

#     def encode_data(self, data_bytes):
#         try:
#             audio_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#             if self.model is None:
#                 return data_bytes

#             if isinstance(self.model, (MuLawCodec, ALawCodec, AMRWBCodec)):
#                 audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
#                 encoded_tensor = self.model.encode(audio_tensor)
#                 return encoded_tensor.cpu().numpy().tobytes()

#             if isinstance(self.model, (DACCodec, TinyTransformerCodec)):
#                 self.send_buffer.extend(audio_np)
#                 # Check if buffer has enough data for the model's required chunk size
#                 if len(self.send_buffer) >= self.chunk_size:
#                     chunk = np.array(self.send_buffer[:self.chunk_size])
                    
#                     # --- OVERLAP FIX 1: ---
#                     # Instead of clearing the whole buffer, we keep the last CHUNK (240 samples)
#                     # as history (overlap) for the next encode operation.
#                     self.send_buffer = self.send_buffer[CHUNK:] # Was: self.send_buffer[self.chunk_size:]
                    
#                     # Add batch and channel dimensions
#                     chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

#                     with torch.no_grad():
#                         if isinstance(self.model, TinyTransformerCodec):
#                             # VQ-Codec encodes and returns indices list
#                             _, indices_list, orig_len, _ = self.model.encode(chunk_tensor)
#                             # Shape from encode is (B*T_latent, 1) per codebook
#                             # We cat them to (B*T_latent, Num_Codebooks)
#                             codes_np = torch.cat(indices_list, dim=1).cpu().numpy().astype(np.int32)
#                             dtype_int = 0
#                         elif isinstance(self.model, DACCodec):
#                             # DAC returns codes tensor
#                             codes_tensor, orig_len = self.model.encode(chunk_tensor)
#                             codes_np = codes_tensor.cpu().numpy().astype(np.int32)
#                             dtype_int = 0
#                         else:
#                             return None

#                         # Pack with metadata
#                         shape = codes_np.shape
#                         header = struct.pack('I', orig_len) + struct.pack('I', len(shape)) + struct.pack('I', dtype_int)
#                         for s in shape:
#                             header += struct.pack('I', s)
#                         payload = header + codes_np.tobytes()
#                         # self.log_message.emit(f"Encoded {chunk.shape[0]} samples into {len(codes_np.tobytes())} bytes.")
#                         return payload
#                 return None # Not enough data in buffer yet

#             return None
#         except Exception as e:
#             self.log_message.emit(f"Encoding error: {e}")
#             import traceback
#             self.log_message.emit(traceback.format_exc())
#             return None

#     def play_file_audio(self, filepath, s):
#         try:
#             wav, _ = librosa.load(filepath, sr=RATE, mono=True)
#             self.log_message.emit(f"Sending file: {filepath}...")
#             wav_int16 = (wav * 32767.0).astype(np.int16)
#             self.send_buffer = [] # Clear buffer for file playback

#             # Read file in small 15ms chunks (CHUNK)
#             for i in range(0, len(wav_int16), CHUNK):
#                 if not self._running or self.file_playback_path != filepath:
#                     break

#                 chunk_data = wav_int16[i:i + CHUNK]
#                 if len(chunk_data) < CHUNK:
#                     chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')

#                 # encode_data will buffer these small chunks until it has 480 samples
#                 payload = self.encode_data(chunk_data.tobytes())
#                 if payload:
#                     s.sendto(payload, (self.target_ip, STREAM_PORT))
                
#                 # Sleep to simulate real-time playback
#                 time.sleep(float(CHUNK) / RATE)

#             self.log_message.emit("File playback finished.")
#             self.send_buffer = []
#         except Exception as e:
#             self.log_message.emit(f"Error playing file: {e}")
#             self.stop_file_playback(from_thread=True)

#     def send_audio(self):
#         stream = None
#         try:
#             # Use CHUNK (240 samples = 15ms) for low-latency input buffer
#             stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#             with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#                 while self._running:
#                     if self.file_playback_event.is_set():
#                         self.file_playback_event.clear()
#                         if self.file_playback_path:
#                             self.play_file_audio(self.file_playback_path, s)
#                             self.file_playback_path = None
#                         continue

#                     data = stream.read(CHUNK, exception_on_overflow=False)
#                     # encode_data buffers this 15ms chunk and overlaps
#                     payload = self.encode_data(b'\x00' * (CHUNK * 2) if self.is_muted else data)
#                     if payload:
#                         # Send payload only when buffer is full (480 samples) and encoded
#                         s.sendto(payload, (self.target_ip, STREAM_PORT))
#         except Exception as e:
#             if self._running:
#                 self.log_message.emit(f"ERROR in send_audio: {e}")
#         finally:
#             if stream and stream.is_active():
#                 stream.stop_stream()
#                 stream.close()
#             if hasattr(self, 'p'):
#                 self.p.terminate()

#     def receive_audio(self):
#         stream = None
#         try:
#             payload = None
#             with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#                 s.bind(('', STREAM_PORT))
#                 s.settimeout(2.0)
#                 # Use CHUNK (240 samples = 15ms) for low-latency output buffer
#                 stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
#                 self.recv_buffer = []

#                 while self._running:
#                     try:
#                         data, _ = s.recvfrom(65536)
#                     except socket.timeout:
#                         if self._running:
#                             # This causes a momentary silence or pop, but prevents stuttering with old data
#                             self.log_message.emit("Socket timeout, resetting decoder state.")
#                         self.recv_buffer = []
#                         continue

#                     if self.model is None: # Uncompressed
#                         payload = data
#                     else:
#                         try:
#                             if isinstance(self.model, (MuLawCodec, ALawCodec, AMRWBCodec)):
#                                 # Traditional codecs process chunk-by-chunk
#                                 latent_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0)
#                                 decoded_tensor = self.model.decode(latent_tensor)
#                                 payload = (decoded_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16).tobytes()

#                             elif isinstance(self.model, (DACCodec, TinyTransformerCodec)):
#                                 # Neural codecs decode a large chunk (480 samples)
#                                 try:
#                                     offset = 0
#                                     orig_len = struct.unpack('I', data[offset:offset + 4])[0]
#                                     offset += 4
#                                     num_dims = struct.unpack('I', data[offset:offset + 4])[0]
#                                     offset += 4
#                                     dtype_int = struct.unpack('I', data[offset:offset + 4])[0]
#                                     offset += 4
                                    
#                                     # Read shape list
#                                     shape = []
#                                     for i in range(num_dims):
#                                          shape.append(struct.unpack('I', data[offset + 4 * i:offset + 4 * (i + 1)])[0])
#                                     offset += 4 * num_dims
                                    
#                                     codes_bytes = data[offset:]
#                                     dtype = np.int32 # We packed as int32
#                                     codes_np = np.frombuffer(codes_bytes, dtype=dtype)
#                                     if shape:
#                                         codes_np = codes_np.reshape(shape)
#                                     codes_tensor = torch.from_numpy(codes_np).to(self.device)

#                                     with torch.no_grad():
#                                         if isinstance(self.model, TinyTransformerCodec):
#                                             Num_Codebooks = self.model.num_codebooks
                                            
#                                             # codes_tensor shape is (T_latent, Num_Codebooks)
#                                             if codes_tensor.dim() == 3: # Should not happen based on encode, but good check
#                                                 codes_tensor = codes_tensor.permute(0, 2, 1).contiguous().view(-1, Num_Codebooks)
                                            
#                                             # Split (T_latent, Num_Codebooks) into list of [(T_latent, 1), (T_latent, 1), ...]
#                                             indices_list = [idx.view(-1, 1) for idx in codes_tensor.chunk(Num_Codebooks, dim=1)]
                                            
#                                             # model.decode now correctly receives the list and applies transformer
#                                             decoded_tensor = self.model.decode(indices_list, orig_len, encoder_outputs=None)
#                                         elif isinstance(self.model, DACCodec):
#                                             decoded_tensor = self.model.decode(codes_tensor, orig_len)

#                                         # decoded_tensor is (B, C, L), e.g., (1, 1, 480)
#                                         decoded_audio = decoded_tensor.squeeze().cpu().numpy()
                                        
#                                         # --- OVERLAP FIX 2: ---
#                                         # We decoded 480 samples, but the first 240 were just
#                                         # for convolutional history. We only play the *new* 240 samples.
#                                         new_audio_chunk = decoded_audio[-CHUNK:] # Get the last 240 samples
#                                         self.recv_buffer.extend(new_audio_chunk)
                                        
#                                         # Drain buffer in small 15ms (CHUNK) chunks for low-latency playback
#                                         while len(self.recv_buffer) >= CHUNK:
#                                             output_chunk = np.array(self.recv_buffer[:CHUNK])
#                                             self.recv_buffer = self.recv_buffer[CHUNK:]
#                                             payload_to_play = (output_chunk * 32767.0).astype(np.int16).tobytes()
#                                             stream.write(payload_to_play)
                                        
#                                         # Set payload to None so it doesn't write garbage outside this loop
#                                         payload = None 
#                                 except Exception as codec_err:
#                                     self.log_message.emit(f"Codec decode error: {codec_err}")
#                                     import traceback
#                                     self.log_message.emit(traceback.format_exc())
#                                     continue
#                         except Exception as e:
#                             self.log_message.emit(f"Decoding error: {e}")
#                             import traceback
#                             self.log_message.emit(traceback.format_exc())
#                             continue

#                     # Write payload only for Uncompressed or traditional codecs
#                     if payload is not None:
#                         try:
#                             stream.write(payload)
#                         except Exception as e:
#                             self.log_message.emit(f"Stream write error: {e}")
#         except Exception as e:
#             if self._running:
#                 self.log_message.emit(f"ERROR in receive_audio: {e}")
#         finally:
#             if stream and stream.is_active():
#                 stream.stop_stream()
#                 stream.close()
#             if hasattr(self, 'p'):
#                 self.p.terminate()

#     def stop(self):
#         self._running = False
#         self.stop_file_playback()


# # --- StreamingTab GUI ---
# class StreamingTab(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.peers = {}
#         self.streamer_worker = None
#         self.streamer_thread = None
#         self._setup_ui()
#         self._start_discovery()
#         self.model = None

#     def _setup_ui(self):
#         layout = QVBoxLayout(self)
#         layout.setSpacing(15)

#         # --- Configuration UI ---
#         config_group = QGroupBox("Configuration")
#         config_layout = QVBoxLayout(config_group)

#         model_layout = QHBoxLayout()
#         model_layout.addWidget(QLabel("<b>Codec:</b>"))
#         self.model_type_combo = QComboBox()

#         model_items = [
#             "Uncompressed",
#             "μ-Law Codec",
#             "A-Law Codec",
#             "AMR-WB (Simulated, ~12.65 kbps)",
#             "Tiny Transformer Codec (Custom, 8kbps, ~15ms)"
#         ]
#         if DAC_AVAILABLE:
#             model_items.append("DAC Codec (16kHz, 20ms)")

#         self.model_type_combo.addItems(model_items)
#         self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
#         model_layout.addWidget(self.model_type_combo)
#         config_layout.addLayout(model_layout)

#         self.model_path_edit = QLineEdit()
#         self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pth).")
#         self.browse_model_button = QPushButton("Browse...")
#         self.browse_model_button.clicked.connect(self.browse_model)
#         self.load_model_button = QPushButton("Load Model")
#         self.load_model_button.clicked.connect(self.load_model)

#         model_path_layout = QHBoxLayout()
#         model_path_layout.addWidget(self.model_path_edit)
#         model_path_layout.addWidget(self.browse_model_button)
#         model_path_layout.addWidget(self.load_model_button)
#         config_layout.addLayout(model_path_layout)

#         self.bitrate_label = QLabel(f"Codec Info: Select a codec")
#         config_layout.addWidget(self.bitrate_label)

#         # --- File Playback ---
#         self.play_file_path_edit = QLineEdit()
#         self.play_file_path_edit.setPlaceholderText("Path to audio file...")
#         self.browse_play_file_button = QPushButton("Browse...")
#         self.browse_play_file_button.clicked.connect(self.browse_play_file)
#         file_playback_layout = QHBoxLayout()
#         file_playback_layout.addWidget(self.play_file_path_edit)
#         file_playback_layout.addWidget(self.browse_play_file_button)
#         config_layout.addLayout(file_playback_layout)
#         layout.addWidget(config_group)

#         # --- Network Section ---
#         network_group = QGroupBox("Network")
#         network_layout = QVBoxLayout(network_group)
#         network_layout.addWidget(QLabel("<b>Available Peers:</b>"))
#         self.peer_list = QListWidget()
#         self.peer_list.setMaximumHeight(100)
#         self.refresh_button = QPushButton("Refresh List")
#         self.refresh_button.clicked.connect(self.send_broadcast)
#         network_layout.addWidget(self.peer_list)
#         network_layout.addWidget(self.refresh_button)
#         layout.addWidget(network_group)

#         # --- Controls ---
#         controls_group = QGroupBox("Controls")
#         controls_layout = QHBoxLayout(controls_group)
#         self.connect_button = QPushButton("Start Streaming")
#         self.connect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;")
#         self.connect_button.clicked.connect(self.start_streaming)
#         self.disconnect_button = QPushButton("Stop Streaming")
#         self.disconnect_button.setStyleSheet("background-color: #f44336; color: white; padding: 5px; border-radius: 5px;")
#         self.disconnect_button.setEnabled(False)
#         self.disconnect_button.clicked.connect(self.stop_streaming)
#         self.play_file_button = QPushButton("▶ Play File")
#         self.play_file_button.setEnabled(False)
#         self.play_file_button.clicked.connect(self.start_file_playback)
#         self.mute_mic_checkbox = QCheckBox("Mute Mic")
#         self.mute_mic_checkbox.stateChanged.connect(self.on_mute_changed)
#         self.status_label = QLabel("<b>Status:</b> <font color='red'>Disconnected</font>")

#         controls_layout.addWidget(self.connect_button)
#         controls_layout.addWidget(self.disconnect_button)
#         controls_layout.addWidget(self.play_file_button)
#         controls_layout.addStretch()
#         controls_layout.addWidget(self.mute_mic_checkbox)
#         controls_layout.addWidget(self.status_label)
#         layout.addWidget(controls_group)

#         # --- Log Output ---
#         log_group = QGroupBox("Log")
#         log_layout = QVBoxLayout(log_group)
#         self.log_text_edit = QTextEdit()
#         self.log_text_edit.setReadOnly(True)
#         log_layout.addWidget(self.log_text_edit)
#         layout.addWidget(log_group)

#         self.on_model_type_changed(self.model_type_combo.currentText())

#     def log(self, message):
#         # Ensure log updates happen on the main GUI thread
#         if threading.current_thread() != threading.main_thread():
#             # This is a way to safely post to the main thread, but
#             # since we connect signals, this is usually handled.
#             # For direct calls from other threads (which we avoid), this is needed.
#             # However, all our log messages come from signals, so this is safe.
#             pass
#         self.log_text_edit.append(message)
#         self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

#     def on_model_type_changed(self, model_name):
#         is_neural = any(x in model_name for x in ["DAC", "Transformer"])
#         self.model_path_edit.setEnabled(is_neural)
#         self.browse_model_button.setEnabled(is_neural)
#         self.load_model_button.setEnabled(is_neural)

#         if "Uncompressed" in model_name:
#             self.model_path_edit.setText("N/A")
#             self.bitrate_label.setText("Codec Bitrate: 256 kbps")
#             self.model = None # Clear model on selection
#         elif "μ-Law" in model_name or "A-Law" in model_name:
#             self.model_path_edit.setText("N/A (Traditional)")
#             self.bitrate_label.setText("Codec Bitrate: 128 kbps")
#             self.model = None # Clear model on selection
#         elif "AMR-WB" in model_name:
#             self.model_path_edit.setText("N/A (Simulated)")
#             self.bitrate_label.setText("Codec Bitrate: ~12.65 kbps")
#             self.model = None # Clear model on selection
#         elif "DAC" in model_name:
#             self.model_path_edit.setText("(auto-download if not provided)")
#             self.bitrate_label.setText("Codec: DAC | Bitrate: ~8-12 kbps | Latency: 20ms")
#         elif "Tiny Transformer" in model_name:
#             self.model_path_edit.setText("MANDATORY: Path to your trained model (.pth)")
#             self.bitrate_label.setText("Codec: Tiny Transformer | Bitrate: 8.00 kbps | Latency: 15ms")
            
#         # We try to load the model immediately if it's a non-path model.
#         if "N/A" in self.model_path_edit.text():
#             self.load_model()

#     def on_mute_changed(self, state):
#         if self.streamer_worker:
#             self.streamer_worker.set_mute(state == Qt.Checked)

#     def browse_play_file(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
#         if filepath:
#             self.play_file_path_edit.setText(filepath)

#     def browse_model(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
#         if filepath:
#             self.model_path_edit.setText(filepath)

#     def load_model(self):
#         model_name = self.model_type_combo.currentText()
#         self.log(f"Loading model: {model_name} ...")
#         self.model = None
        
#         try:
#             if "Uncompressed" in model_name:
#                 self.model = None
#             elif "μ-Law" in model_name:
#                 self.model = MuLawCodec()
#             elif "A-Law" in model_name:
#                 self.model = ALawCodec()
#             elif "AMR-WB" in model_name:
#                 self.model = AMRWBCodec()
#             elif "DAC" in model_name:
#                 if not DAC_AVAILABLE:
#                     self.log("ERROR: DAC not installed.")
#                     return
#                 self.model = DACCodec(model_type="16khz")
#             elif "Transformer" in model_name:
#                 model_path = self.model_path_edit.text().strip()
#                 if not os.path.exists(model_path):
#                     self.log(f"Error: Model path not found: {model_path}")
#                     return
                
#                 # Use the class method load_model for checkpoints
#                 self.model = TinyTransformerCodec.load_model(model_path)
#             else:
#                 self.model = None

#             # self.play_file_button.setEnabled(True) # Removed: Enable only when streaming
#             self.log(f"✅ Model loaded successfully: {model_name}")
#         except Exception as e:
#             self.model = None
#             self.log(f"❌ Failed to load model: {e}")

#     def _start_discovery(self):
#         self.discovery_thread = QThread()
#         self.discovery_worker = DiscoveryWorker()
#         self.discovery_worker.moveToThread(self.discovery_thread)
#         self.discovery_thread.started.connect(self.discovery_worker.run)
#         self.discovery_worker.peer_found.connect(self.on_peer_found)
#         self.discovery_thread.start()
#         self.send_broadcast()

#     def send_broadcast(self):
#         threading.Thread(target=self._send_broadcast_thread, daemon=True).start()

#     def _send_broadcast_thread(self):
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#                 s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
#                 message = f"{DEVICE_ID}:{socket.gethostname()}".encode()
#                 s.sendto(message, ('<broadcast>', BROADCAST_PORT))
#             # self.log("Broadcast sent to discover peers...") # Too noisy
#         except Exception as e:
#             self.log(f"Error sending broadcast: {e}")


#     def on_peer_found(self, name, ip):
#         # Use IP as the key to prevent duplicates
#         if ip not in self.peers.values():
#             self.peers[name] = ip
#             self.peer_list.addItem(f"{name} ({ip})")
#             self.log(f"Discovered peer: {name} at {ip}")

#     def start_streaming(self):
#         selected_items = self.peer_list.selectedItems()
#         if not selected_items:
#             self.log("Please select a peer to stream to.")
#             return
        
#         target_text = selected_items[0].text()
#         try:
#             target_ip = target_text.split('(')[1].strip(')')
#         except IndexError:
#             self.log(f"Error: Could not parse IP from selected peer '{target_text}'.")
#             return
            
#         self.log(f"Starting streaming to {target_ip}...")

#         if self.model is None and "Uncompressed" not in self.model_type_combo.currentText():
#             self.log("ERROR: Model is not loaded. Please load the model before streaming.")
#             return

#         self.streamer_thread = QThread()
#         self.streamer_worker = StreamerWorker(target_ip, self.model, self.model_type_combo.currentText())
#         self.streamer_worker.moveToThread(self.streamer_thread)
#         self.streamer_worker.log_message.connect(self.log) # Connect log signal
#         self.streamer_thread.started.connect(self.streamer_worker.run)
#         self.streamer_thread.start()

#         self.status_label.setText("<b>Status:</b> <font color='green'>Streaming</font>")
#         self.connect_button.setEnabled(False)
#         self.disconnect_button.setEnabled(True)
#         self.play_file_button.setEnabled(True)
#         self.model_type_combo.setEnabled(False) # Disable model switching while streaming
#         self.load_model_button.setEnabled(False)
#         self.log("✅ Streaming started successfully.")

#     def stop_streaming(self):
#         if self.streamer_worker:
#             self.streamer_worker.stop()
#             self.streamer_thread.quit()
#             self.streamer_thread.wait()
#             self.streamer_worker = None
#         self.status_label.setText("<b>Status:</b> <font color='red'>Disconnected</font>")
#         self.connect_button.setEnabled(True)
#         self.disconnect_button.setEnabled(False)
#         self.play_file_button.setEnabled(False)
#         self.model_type_combo.setEnabled(True) # Re-enable model switching
#         self.on_model_type_changed(self.model_type_combo.currentText()) # Re-enable load button if needed
#         self.log("⛔ Streaming stopped.")

#     def start_file_playback(self):
#         filepath = self.play_file_path_edit.text().strip()
#         if not os.path.exists(filepath):
#             self.log("Error: File not found.")
#             return
#         if self.streamer_worker:
#             self.streamer_worker.start_file_playback(filepath)
#             self.log(f"Playing file: {filepath}")
#         else:
#             self.log("Error: Streaming not active.")

import os
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox,
    QGroupBox, QCheckBox, QListWidget
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt

import socket
import pyaudio
import numpy as np
import torch
import time
import librosa
import struct

from model import (
    MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec,
    HOP_SIZE, DAC_AVAILABLE
)

# --- Configuration ---
BROADCAST_PORT = 37020
STREAM_PORT = 37021
DEVICE_ID = "NeuralCodecPC"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = HOP_SIZE  # 15ms = 240 samples


# --- Network Discovery Worker ---
class DiscoveryWorker(QObject):
    peer_found = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('', BROADCAST_PORT))
            except OSError as e:
                print(f"Discovery: Could not bind to port {BROADCAST_PORT}. {e}")
                return
            s.settimeout(1.0)
            while self._running:
                try:
                    data, addr = s.recvfrom(1024)
                    message = data.decode()
                    if message.startswith(DEVICE_ID):
                        self.peer_found.emit(message.split(':')[1], addr[0])
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Discovery error: {e}")

    def stop(self):
        self._running = False


# --- Audio Streaming Worker ---
class StreamerWorker(QObject):
    log_message = pyqtSignal(str)

    def __init__(self, target_ip, model, model_type_str):
        super().__init__()
        self.target_ip = target_ip
        self.model = model
        self.model_type_str = model_type_str
        self.p = pyaudio.PyAudio()
        self._running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_muted = False
        self.file_playback_path = None
        self.file_playback_event = threading.Event()

        # Codec-specific buffering
        self.send_buffer = []
        self.recv_buffer = []

        if isinstance(self.model, DACCodec):
            self.chunk_size = 320  # DAC uses 20ms chunks (320 samples)
        elif isinstance(self.model, TinyTransformerCodec):
            self.chunk_size = int(0.03 * RATE)  # 30ms = 480 samples
        else:
            self.chunk_size = CHUNK # 15ms = 240 samples

    def run(self):
        self.sender_thread = threading.Thread(target=self.send_audio, daemon=True)
        self.receiver_thread = threading.Thread(target=self.receive_audio, daemon=True)
        self.sender_thread.start()
        self.receiver_thread.start()

    def set_mute(self, muted):
        self.is_muted = muted
        self.log_message.emit(f"Microphone {'muted' if muted else 'unmuted'}.")

    def start_file_playback(self, filepath):
        self.file_playback_path = filepath
        self.file_playback_event.set()
        self.log_message.emit(f"Queued file for playback: {filepath}")

    def stop_file_playback(self, from_thread=False):
        self.file_playback_path = None
        if not from_thread:
            self.file_playback_event.clear()

    def encode_data(self, data_bytes):
        try:
            audio_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if self.model is None:
                return data_bytes

            if isinstance(self.model, (MuLawCodec, ALawCodec, AMRWBCodec)):
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                encoded_tensor = self.model.encode(audio_tensor)
                return encoded_tensor.cpu().numpy().tobytes()

            self.send_buffer.extend(audio_np)
            
            # Check if buffer has enough data for the model's required chunk size
            if len(self.send_buffer) >= self.chunk_size:
                chunk = np.array(self.send_buffer[:self.chunk_size])
                
                if isinstance(self.model, TinyTransformerCodec):
                    # --- TinyTransformer Overlap-Add Logic (Correct) ---
                    # Keep the last CHUNK (240 samples) as history
                    self.send_buffer = self.send_buffer[CHUNK:] 
                elif isinstance(self.model, DACCodec):
                    # --- DAC No-Overlap Logic (FIXED) ---
                    # Process exactly 320 samples and clear buffer
                    self.send_buffer = []

                # Add batch and channel dimensions
                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

                with torch.no_grad():
                    if isinstance(self.model, TinyTransformerCodec):
                        # VQ-Codec encodes and returns indices list
                        _, indices_list, orig_len, _ = self.model.encode(chunk_tensor)
                        codes_np = torch.cat(indices_list, dim=1).cpu().numpy().astype(np.int32)
                    elif isinstance(self.model, DACCodec):
                        # DAC returns codes tensor
                        codes_tensor, orig_len = self.model.encode(chunk_tensor)
                        codes_np = codes_tensor.cpu().numpy().astype(np.int32)
                    
                    dtype_int = 0 # Placeholder, we always use int32 for codes
                    # Pack with metadata
                    shape = codes_np.shape
                    header = struct.pack('I', orig_len) + struct.pack('I', len(shape)) + struct.pack('I', dtype_int)
                    for s in shape:
                        header += struct.pack('I', s)
                    payload = header + codes_np.tobytes()
                    return payload
            
            return None # Not enough data in buffer yet

        except Exception as e:
            self.log_message.emit(f"Encoding error: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            return None

    def play_file_audio(self, filepath, s):
        try:
            wav, _ = librosa.load(filepath, sr=RATE, mono=True)
            self.log_message.emit(f"Sending file: {filepath}...")
            wav_int16 = (wav * 32767.0).astype(np.int16)
            self.send_buffer = [] # Clear buffer for file playback

            # Read file in small 15ms chunks (CHUNK)
            for i in range(0, len(wav_int16), CHUNK):
                if not self._running or self.file_playback_path != filepath:
                    break

                chunk_data = wav_int16[i:i + CHUNK]
                if len(chunk_data) < CHUNK:
                    chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')

                # encode_data will buffer these small chunks
                payload = self.encode_data(chunk_data.tobytes())
                if payload:
                    s.sendto(payload, (self.target_ip, STREAM_PORT))
                
                # Sleep to simulate real-time playback
                time.sleep(float(CHUNK) / RATE)

            self.log_message.emit("File playback finished.")
            self.send_buffer = []
        except Exception as e:
            self.log_message.emit(f"Error playing file: {e}")
            self.stop_file_playback(from_thread=True)

    def send_audio(self):
        stream = None
        try:
            # Use CHUNK (240 samples = 15ms) for low-latency input buffer
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                while self._running:
                    if self.file_playback_event.is_set():
                        self.file_playback_event.clear()
                        if self.file_playback_path:
                            self.play_file_audio(self.file_playback_path, s)
                            self.file_playback_path = None
                        continue

                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # encode_data buffers this 15ms chunk
                    payload = self.encode_data(b'\x00' * (CHUNK * 2) if self.is_muted else data)
                    if payload:
                        # Send payload only when buffer is full and encoded
                        s.sendto(payload, (self.target_ip, STREAM_PORT))
        except Exception as e:
            if self._running:
                self.log_message.emit(f"ERROR in send_audio: {e}")
        finally:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
            if hasattr(self, 'p'):
                self.p.terminate()

    def receive_audio(self):
        stream = None
        try:
            payload = None
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('', STREAM_PORT))
                s.settimeout(2.0)
                # Use CHUNK (240 samples = 15ms) for low-latency output buffer
                stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
                self.recv_buffer = []

                while self._running:
                    try:
                        data, _ = s.recvfrom(65536)
                    except socket.timeout:
                        if self._running:
                            self.log_message.emit("Socket timeout, resetting decoder state.")
                        self.recv_buffer = []
                        continue

                    if self.model is None: # Uncompressed
                        payload = data
                    else:
                        try:
                            if isinstance(self.model, (MuLawCodec, ALawCodec, AMRWBCodec)):
                                # Traditional codecs process chunk-by-chunk
                                latent_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0)
                                decoded_tensor = self.model.decode(latent_tensor)
                                payload = (decoded_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16).tobytes()

                            elif isinstance(self.model, (DACCodec, TinyTransformerCodec)):
                                # --- START NEURAL CODEC DECODE ---
                                try:
                                    offset = 0
                                    orig_len = struct.unpack('I', data[offset:offset + 4])[0]
                                    offset += 4
                                    num_dims = struct.unpack('I', data[offset:offset + 4])[0]
                                    offset += 4
                                    dtype_int = struct.unpack('I', data[offset:offset + 4])[0]
                                    offset += 4
                                    
                                    shape = []
                                    for i in range(num_dims):
                                         shape.append(struct.unpack('I', data[offset + 4 * i:offset + 4 * (i + 1)])[0])
                                    offset += 4 * num_dims
                                    
                                    codes_bytes = data[offset:]
                                    dtype = np.int32 # We packed as int32
                                    codes_np = np.frombuffer(codes_bytes, dtype=dtype)
                                    if shape:
                                        codes_np = codes_np.reshape(shape)
                                    codes_tensor = torch.from_numpy(codes_np).to(self.device)

                                    with torch.no_grad():
                                        if isinstance(self.model, TinyTransformerCodec):
                                            # --- Tiny Transformer Overlap-Save Logic (Correct) ---
                                            Num_Codebooks = self.model.num_codebooks
                                            indices_list = [idx.view(-1, 1) for idx in codes_tensor.chunk(Num_Codebooks, dim=1)]
                                            decoded_tensor = self.model.decode(indices_list, orig_len, encoder_outputs=None)
                                            
                                            decoded_audio = decoded_tensor.squeeze().cpu().numpy()
                                            new_audio_chunk = decoded_audio[-CHUNK:] # Get the last 240 samples
                                            self.recv_buffer.extend(new_audio_chunk)
                                            
                                            # Drain buffer in 15ms (CHUNK) chunks
                                            while len(self.recv_buffer) >= CHUNK:
                                                output_chunk = np.array(self.recv_buffer[:CHUNK])
                                                self.recv_buffer = self.recv_buffer[CHUNK:]
                                                payload_to_play = (output_chunk * 32767.0).astype(np.int16).tobytes()
                                                stream.write(payload_to_play)
                                            
                                        elif isinstance(self.model, DACCodec):
                                            # --- DAC No-Overlap Logic (FIXED) ---
                                            decoded_tensor = self.model.decode(codes_tensor, orig_len)
                                            decoded_audio = decoded_tensor.squeeze().cpu().numpy()
                                            
                                            # Convert to int16 bytes
                                            payload_to_play = (decoded_audio * 32767.0).astype(np.int16).tobytes()
                                            
                                            # Write the entire 320-sample (20ms) chunk directly.
                                            stream.write(payload_to_play)

                                    # Set payload to None so it doesn't write garbage outside this loop
                                    payload = None
                                except Exception as codec_err:
                                    self.log_message.emit(f"Codec decode error: {codec_err}")
                                    import traceback
                                    self.log_message.emit(traceback.format_exc())
                                    continue
                                # --- END NEURAL CODEC DECODE ---
                        except Exception as e:
                            self.log_message.emit(f"Decoding error: {e}")
                            import traceback
                            self.log_message.emit(traceback.format_exc())
                            continue

                    # Write payload only for Uncompressed or traditional codecs
                    if payload is not None:
                        try:
                            stream.write(payload)
                        except Exception as e:
                            self.log_message.emit(f"Stream write error: {e}")
        except Exception as e:
            if self._running:
                self.log_message.emit(f"ERROR in receive_audio: {e}")
        finally:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
            if hasattr(self, 'p'):
                self.p.terminate()

    def stop(self):
        self._running = False
        self.stop_file_playback()


# --- StreamingTab GUI ---
class StreamingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.peers = {}
        self.streamer_worker = None
        self.streamer_thread = None
        self._setup_ui()
        self._start_discovery()
        self.model = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # --- Configuration UI ---
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("<b>Codec:</b>"))
        self.model_type_combo = QComboBox()

        model_items = [
            "Uncompressed",
            "μ-Law Codec",
            "A-Law Codec",
            "AMR-WB (Simulated, ~12.65 kbps)",
            "Tiny Transformer Codec (Custom, ~9.3kbps, ~15ms)" # UPDATED
        ]
        if DAC_AVAILABLE:
            model_items.append("DAC Codec (16kHz, 20ms)")

        self.model_type_combo.addItems(model_items)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        config_layout.addLayout(model_layout)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pth).")
        self.browse_model_button = QPushButton("Browse...")
        self.browse_model_button.clicked.connect(self.browse_model)
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.browse_model_button)
        model_path_layout.addWidget(self.load_model_button)
        config_layout.addLayout(model_path_layout)

        self.bitrate_label = QLabel(f"Codec Info: Select a codec")
        config_layout.addWidget(self.bitrate_label)

        # --- File Playback ---
        self.play_file_path_edit = QLineEdit()
        self.play_file_path_edit.setPlaceholderText("Path to audio file...")
        self.browse_play_file_button = QPushButton("Browse...")
        self.browse_play_file_button.clicked.connect(self.browse_play_file)
        file_playback_layout = QHBoxLayout()
        file_playback_layout.addWidget(self.play_file_path_edit)
        file_playback_layout.addWidget(self.browse_play_file_button)
        config_layout.addLayout(file_playback_layout)
        layout.addWidget(config_group)

        # --- Network Section ---
        network_group = QGroupBox("Network")
        network_layout = QVBoxLayout(network_group)
        network_layout.addWidget(QLabel("<b>Available Peers:</b>"))
        self.peer_list = QListWidget()
        self.peer_list.setMaximumHeight(100)
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.send_broadcast)
        network_layout.addWidget(self.peer_list)
        network_layout.addWidget(self.refresh_button)
        layout.addWidget(network_group)

        # --- Controls ---
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.connect_button = QPushButton("Start Streaming")
        self.connect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;")
        self.connect_button.clicked.connect(self.start_streaming)
        self.disconnect_button = QPushButton("Stop Streaming")
        self.disconnect_button.setStyleSheet("background-color: #f44336; color: white; padding: 5px; border-radius: 5px;")
        self.disconnect_button.setEnabled(False)
        self.disconnect_button.clicked.connect(self.stop_streaming)
        self.play_file_button = QPushButton("▶ Play File")
        self.play_file_button.setEnabled(False)
        self.play_file_button.clicked.connect(self.start_file_playback)
        self.mute_mic_checkbox = QCheckBox("Mute Mic")
        self.mute_mic_checkbox.stateChanged.connect(self.on_mute_changed)
        self.status_label = QLabel("<b>Status:</b> <font color='red'>Disconnected</font>")

        controls_layout.addWidget(self.connect_button)
        controls_layout.addWidget(self.disconnect_button)
        controls_layout.addWidget(self.play_file_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.mute_mic_checkbox)
        controls_layout.addWidget(self.status_label)
        layout.addWidget(controls_group)

        # --- Log Output ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        layout.addWidget(log_group)

        self.on_model_type_changed(self.model_type_combo.currentText())

    def log(self, message):
        # This method is designed to be thread-safe as it's connected
        # via a Qt Signal, which marshals the call to the main GUI thread.
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_model_type_changed(self, model_name):
        is_neural = any(x in model_name for x in ["DAC", "Transformer"])
        self.model_path_edit.setEnabled(is_neural)
        self.browse_model_button.setEnabled(is_neural)
        self.load_model_button.setEnabled(is_neural)

        if "Uncompressed" in model_name:
            self.model_path_edit.setText("N/A")
            self.bitrate_label.setText("Codec Bitrate: 256 kbps")
            self.model = None # Clear model on selection
        elif "μ-Law" in model_name or "A-Law" in model_name:
            self.model_path_edit.setText("N/A (Traditional)")
            self.bitrate_label.setText("Codec Bitrate: 128 kbps")
            self.model = None # Clear model on selection
        elif "AMR-WB" in model_name:
            self.model_path_edit.setText("N/A (Simulated)")
            self.bitrate_label.setText("Codec Bitrate: ~12.65 kbps")
            self.model = None # Clear model on selection
        elif "DAC" in model_name:
            self.model_path_edit.setText("(auto-download if not provided)")
            self.bitrate_label.setText("Codec: DAC | Bitrate: ~8-12 kbps | Latency: 20ms")
        elif "Tiny Transformer" in model_name:
            self.model_path_edit.setText("MANDATORY: Path to your trained model (.pth)")
            self.bitrate_label.setText("Codec: Tiny Transformer | Bitrate: ~9.33 kbps | Latency: 15ms")
            
        if "N/A" in self.model_path_edit.text():
            self.load_model()

    def on_mute_changed(self, state):
        if self.streamer_worker:
            self.streamer_worker.set_mute(state == Qt.Checked)

    def browse_play_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if filepath:
            self.play_file_path_edit.setText(filepath)

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
        if filepath:
            self.model_path_edit.setText(filepath)

    def load_model(self):
        model_name = self.model_type_combo.currentText()
        self.log(f"Loading model: {model_name} ...")
        self.model = None
        
        try:
            if "Uncompressed" in model_name:
                self.model = None
            elif "μ-Law" in model_name:
                self.model = MuLawCodec()
            elif "A-Law" in model_name:
                self.model = ALawCodec()
            elif "AMR-WB" in model_name:
                self.model = AMRWBCodec()
            elif "DAC" in model_name:
                if not DAC_AVAILABLE:
                    self.log("ERROR: DAC not installed.")
                    return
                self.model = DACCodec(model_type="16khz")
            elif "Transformer" in model_name:
                model_path = self.model_path_edit.text().strip()
                if not os.path.exists(model_path):
                    self.log(f"Error: Model path not found: {model_path}")
                    return
                
                # Use the class method load_model for checkpoints
                self.model = TinyTransformerCodec.load_model(model_path)
            else:
                self.model = None

            self.log(f"✅ Model loaded successfully: {model_name}")
        except Exception as e:
            self.model = None
            self.log(f"❌ Failed to load model: {e}")

    def _start_discovery(self):
        self.discovery_thread = QThread()
        self.discovery_worker = DiscoveryWorker()
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_worker.peer_found.connect(self.on_peer_found)
        self.discovery_thread.start()
        self.send_broadcast()

    def send_broadcast(self):
        threading.Thread(target=self._send_broadcast_thread, daemon=True).start()

    def _send_broadcast_thread(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                message = f"{DEVICE_ID}:{socket.gethostname()}".encode()
                s.sendto(message, ('<broadcast>', BROADCAST_PORT))
        except Exception as e:
            self.log(f"Error sending broadcast: {e}")


    def on_peer_found(self, name, ip):
        if ip not in self.peers.values():
            self.peers[name] = ip
            self.peer_list.addItem(f"{name} ({ip})")
            self.log(f"Discovered peer: {name} at {ip}")

    def start_streaming(self):
        selected_items = self.peer_list.selectedItems()
        if not selected_items:
            self.log("Please select a peer to stream to.")
            return
        
        target_text = selected_items[0].text()
        try:
            target_ip = target_text.split('(')[1].strip(')')
        except IndexError:
            self.log(f"Error: Could not parse IP from selected peer '{target_text}'.")
            return
            
        self.log(f"Starting streaming to {target_ip}...")

        if self.model is None and "Uncompressed" not in self.model_type_combo.currentText():
            self.log("ERROR: Model is not loaded. Please load the model before streaming.")
            return

        self.streamer_thread = QThread()
        self.streamer_worker = StreamerWorker(target_ip, self.model, self.model_type_combo.currentText())
        self.streamer_worker.moveToThread(self.streamer_thread)
        self.streamer_worker.log_message.connect(self.log) # Connect log signal
        self.streamer_thread.started.connect(self.streamer_worker.run)
        self.streamer_thread.start()

        self.status_label.setText("<b>Status:</b> <font color='green'>Streaming</font>")
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.play_file_button.setEnabled(True)
        self.model_type_combo.setEnabled(False) 
        self.load_model_button.setEnabled(False)
        self.log("✅ Streaming started successfully.")

    def stop_streaming(self):
        if self.streamer_worker:
            self.streamer_worker.stop()
            self.streamer_thread.quit()
            self.streamer_thread.wait()
            self.streamer_worker = None
        self.status_label.setText("<b>Status:</b> <font color='red'>Disconnected</font>")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.play_file_button.setEnabled(False)
        self.model_type_combo.setEnabled(True) 
        self.on_model_type_changed(self.model_type_combo.currentText()) 
        self.log("⛔ Streaming stopped.")

    def start_file_playback(self):
        filepath = self.play_file_path_edit.text().strip()
        if not os.path.exists(filepath):
            self.log("Error: File not found.")
            return
        if self.streamer_worker:
            self.streamer_worker.start_file_playback(filepath)
            self.log(f"Playing file: {filepath}")
        else:
            self.log("Error: Streaming not active.")
