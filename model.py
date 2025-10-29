
# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# import os
# import struct
# import subprocess
# import sys
# import math

# # --- Global Configuration (Synchronized with 8kbps, 15ms LATENCY Trainer) ---
# SR = 16000
# CHANNELS = 1
# LATENT_DIM = 48 # Synchronized: Reduced from 64
# BLOCKS = 4
# HEADS = 4
# KERNEL_SIZE = 3

# # Synchronized: 2 * 2 * 2 * 3 = 24x downsampling for 15ms Frame Time (16000 / 24 = 666.67 Hz)
# STRIDES = [2, 2, 2, 3] 
# DOWN_FACTOR = np.prod(STRIDES) # Now 24

# # Synchronized: 2 Codebooks (64 entries each) for 8.00 kbps
# NUM_CODEBOOKS = 2
# CODEBOOK_SIZE = 64
# COMMITMENT_COST = 0.5 # Synchronized
# TRANSFORMER_BLOCKS = 2 # Synchronized

# # HOP size (15ms = 240 samples) for low latency streaming I/O chunks
# HOP_SIZE = int(SR / (1000 / 15)) # 240 samples (15ms)

# # Models cache directory
# MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
# if not os.path.exists(MODELS_CACHE_DIR):
#     os.makedirs(MODELS_CACHE_DIR)

# # --- DAC INTEGRATION (Unchanged) ---
# try:
#     import dac
#     DAC_AVAILABLE = True
# except ImportError:
#     DAC_AVAILABLE = False


# # --- CUSTOM TINY VQ-CODEC COMPONENTS (Causal Architecture) ---
# class CausalConvBlock(nn.Module):
#     """Causal Conv1d block matching the architecture from the trainer script."""
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super().__init__()
#         self.padding_amount = kernel_size - 1
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
#         self.norm = nn.GroupNorm(1, out_channels)
#         self.activation = nn.GELU() # Synchronized
#         self.stride = stride

#     def forward(self, x):
#         # Apply padding only to the left (causal)
#         x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
#         x = self.activation(self.norm(self.conv(x)))
#         return x

# class OptimizedTransformerBlock(nn.Module):
#     """Optimized transformer block matching the exact 4-module FFN sequence of the trainer script."""
#     def __init__(self, dim, heads):
#         super().__init__()
#         # NOTE: The training script uses batch_first=True and dropout=0.1.
#         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
        
#         # FIX: The FFN sequence must match the 4 layers in the checkpoint (L, GELU, Dropout, L)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 2), # Layer 0
#             nn.GELU(),               # Layer 1
#             nn.Dropout(0.1),         # Layer 2
#             nn.Linear(dim * 2, dim)  # Layer 3
#         )

#     def forward(self, x):
#         B, C, T = x.shape
#         x_attn = x.transpose(1, 2)
        
#         # Causal mask ensures future tokens are not attended to
#         attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
#         attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
#         x_attn = self.norm1(x_attn + attn_output)
        
#         ffn_output = self.ffn(x_attn)
#         x_attn = self.norm2(x_attn + ffn_output)
        
#         return x_attn.transpose(1, 2)

# class ImprovedVectorQuantizer(nn.Module):
#     """
#     Vector Quantization layer, synchronized with the training script's class definition 
#     to correctly load EMA buffers.
#     """
#     def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost
        
#         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
#         # Include EMA buffers/parameters for compatibility
#         self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
#         self.register_buffer('ema_w', self.embedding.weight.data.clone())
#         self.ema_decay = 0.99


#     def forward(self, inputs):
#         # Flatten input (B, C, T) -> (B*T, C)
#         flat_input = inputs.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        
#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
#                       + torch.sum(self.embedding.weight**2, dim=1)
#                       - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
#         # Find the nearest codebook vector index
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
#         # Create one-hot vectors
#         encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
        
#         # Quantize vector
#         quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], -1).transpose(1, 2)
        
#         # Apply STE
#         quantized = inputs + (quantized - inputs).detach()
        
#         # Loss logic is removed (only needed for training)
        
#         return quantized, encoding_indices

# class TinyTransformerCodec(nn.Module):
#     def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.sr = sr
#         self.downsampling_factor = DOWN_FACTOR
#         self.num_codebooks = NUM_CODEBOOKS

#         # --- Encoder ---
#         self.encoder_convs = nn.ModuleList()
#         in_c = CHANNELS
#         encoder_channels = []
#         for i in range(blocks):
#             out_c = min(latent_dim, 8 * (2**i)) 
#             encoder_channels.append(out_c)
#             stride = STRIDES[i]
#             self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
#             in_c = out_c
        
#         self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

#         # --- Vector Quantization (Using the compatible class) ---
#         self.quantizers = nn.ModuleList([
#             ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
#             for _ in range(NUM_CODEBOOKS)
#         ])
        
#         # --- Transformer (Using the compatible block) ---
#         self.transformer = nn.Sequential(*[
#             OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
#             for _ in range(TRANSFORMER_BLOCKS)
#         ])
#         self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

#         # --- Decoder ---
#         self.decoder_tconvs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
        
#         in_c = latent_dim * NUM_CODEBOOKS
#         for i in range(blocks):
#             idx = blocks - 1 - i
#             stride = STRIDES[idx]
            
#             if idx > 0:
#                 out_c = encoder_channels[idx - 1]
#             else:
#                 out_c = 16
            
#             self.decoder_tconvs.append(
#                 nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
#             )
            
#             if idx > 0:
#                 skip_in_channels = encoder_channels[idx - 1]
#                 # This skip_conv layer still needs to be defined, even if unused,
#                 # to match the saved model checkpoint's keys.
#                 self.skip_convs.append(
#                     nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
#                 )
#             in_c = out_c
        
#         self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

#     @classmethod
#     def load_model(cls, model_path):
#         """Loads the model weights and returns the initialized model."""
#         model = cls()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Trained model not found at path: {model_path}")
            
#         try:
#             checkpoint = torch.load(model_path, map_location=device)
#             state_dict = checkpoint.get('model_state_dict', checkpoint)
#             model.load_state_dict(state_dict)
#             model.to(device)
#             model.eval()
#             print(f"TinyTransformerCodec (VQ-Codec) loaded successfully from {model_path}.")
#             return model
#         except Exception as e:
#             print(f"Error loading model state dict: {e}")
#             raise

#     def encode(self, x):
#         """
#         Encodes audio into quantized latent codes/indices.
#         """
#         x = x.view(x.size(0), CHANNELS, -1)
#         input_length = x.shape[-1]
#         encoder_outputs = []
        
#         # Encoder
#         for layer in self.encoder_convs:
#             x = layer(x)
#             encoder_outputs.append(x)
        
#         # Pre-quantization
#         z_e = self.pre_quant(x)
        
#         # Vector Quantization
#         z_q_list = []
#         indices_list = []
#         z_e_split = z_e.chunk(self.num_codebooks, dim=1)
        
#         for i in range(self.num_codebooks):
#             z_q, indices = self.quantizers[i](z_e_split[i]) # VQ layer returns quantized and indices
#             z_q_list.append(z_q)
#             indices_list.append(indices)
        
#         # Concatenated quantized latent features
#         quantized_codes = torch.cat(z_q_list, dim=1)
        
#         # Pass through transformer
#         codes = self.transformer(quantized_codes)
#         codes = self.post_transformer(codes)
        
#         return codes, indices_list, input_length, encoder_outputs

#     def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
#         """
#         Decodes from VQ-quantized float codes (for evaluation) OR integer indices (for streaming).
#         """
        
#         if isinstance(codes_or_indices_list, list):
#             # Case 1: Decoding from raw integer indices (streaming receiver)
#             indices_list = codes_or_indices_list
#             z_q_list = []
            
#             for i, indices in enumerate(indices_list):
#                 # indices is (T_latent, 1). Squeeze to (T_latent) for embedding lookup
#                 quantized = self.quantizers[i].embedding(indices.squeeze(1))
                
#                 # Add batch dim: (T_latent, C_latent) -> (1, T_latent, C_latent)
#                 quantized = quantized.unsqueeze(0)
#                 # Transpose to (B, C, T): (1, T_latent, C_latent) -> (1, C_latent, T_latent)
#                 quantized = quantized.transpose(1, 2)
                
#                 z_q_list.append(quantized)
            
#             x_pre_transformer = torch.cat(z_q_list, dim=1)
            
#             # Streaming path must also pass through transformer
#             x = self.transformer(x_pre_transformer)
#             x = self.post_transformer(x)
            
#             # In streaming, we have no skip connections
#             encoder_outputs = None 
            
#         elif isinstance(codes_or_indices_list, torch.Tensor):
#             # Case 2: Decoding from the quantized float tensor (evaluation)
#             # This tensor 'codes' is already post-transformer from the encode() step.
#             x = codes_or_indices_list
#             # 'encoder_outputs' is passed from the function arguments
#         else:
#             raise ValueError("Decoding input must be a torch.Tensor (quantized codes) or a list of Tensors (indices).")

#         # Decoder
#         for i, tconv in enumerate(self.decoder_tconvs):
#             x = F.gelu(tconv(x)) # Synchronized activation
            
#             # --- START: CRITICAL FIX ---
#             # This logic is COMMENTED OUT to match your training script.
#             # Your model was trained to reconstruct audio *without* skip connections.
#             # Using them here (even in evaluation) would cause a mismatch.
            
#             # if encoder_outputs and i < len(self.skip_convs):
#             #     encoder_idx = len(self.encoder_convs) - 2 - i
                
#             #     if 0 <= encoder_idx < len(encoder_outputs):
#             #         skip_features = encoder_outputs[encoder_idx]
#             #         min_len = min(skip_features.shape[-1], x.shape[-1])
#             #         skip_features = skip_features[..., :min_len]
#             #         x_trim = x[..., :min_len]
                    
#             #         x_cat = torch.cat([x_trim, skip_features], dim=1)
#             #         x_processed = self.skip_convs[i](x_cat)
                    
#             #         if x.shape[-1] > min_len:
#             #             x = torch.cat([x_processed, x[..., min_len:]], dim=-1)
#             #         else:
#             #             x = x_processed
#             # --- END: CRITICAL FIX ---
        
#         # Final output
#         x = torch.tanh(self.post_decoder_final(x))
        
#         # Match input length
#         if input_length is not None:
#             if x.shape[-1] > input_length:
#                 x = x[..., :input_length]
#             elif x.shape[-1] < input_length:
#                 x = F.pad(x, (0, input_length - x.shape[-1]))
        
#         return x.view(x.size(0), CHANNELS, -1)


# # --- TRADITIONAL CODECS (Unchanged) ---

# class MuLawCodec:
#     """μ-law codec for baseline comparison"""
#     def __init__(self, quantization_channels=256):
#         self.mu = float(quantization_channels - 1)
    
#     def encode(self, x):
#         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
#         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
#         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
#         y = (z_float / self.mu) * 2.0 - 1.0
#         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# class ALawCodec:
#     """A-law codec for baseline comparison"""
#     def __init__(self):
#         self.A = 87.6
    
#     def encode(self, x):
#         a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
#         abs_x = torch.abs(x)
#         encoded = torch.zeros_like(x)
#         cond = abs_x < (1 / self.A)
#         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
#         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
#         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
#         y = (z_float / 127.5) - 1.0
#         abs_y = torch.abs(y)
#         decoded = torch.zeros_like(y)
#         cond = abs_y < (1 / (1 + torch.log(a_t)))
#         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
#         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
#         return decoded.unsqueeze(1)

# class AMRWBCodec:
#     """Simulated AMR-WB (12.65 kbps mode, 20ms frame) using Mu-Law for quantization."""
#     def __init__(self, mode=6): # Use mode 6 (12.65 kbps)
#         self.mu_codec = MuLawCodec()
#         self.sample_rate = 16000
#         self.frame_size = 320 # 20ms at 16kHz
#         self.codec_bitrate = 12.65
        
#     def encode(self, x):
#         return self.mu_codec.encode(x)
    
#     def decode(self, z):
#         return self.mu_codec.decode(z)

# # --- DAC CODEC (Unchanged) ---
# class DACCodec:
#     """Wrapper for Descript Audio Codec (DAC)"""
#     def __init__(self, model_path=None, model_type="16khz"):
#         if not DAC_AVAILABLE:
#             raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_type = model_type
#         print(f"Loading DAC {model_type} model...")
#         try:
#             model_path = dac.utils.download(model_type=model_type)
#             self.model = dac.DAC.load(model_path)
#             print(f"DAC model loaded successfully")
#         except Exception as e:
#             print(f"Error loading DAC model: {e}")
#             raise
#         self.model.to(self.device)
#         self.model.eval()
#         self.sample_rate = 16000 if "16khz" in model_type else 44100
#         self.hop_size = 320
#         self.chunk_size = 320
        
#     def encode(self, audio_tensor):
#         with torch.no_grad():
#             if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
#             audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
#             original_length = audio_tensor.shape[-1]
#             if original_length % self.hop_size != 0:
#                 pad_length = self.hop_size - (original_length % self.hop_size)
#                 audio_tensor = F.pad(audio_tensor, (0, pad_length))
#             _, codes, _, _, _ = self.model.encode(audio_tensor)
#             return codes, original_length # Returns the integer codes tensor and original length
    
#     def decode(self, codes, original_length=None):
#         with torch.no_grad():
#             if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
#             codes = codes.to(self.device)
#             z = self.model.quantizer.from_codes(codes)[0]
#             audio_recon = self.model.decode(z)
#             if original_length is not None and audio_recon.shape[-1] > original_length:
#                 audio_recon = audio_recon[..., :original_length]
#             return audio_recon


# # --- UTILITY FUNCTIONS (Unchanged) ---
# def get_model_cache_info():
#     """Returns information about cached models"""
#     cache_info = {}
#     if os.path.exists(MODELS_CACHE_DIR):
#         for file in os.listdir(MODELS_CACHE_DIR):
#             file_path = os.path.join(MODELS_CACHE_DIR, file)
#             if os.path.isfile(file_path):
#                 file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
#                 cache_info[file] = f"{file_size:.2f} MB"
#     return cache_info

# def clear_model_cache():
#     """Clears all cached models"""
#     import shutil
#     if os.path.exists(MODELS_CACHE_DIR):
#         shutil.rmtree(MODELS_CACHE_DIR)
#         os.makedirs(MODELS_CACHE_DIR)
#         print("Model cache cleared.")


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import struct
import subprocess
import sys
import math

# --- Global Configuration (Synchronized with ~9.3kbps, 15ms LATENCY Trainer) ---
SR = 16000
CHANNELS = 1
LATENT_DIM = 64  # OPTIMIZED: Increased from 48
BLOCKS = 4
HEADS = 8        # OPTIMIZED: Increased from 4
KERNEL_SIZE = 3

# Synchronized: 2 * 2 * 2 * 3 = 24x downsampling (Unchanged)
STRIDES = [2, 2, 2, 3] 
DOWN_FACTOR = np.prod(STRIDES) # 24

# Synchronized: 2 Codebooks (128 entries each) for 9.33 kbps
NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 128 # OPTIMIZED: Increased from 64
COMMITMENT_COST = 0.25  # OPTIMIZED: Reduced from 0.5
TRANSFORMER_BLOCKS = 4  # OPTIMIZED: Increased from 2

# HOP size (15ms = 240 samples) (Unchanged)
HOP_SIZE = int(SR / (1000 / 15)) # 240 samples (15ms)

# Models cache directory
MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)

# --- DAC INTEGRATION (Unchanged) ---
try:
    import dac
    DAC_AVAILABLE = True
except ImportError:
    DAC_AVAILABLE = False


# --- CUSTOM TINY VQ-CODEC COMPONENTS (Causal Architecture) ---
class CausalConvBlock(nn.Module):
    """Causal Conv1d block matching the architecture from the trainer script."""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding_amount = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU() # Synchronized
        self.stride = stride

    def forward(self, x):
        # Apply padding only to the left (causal)
        x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
        x = self.activation(self.norm(self.conv(x)))
        return x

class OptimizedTransformerBlock(nn.Module):
    """Optimized transformer block matching the exact 4-module FFN sequence of the trainer script."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN sequence must match the 4 layers in the checkpoint
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_attn = x.transpose(1, 2)
        
        # Causal mask ensures future tokens are not attended to
        attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
        x_attn = self.norm1(x_attn + attn_output)
        
        ffn_output = self.ffn(x_attn)
        x_attn = self.norm2(x_attn + ffn_output)
        
        return x_attn.transpose(1, 2)

class ImprovedVectorQuantizer(nn.Module):
    """
    Vector Quantization layer, synchronized with the training script's class definition 
    to correctly load EMA buffers.
    """
    def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        # Include EMA buffers/parameters for compatibility
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.ema_decay = 0.99


    def forward(self, inputs):
        # Flatten input (B, C, T) -> (B*T, C)
        flat_input = inputs.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                      + torch.sum(self.embedding.weight**2, dim=1)
                      - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Find the nearest codebook vector index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Create one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize vector
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], -1).transpose(1, 2)
        
        # Apply STE
        quantized = inputs + (quantized - inputs).detach()
        
        # Loss logic is removed (only needed for training)
        
        return quantized, encoding_indices

class TinyTransformerCodec(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
        super().__init__()
        self.latent_dim = latent_dim
        self.sr = sr
        self.downsampling_factor = DOWN_FACTOR
        self.num_codebooks = NUM_CODEBOOKS

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList()
        in_c = CHANNELS
        encoder_channels = []
        for i in range(blocks):
            # OPTIMIZED: Scaled channels based on new LATENT_DIM
            out_c = min(latent_dim, 16 * (2**i)) # Start with 16, up to 64
            encoder_channels.append(out_c)
            stride = STRIDES[i]
            self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
            in_c = out_c
        
        self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

        # --- Vector Quantization (Using the compatible class) ---
        self.quantizers = nn.ModuleList([
            ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
            for _ in range(NUM_CODEBOOKS)
        ])
        
        # --- Transformer (Using the compatible block) ---
        self.transformer = nn.Sequential(*[
            OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
            for _ in range(TRANSFORMER_BLOCKS)
        ])
        self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

        # --- Decoder ---
        self.decoder_tconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        in_c = latent_dim * NUM_CODEBOOKS
        for i in range(blocks):
            idx = blocks - 1 - i
            stride = STRIDES[idx]
            
            if idx > 0:
                out_c = encoder_channels[idx - 1]
            else:
                out_c = 16 # Final channel before output
            
            self.decoder_tconvs.append(
                nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
            )
            
            if idx > 0:
                skip_in_channels = encoder_channels[idx - 1]
                # This skip_conv layer still needs to be defined to match checkpoint keys
                self.skip_convs.append(
                    nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
                )
            in_c = out_c
        
        self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

    @classmethod
    def load_model(cls, model_path):
        """Loads the model weights and returns the initialized model."""
        model = cls()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at path: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"TinyTransformerCodec (VQ-Codec) loaded successfully from {model_path}.")
            return model
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            raise

    def encode(self, x):
        """
        Encodes audio into quantized latent codes/indices.
        """
        x = x.view(x.size(0), CHANNELS, -1)
        input_length = x.shape[-1]
        encoder_outputs = []
        
        # Encoder
        for layer in self.encoder_convs:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Pre-quantization
        z_e = self.pre_quant(x)
        
        # Vector Quantization
        z_q_list = []
        indices_list = []
        z_e_split = z_e.chunk(self.num_codebooks, dim=1)
        
        for i in range(self.num_codebooks):
            z_q, indices = self.quantizers[i](z_e_split[i]) # VQ layer returns quantized and indices
            z_q_list.append(z_q)
            indices_list.append(indices)
        
        # Concatenated quantized latent features
        quantized_codes = torch.cat(z_q_list, dim=1)
        
        # Pass through transformer
        codes = self.transformer(quantized_codes)
        codes = self.post_transformer(codes)
        
        return codes, indices_list, input_length, encoder_outputs

    def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
        """
        Decodes from VQ-quantized float codes (for evaluation) OR integer indices (for streaming).
        """
        
        if isinstance(codes_or_indices_list, list):
            # Case 1: Decoding from raw integer indices (streaming receiver)
            indices_list = codes_or_indices_list
            z_q_list = []
            
            for i, indices in enumerate(indices_list):
                # indices is (T_latent, 1). Squeeze to (T_latent) for embedding lookup
                quantized = self.quantizers[i].embedding(indices.squeeze(1))
                
                # Add batch dim: (T_latent, C_latent) -> (1, T_latent, C_latent)
                quantized = quantized.unsqueeze(0)
                # Transpose to (B, C, T): (1, T_latent, C_latent) -> (1, C_latent, T_latent)
                quantized = quantized.transpose(1, 2)
                
                z_q_list.append(quantized)
            
            x_pre_transformer = torch.cat(z_q_list, dim=1)
            
            # Streaming path must also pass through transformer
            x = self.transformer(x_pre_transformer)
            x = self.post_transformer(x)
            
            # In streaming, we have no skip connections
            encoder_outputs = None 
            
        elif isinstance(codes_or_indices_list, torch.Tensor):
            # Case 2: Decoding from the quantized float tensor (evaluation)
            # This tensor 'codes' is already post-transformer from the encode() step.
            x = codes_or_indices_list
            # 'encoder_outputs' is passed from the function arguments
        else:
            raise ValueError("Decoding input must be a torch.Tensor (quantized codes) or a list of Tensors (indices).")

        # Decoder
        for i, tconv in enumerate(self.decoder_tconvs):
            x = F.gelu(tconv(x)) # Synchronized activation
            
            # --- SKIP CONNECTION LOGIC ---
            # This logic is INTENTIONALLY COMMENTED OUT to match the training
            # script, which trains *without* skip connections for streaming.
            # The model learns to reconstruct audio from the latent codes alone.
            #
            # if encoder_outputs and i < len(self.skip_convs):
            #     encoder_idx = len(self.encoder_convs) - 2 - i
                
            #     if 0 <= encoder_idx < len(encoder_outputs):
            #         skip_features = encoder_outputs[encoder_idx]
            #         min_len = min(skip_features.shape[-1], x.shape[-1])
            #         skip_features = skip_features[..., :min_len]
            #         x_trim = x[..., :min_len]
                    
            #         x_cat = torch.cat([x_trim, skip_features], dim=1)
            #         x_processed = self.skip_convs[i](x_cat)
                    
            #         if x.shape[-1] > min_len:
            #             x = torch.cat([x_processed, x[..., min_len:]], dim=-1)
            #         else:
            #             x = x_processed
            # --- END SKIP LOGIC ---
        
        # Final output
        x = torch.tanh(self.post_decoder_final(x))
        
        # Match input length
        if input_length is not None:
            if x.shape[-1] > input_length:
                x = x[..., :input_length]
            elif x.shape[-1] < input_length:
                x = F.pad(x, (0, input_length - x.shape[-1]))
        
        return x.view(x.size(0), CHANNELS, -1)


# --- TRADITIONAL CODECS (Unchanged) ---

class MuLawCodec:
    """μ-law codec for baseline comparison"""
    def __init__(self, quantization_channels=256):
        self.mu = float(quantization_channels - 1)
    
    def encode(self, x):
        mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
        encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
        return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    
    def decode(self, z):
        z_float = z.to(torch.float32)
        mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
        y = (z_float / self.mu) * 2.0 - 1.0
        return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

class ALawCodec:
    """A-law codec for baseline comparison"""
    def __init__(self):
        self.A = 87.6
    
    def encode(self, x):
        a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
        abs_x = torch.abs(x)
        encoded = torch.zeros_like(x)
        cond = abs_x < (1 / self.A)
        encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
        encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
        return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    
    def decode(self, z):
        z_float = z.to(torch.float32)
        a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
        y = (z_float / 127.5) - 1.0
        abs_y = torch.abs(y)
        decoded = torch.zeros_like(y)
        cond = abs_y < (1 / (1 + torch.log(a_t)))
        decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
        decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
        return decoded.unsqueeze(1)

class AMRWBCodec:
    """Simulated AMR-WB (12.65 kbps mode, 20ms frame) using Mu-Law for quantization."""
    def __init__(self, mode=6): # Use mode 6 (12.65 kbps)
        self.mu_codec = MuLawCodec()
        self.sample_rate = 16000
        self.frame_size = 320 # 20ms at 16kHz
        self.codec_bitrate = 12.65
        
    def encode(self, x):
        return self.mu_codec.encode(x)
    
    def decode(self, z):
        return self.mu_codec.decode(z)

# --- DAC CODEC (Unchanged) ---
class DACCodec:
    """Wrapper for Descript Audio Codec (DAC)"""
    def __init__(self, model_path=None, model_type="16khz"):
        if not DAC_AVAILABLE:
            raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        print(f"Loading DAC {model_type} model...")
        try:
            model_path = dac.utils.download(model_type=model_type)
            self.model = dac.DAC.load(model_path)
            print(f"DAC model loaded successfully")
        except Exception as e:
            print(f"Error loading DAC model: {e}")
            raise
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 16000 if "16khz" in model_type else 44100
        self.hop_size = 320
        self.chunk_size = 320
        
    def encode(self, audio_tensor):
        with torch.no_grad():
            if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
            original_length = audio_tensor.shape[-1]
            if original_length % self.hop_size != 0:
                pad_length = self.hop_size - (original_length % self.hop_size)
                audio_tensor = F.pad(audio_tensor, (0, pad_length))
            _, codes, _, _, _ = self.model.encode(audio_tensor)
            return codes, original_length # Returns the integer codes tensor and original length
    
    def decode(self, codes, original_length=None):
        with torch.no_grad():
            if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
            codes = codes.to(self.device)
            z = self.model.quantizer.from_codes(codes)[0]
            audio_recon = self.model.decode(z)
            if original_length is not None and audio_recon.shape[-1] > original_length:
                audio_recon = audio_recon[..., :original_length]
            return audio_recon


# --- UTILITY FUNCTIONS (Unchanged) ---
def get_model_cache_info():
    """Returns information about cached models"""
    cache_info = {}
    if os.path.exists(MODELS_CACHE_DIR):
        for file in os.listdir(MODELS_CACHE_DIR):
            file_path = os.path.join(MODELS_CACHE_DIR, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                cache_info[file] = f"{file_size:.2f} MB"
    return cache_info

def clear_model_cache():
    """Clears all cached models"""
    import shutil
    if os.path.exists(MODELS_CACHE_DIR):
        shutil.rmtree(MODELS_CACHE_DIR)
        os.makedirs(MODELS_CACHE_DIR)
        print("Model cache cleared.")
