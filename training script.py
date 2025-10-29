################################################################################
#
# A-to-Z Tiny Transformer Codec Training Script (Google Colab)
#
# This script trains the TinyTransformerCodec model from your `model.py` file.
#
# Features:
# 1. Auto-installs all required packages.
# 2. Connects to Google Drive to save/load checkpoints.
# 3. Resumes training automatically from the latest checkpoint.
# 4. Uses a Multi-Resolution STFT (MR-STFT) Loss for high perceptual quality.
# 5. Validates using the *exact* streaming Overlap-Save (OaS) method
#    from your application to get accurate PESQ/STOI scores.
# 6. Saves the best-performing model based on PESQ score.
#
################################################################################

print("Starting training script...")

# === 1. SETUP AND PACKAGE INSTALLATION ===
import subprocess
import sys
import os

def install_packages():
    """Installs all required packages for training."""
    print("Installing required packages...")
    packages = [
        "torch", "torchaudio", "numpy", "librosa",
        "pesq[speechmetrics]", "pystoi", "einops", "tqdm",
        "wandb", "soundfile"
    ]
    for package in packages:
        try:
            print(f"Installing {package}...")
            if package == "pesq[speechmetrics]":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pesq[speechmetrics]"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Failed to install {package}. Error: {e}")
            print("Please try installing manually. Script will attempt to continue...")
        except Exception as e:
            print(f"An unexpected error occurred during installation of {package}. Error: {e}")

# Uncomment the line below to run the installation
# install_packages() # <-- Keep this commented if packages are already installed
print("Package check complete.")

# === 2. IMPORTS ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torchaudio
    from torchaudio.transforms import Resample

    import numpy as np
    import librosa
    import soundfile as sf
    from pesq import pesq
    from pystoi import stoi
    from tqdm import tqdm
    import math
    import time
    import random
    import wandb as wandb_module # <-- 1. Renamed import
    from google.colab import drive
except ImportError as e:
    print(f"FATAL ERROR: A required package is missing: {e}")
    print("Please run the `install_packages()` function by uncommenting the line above.")
    sys.exit(1)

print("All libraries imported successfully.")

# === 3. GOOGLE DRIVE & PATH SETUP ===
def mount_drive_and_setup_paths():
    """Mounts Google Drive and creates checkpoint directories."""
    print("Mounting Google Drive at /content/drive...")
    try:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        print("Checkpoints will be saved locally to /content/checkpoints.")
        print("WARNING: These will be DELETED when the Colab instance resets.")

    base_dir = "/content/drive/MyDrive" if os.path.exists("/content/drive") else "/content"
    checkpoint_dir = os.path.join(base_dir, "TinyTransformer_Checkpoints")
    dataset_dir = "/content/data" # Local dir for dataset

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Dataset directory: {dataset_dir}")

    return checkpoint_dir, dataset_dir

CHECKPOINT_DIR, DATASET_DIR = mount_drive_and_setup_paths()

# === 4. MODEL HYPERPARAMETERS (Copied from model.py) ===
# These *must* match your model.py file exactly.
SR = 16000
CHANNELS = 1
LATENT_DIM = 64
BLOCKS = 4
HEADS = 8
KERNEL_SIZE = 3
STRIDES = [2, 2, 2, 3]
DOWN_FACTOR = np.prod(STRIDES) # 24
NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 128
COMMITMENT_COST = 0.25
TRANSFORMER_BLOCKS = 4

# Training-specific Hyperparameters
# This is the 30ms (480 sample) window your app uses for OaS
TRAIN_WINDOW_SAMPLES = int(0.03 * SR) # 480 samples
# This is the 15ms (240 sample) hop your app uses
STREAMING_HOP_SAMPLES = int(0.015 * SR) # 240 samples

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
VALIDATE_EVERY_N_EPOCHS = 1
GRADIENT_CLIP_VAL = 1.0

# === 5. MODEL DEFINITION (Adapted from model.py for Training) ===
# We copy the model classes and modify the VQ and main model's
# `forward` pass to return the VQ loss for training.

class CausalConvBlock(nn.Module):
    """Causal Conv1d block matching the architecture from the trainer script."""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding_amount = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self.stride = stride

    def forward(self, x):
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
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_attn = x.transpose(1, 2)
        attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
        x_attn = self.norm1(x_attn + attn_output)
        ffn_output = self.ffn(x_attn)
        x_attn = self.norm2(x_attn + ffn_output)
        return x_attn.transpose(1, 2)

class ImprovedVectorQuantizer(nn.Module):
    """
    Vector Quantization layer from model.py, *modified* to compute
    and return the VQ commitment loss during training.
    """
    def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # EMA buffers for codebook updates (from your model.py)
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

        # --- VQ Loss Calculation (for training) ---
        loss_e = F.mse_loss(quantized.detach(), inputs) # Encoder loss
        loss_q = F.mse_loss(quantized, inputs.detach()) # Codebook loss
        vq_loss = loss_e + self.commitment_cost * loss_q

        # EMA Updates (only in training mode)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                    (1 - self.ema_decay) * torch.sum(encodings, 0)

            # Laplace smoothing to avoid zero counts
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw

            # Update embedding weights
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Apply STE (Straight-Through Estimator)
        quantized = inputs + (quantized - inputs).detach()

        return quantized, encoding_indices, vq_loss

class TinyTransformerCodec(nn.Module):
    """
    Main model from model.py, with a single `forward` pass
    that returns the reconstructed audio and the VQ loss.
    """
    def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
        super().__init__()
        self.latent_dim = latent_dim
        self.sr = sr
        self.downsampling_factor = DOWN_FACTOR
        self.num_codebooks = NUM_CODEBOOKS

        # --- Encoder (from model.py) ---
        self.encoder_convs = nn.ModuleList()
        in_c = CHANNELS
        encoder_channels = []
        for i in range(blocks):
            out_c = min(latent_dim, 16 * (2**i))
            encoder_channels.append(out_c)
            stride = STRIDES[i]
            self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
            in_c = out_c

        self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

        # --- Vector Quantization (Using the training-ready version) ---
        self.quantizers = nn.ModuleList([
            ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
            for _ in range(NUM_CODEBOOKS)
        ])

        # --- Transformer (from model.py) ---
        self.transformer = nn.Sequential(*[
            OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
            for _ in range(TRANSFORMER_BLOCKS)
        ])
        self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

        # --- Decoder (from model.py) ---
        self.decoder_tconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList() # Must be defined to load state_dict, even if unused

        in_c = latent_dim * NUM_CODEBOOKS
        for i in range(blocks):
            idx = blocks - 1 - i
            stride = STRIDES[idx]
            if idx > 0:
                out_c = encoder_channels[idx - 1]
            else:
                out_c = 16
            self.decoder_tconvs.append(
                nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
            )
            if idx > 0:
                skip_in_channels = encoder_channels[idx - 1]
                self.skip_convs.append(
                    nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
                )
            in_c = out_c

        self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

    def forward(self, x):
        """
        Training forward pass.
        Returns: reconstructed_audio, total_vq_loss
        """
        x = x.view(x.size(0), CHANNELS, -1)
        input_length = x.shape[-1]

        # --- Encoder ---
        for layer in self.encoder_convs:
            x = layer(x)

        z_e = self.pre_quant(x)

        # --- Vector Quantization ---
        z_q_list = []
        total_vq_loss = 0
        z_e_split = z_e.chunk(self.num_codebooks, dim=1)

        for i in range(self.num_codebooks):
            # Use the VQ that returns loss
            z_q, indices, vq_loss = self.quantizers[i](z_e_split[i])
            z_q_list.append(z_q)
            total_vq_loss = total_vq_loss + vq_loss

        quantized_codes = torch.cat(z_q_list, dim=1)

        # --- Transformer ---
        codes = self.transformer(quantized_codes)
        x = self.post_transformer(codes) # Use 'x' as the variable for the decoder

        # --- Decoder ---
        for i, tconv in enumerate(self.decoder_tconvs):
            x = F.gelu(tconv(x))
            #
            # CRITICAL: Skip connections are INTENTIONALLY disabled
            # to match your model.py's streaming logic.
            #

        x = torch.tanh(self.post_decoder_final(x))

        # Match input length
        if x.shape[-1] > input_length:
            x = x[..., :input_length]
        elif x.shape[-1] < input_length:
            x = F.pad(x, (0, input_length - x.shape[-1]))

        return x, total_vq_loss / self.num_codebooks

    # --- We keep encode/decode for the validation loop ---
    # These are copied *exactly* from your model.py

    @torch.no_grad()
    def encode(self, x):
        x = x.view(x.size(0), CHANNELS, -1)
        input_length = x.shape[-1]
        encoder_outputs = []
        for layer in self.encoder_convs:
            x = layer(x)
            encoder_outputs.append(x)
        z_e = self.pre_quant(x)
        z_q_list = []
        indices_list = []
        z_e_split = z_e.chunk(self.num_codebooks, dim=1)

        # Note: We use the VQ layer's forward, but ignore the vq_loss output
        for i in range(self.num_codebooks):
            z_q, indices, _ = self.quantizers[i](z_e_split[i])
            z_q_list.append(z_q)
            indices_list.append(indices)

        quantized_codes = torch.cat(z_q_list, dim=1)
        codes = self.transformer(quantized_codes)
        codes = self.post_transformer(codes)
        return codes, indices_list, input_length, encoder_outputs

    @torch.no_grad()
    def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
        if isinstance(codes_or_indices_list, list):
            # Case 1: Decoding from raw integer indices
            indices_list = codes_or_indices_list
            z_q_list = []
            for i, indices in enumerate(indices_list):
                quantized = self.quantizers[i].embedding(indices.squeeze(1))
                quantized = quantized.unsqueeze(0).transpose(1, 2)
                z_q_list.append(quantized)
            x = torch.cat(z_q_list, dim=1)
            x = self.transformer(x)
            x = self.post_transformer(x)
            encoder_outputs = None # No skip connections in this path
        elif isinstance(codes_or_indices_list, torch.Tensor):
            # Case 2: Decoding from the quantized float tensor (evaluation)
            x = codes_or_indices_list
        else:
            raise ValueError("Decoding input must be a torch.Tensor or a list of Tensors.")

        # Decoder
        for i, tconv in enumerate(self.decoder_tconvs):
            x = F.gelu(tconv(x))
            # Skip connections are intentionally omitted as per model.py

        x = torch.tanh(self.post_decoder_final(x))
        if input_length is not None:
            if x.shape[-1] > input_length:
                x = x[..., :input_length]
            elif x.shape[-1] < input_length:
                x = F.pad(x, (0, input_length - x.shape[-1]))
        return x.view(x.size(0), CHANNELS, -1)


print("Model architecture defined.")

# === 6. LOSS FUNCTIONS (MR-STFT) ===

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and return magnitude and phase."""
    # FIX: Changed pad_mode to 'constant' to avoid padding error when
    # fft_size // 2 (e.g., 1024) is larger than the input chunk (480).
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
                        return_complex=True, center=True, pad_mode='constant')
    return torch.abs(x_stft), torch.angle(x_stft)

class SpectralConvergengeLoss(nn.Module):
    """Spectral convergence loss."""
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x_mag, y_mag):
        return self.l1(torch.log(y_mag + 1e-7), torch.log(x_mag + 1e-7))

class MRSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss, essential for perceptual quality."""
    def __init__(self,
                 fft_sizes=[2048, 1024, 512, 256, 128],
                 hop_sizes=[240, 120, 60, 30, 15],
                 win_lengths=[1200, 600, 240, 120, 60]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        # FIX: Register windows as buffers instead of using nn.ModuleList
        # This makes them part of the module's state and moves them with .to(device)
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f'window_{i}', torch.hann_window(win_len))

        self.spec_conv_loss = SpectralConvergengeLoss()
        self.log_stft_mag_loss = LogSTFTMagnitudeLoss()

    def forward(self, y, x):
        total_sc_loss = 0.0
        total_mag_loss = 0.0

        # FIX: Get windows from buffers. They are auto-moved to the correct device.
        windows = []
        for i in range(len(self.win_lengths)):
             windows.append(getattr(self, f'window_{i}'))

        for fft_size, hop_size, win_length, window in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths, windows
        ):
            x_mag, _ = stft(x.squeeze(1), fft_size, hop_size, win_length, window)
            y_mag, _ = stft(y.squeeze(1), fft_size, hop_size, win_length, window)

            total_sc_loss += self.spec_conv_loss(x_mag, y_mag)
            total_mag_loss += self.log_stft_mag_loss(x_mag, y_mag)

        return total_sc_loss + total_mag_loss

print("Loss functions defined.")

# === 7. DATASET AND DATALOADER ===

class LibriTTSChunkDataset(Dataset):
    """
    Dataset to load LibriTTS, resample to 16kHz,
    and serve random 480-sample (30ms) chunks.
    """
    def __init__(self, root, url, download=True, target_sr=SR, chunk_size=TRAIN_WINDOW_SAMPLES):
        self.dataset = torchaudio.datasets.LIBRITTS(root=root, url=url, download=download)
        self.target_sr = target_sr
        self.chunk_size = chunk_size
        self.resamplers = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # FIX: Unpack 7 values from LIBRITTS dataset, matching older torchaudio API
        wav, sr, _, _, _, _, _ = self.dataset[idx]

        # Resample if necessary
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = Resample(sr, self.target_sr)
            wav = self.resamplers[sr](wav)

        # Ensure mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Normalize
        wav = wav / (torch.max(torch.abs(wav)) + 1e-6)

        # Get a random chunk
        if wav.shape[1] < self.chunk_size:
            # Pad if audio is too short
            wav = F.pad(wav, (0, self.chunk_size - wav.shape[1]), 'constant', 0)
        else:
            start = random.randint(0, wav.shape[1] - self.chunk_size)
            wav = wav[:, start:start + self.chunk_size]

        return wav.squeeze(0) # Return (L,) tensor

def collate_fn_train(batch):
    """Collate function to stack chunks."""
    # Batch is already a list of [L,] tensors
    return torch.stack(batch).unsqueeze(1) # (B, 1, L)

class LibriTTSValidationDataset(Dataset):
    """
    Dataset for validation. Returns *full* utterances
    for OaS validation.
    """
    def __init__(self, root, url, download=True, target_sr=SR):
        self.dataset = torchaudio.datasets.LIBRITTS(root=root, url=url, download=download)
        self.target_sr = target_sr
        self.resamplers = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # FIX: Unpack 7 values from LIBRITTS dataset, matching older torchaudio API
        wav, sr, _, _, _, _, _ = self.dataset[idx]

        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = Resample(sr, self.target_sr)
            wav = self.resamplers[sr](wav)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav / (torch.max(torch.abs(wav)) + 1e-6)
        return wav.squeeze(0) # Return (L,) tensor

def collate_fn_val(batch):
    """Collate for validation. Returns a list of full-length tensors."""
    return batch # Return list of variable-length tensors

print("Datasets defined.")

# === 8. CHECKPOINT UTILITIES ===

def save_checkpoint(model, optimizer, epoch, best_pesq, checkpoint_dir, filename="latest_model.pth"):
    """Saves a training checkpoint."""
    path = os.path.join(checkpoint_dir, filename)
    print(f"Saving checkpoint to {path}...")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_pesq': best_pesq,
    }
    torch.save(state, path)
    print("Checkpoint saved.")

def load_checkpoint(model, optimizer, checkpoint_dir, filename="latest_model.pth"):
    """Loads a training checkpoint."""
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return 0, 0.0 # start_epoch, best_pesq

    print(f"Loading checkpoint from {path}...")
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pesq = checkpoint.get('best_pesq', 0.0) # Handle older checkpoints
        print(f"Resuming training from Epoch {start_epoch} (Best PESQ: {best_pesq:.4f})")
        return start_epoch, best_pesq
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        return 0, 0.0

print("Checkpoint utilities defined.")

# === 9. VALIDATION FUNCTION (with OaS) ===

@torch.no_grad()
def validate(model, val_loader, device):
    """
    Performs validation using the *exact* Overlap-Save (OaS)
    streaming method from evaluation_tab.py to get accurate
    PESQ and STOI scores.
    """
    model.eval()
    print("Running OaS Validation...")

    all_pesq = []
    all_stoi = []

    # We use a small subset of validation data for speed
    num_val_files = 20

    for i, original_wav in enumerate(tqdm(val_loader, total=num_val_files)):
        if i >= num_val_files:
            break

        original_wav = original_wav.to(device) # (L,)

        # --- OaS Streaming Simulation ---
        reconstructed_chunks = []

        # Iterate over the audio with a hop of 240 samples
        for chunk_i in range(0, original_wav.shape[0], STREAMING_HOP_SAMPLES):

            # Get a window of 480 samples
            chunk = original_wav[chunk_i : chunk_i + TRAIN_WINDOW_SAMPLES]

            # Pad the final chunk if it's too short
            if len(chunk) < TRAIN_WINDOW_SAMPLES:
                chunk = F.pad(chunk, (0, TRAIN_WINDOW_SAMPLES - len(chunk)), 'constant', 0)

            # Add batch and channel dims
            chunk_tensor = chunk.unsqueeze(0).unsqueeze(0) # (1, 1, 480)

            # Use the model's single forward pass for eval
            # We ignore the vq_loss output during validation
            reconstructed_tensor, _ = model(chunk_tensor) # (1, 1, 480)

            decoded_audio = reconstructed_tensor.squeeze().detach().cpu().numpy()

            # OaS: Keep only the *last* 240 samples (the valid new audio)
            new_audio = decoded_audio[-STREAMING_HOP_SAMPLES:]
            reconstructed_chunks.append(new_audio)

        if not reconstructed_chunks:
            continue

        reconstructed_wav = np.concatenate(reconstructed_chunks)
        original_wav_np = original_wav.cpu().numpy()

        # Ensure lengths match for metrics
        min_len = min(len(original_wav_np), len(reconstructed_wav))
        original_chunk, reconstructed_chunk = original_wav_np[:min_len], reconstructed_wav[:min_len]

        # --- Calculate Metrics ---
        try:
            pesq_score = pesq(SR, original_chunk, reconstructed_chunk, 'wb')
            stoi_score = stoi(original_chunk, reconstructed_chunk, SR, extended=False)
            all_pesq.append(pesq_score)
            all_stoi.append(stoi_score)
        except Exception as e:
            print(f"Warning: Metric calculation failed for one file: {e}")

    if not all_pesq: # Handle case where validation fails on all files
        print("Warning: All validation metrics failed. Returning 0.0")
        return 0.0, 0.0

    avg_pesq = np.mean(all_pesq)
    avg_stoi = np.mean(all_stoi)

    print(f"Validation Complete - Avg PESQ: {avg_pesq:.4f} | Avg STOI: {avg_stoi:.4f}")
    return avg_pesq, avg_stoi

# === 10. TRAINING FUNCTION ===

def train_one_epoch(model, train_loader, optimizer, mr_stft_loss, device, scaler, epoch, wandb_logger): # <-- 2. Added wandb_logger
    """Trains the model for one epoch."""
    model.train()

    total_loss = 0
    total_l1_loss = 0
    total_stft_loss = 0
    total_vq_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

    for i, audio_chunk in enumerate(pbar):
        audio_chunk = audio_chunk.to(device) # (B, 1, 480)

        optimizer.zero_grad()

        # Use mixed precision
        # FIX: Use torch.amp.autocast for newer torch versions
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            reconstructed_audio, vq_loss = model(audio_chunk)

            # --- Calculate Losses ---
            l1_loss = F.l1_loss(reconstructed_audio, audio_chunk)
            stft_loss = mr_stft_loss(reconstructed_audio, audio_chunk)

            loss = l1_loss + stft_loss + vq_loss

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)

        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        total_loss += loss.item()
        total_l1_loss += l1_loss.item()
        total_stft_loss += stft_loss.item()
        total_vq_loss += vq_loss.item()

        if i % 25 == 0:
            avg_loss = total_loss / (i + 1)
            avg_l1 = total_l1_loss / (i + 1)
            avg_stft = total_stft_loss / (i + 1)
            avg_vq = total_vq_loss / (i + 1)

            pbar.set_postfix(
                Loss=f"{avg_loss:.4f}",
                L1=f"{avg_l1:.4f}",
                STFT=f"{avg_stft:.4f}",
                VQ=f"{avg_vq:.4f}"
            )

            wandb_logger.log({ # <-- 3. Use wandb_logger
                "train_loss": avg_loss,
                "train_l1_loss": avg_l1,
                "train_stft_loss": avg_stft,
                "train_vq_loss": avg_vq,
                "epoch": epoch,
                "step": epoch * len(train_loader) + i
            })

    return total_loss / len(train_loader)

# === 11. MAIN EXECUTION ===

def main():
    # --- 1. Setup ---
    print("Setting up training environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    torch.backends.cudnn.benchmark = True

    # --- 2. WandB Login ---

    # Create a dummy wandb object for fallback
    class DummyWandB:
        def log(self, *args, **kwargs): pass
        def init(self, *args, **kwargs): pass
        def login(self, *args, **kwargs): pass

    try:
        wandb_module.login()
        wandb_module.init(
            project="tiny-transformer-codec-v2",
            config={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "latent_dim": LATENT_DIM,
                "num_codebooks": NUM_CODEBOOKS,
                "codebook_size": CODEBOOK_SIZE,
                "transformer_blocks": TRANSFORMER_BLOCKS,
                "loss": "L1 + MR-STFT + VQ"
            }
        )
        print("Weights & Biases initialized.")
        wandb_logger = wandb_module # <-- 4. Use real module
    except Exception as e:
        print(f"Could not initialize WandB: {e}")
        print("Proceeding without WandB logging.")
        wandb_logger = DummyWandB() # <-- 5. Use dummy object

    # --- 3. Load Data ---
    print("Loading datasets...")
    try:
        train_dataset = LibriTTSChunkDataset(root=DATASET_DIR, url="train-clean-100", download=True)
        val_dataset = LibriTTSValidationDataset(root=DATASET_DIR, url="dev-clean", download=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_train,
            num_workers=2,
            pin_memory=True
        )

        # Validation loader is just a list of tensors
        val_loader = collate_fn_val([val_dataset[i] for i in range(len(val_dataset))])

        print(f"Training data: {len(train_dataset)} files.")
        print(f"Validation data: {len(val_dataset)} files.")
    except Exception as e:
        print(f"FATAL: Failed to load dataset: {e}")
        print("Please check your internet connection and disk space.")
        return

    # --- 4. Initialize Model, Optimizers, Loss ---
    print("Initializing model and optimizers...")
    model = TinyTransformerCodec().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.8, 0.99))
    # FIX: Removed 'verbose=True' as it's not supported in all torch versions
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)

    # FIX: Use torch.amp.GradScaler(device='cuda') for newer torch versions
    scaler = torch.amp.GradScaler(device='cuda')

    mr_stft_loss = MRSTFTLoss().to(device)

    # Log model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total trainable parameters: {num_params / 1e6:.2f}M")
    wandb_logger.log({"parameters_M": num_params / 1e6}) # <-- 6. Use wandb_logger

    # --- 5. Load Checkpoint (Resume) ---
    start_epoch, best_pesq = load_checkpoint(model, optimizer, CHECKPOINT_DIR, "latest_model.pth")

    # --- 6. Start Training Loop ---
    print(f"--- Starting Training from Epoch {start_epoch} ---")

    for epoch in range(start_epoch, NUM_EPOCHS):

        # --- Training ---
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, mr_stft_loss, device, scaler, epoch, wandb_logger) # <-- 7. Pass logger

        wandb_logger.log({ # <-- 8. Use wandb_logger
            "avg_train_loss_epoch": avg_train_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

        # --- Validation ---
        if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0:
            avg_pesq, avg_stoi = validate(model, val_loader, device)

            wandb_logger.log({ # <-- 9. Use wandb_logger
                "val_pesq": avg_pesq,
                "val_stoi": avg_stoi,
                "epoch": epoch
            })

            # Update LR Scheduler
            scheduler.step(avg_pesq)

            # --- Checkpoint Saving ---

            # Save latest checkpoint
            save_checkpoint(model, optimizer, epoch, best_pesq, CHECKPOINT_DIR, "latest_model.pth")

            # Save best checkpoint
            if avg_pesq > best_pesq:
                print(f"New best PESQ score! {best_pesq:.4f} -> {avg_pesq:.4f}. Saving best model.")
                best_pesq = avg_pesq
                save_checkpoint(model, optimizer, epoch, best_pesq, CHECKPOINT_DIR, "best_model.pth")
                wandb_logger.log({"best_pesq": best_pesq}) # <-- 10. Use wandb_logger

                # Also save the model.py-compatible file
                torch.save(
                    {'model_state_dict': model.state_dict()},
                    os.path.join(CHECKPOINT_DIR, "tiny_transformer_best.pt")
                )

            # --- Check for Target Metrics ---
            if avg_pesq > 3.6 and avg_stoi > 0.9:
                print("\n" + "="*50)
                print(f"!!! TARGET METRICS ACHIEVED AT EPOCH {epoch} !!!")
                print(f"PESQ: {avg_pesq:.4f} (> 3.6)")
                print(f"STOI: {avg_stoi:.4f} (> 0.9)")
                print("Training complete. You can find the best model in your Google Drive.")
                print(f"Best model path: {os.path.join(CHECKPOINT_DIR, 'tiny_transformer_best.pt')}")
                print("="*50 + "\n")
                break

    print("Training finished.")

if __name__ == "__main__":
    main()

