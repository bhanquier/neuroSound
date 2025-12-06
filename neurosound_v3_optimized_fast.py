"""
NeuroSound v3 - Optimized Edition
==================================
Version optimisée avec accélérations majeures:
- Vectorisation NumPy poussée
- Cache intelligent
- Algorithmes optimisés
- Parallélisation préparée
"""

import numpy as np
import wave
import struct
from collections import defaultdict
import time
from functools import lru_cache


# ============================================================================
# OPTIMISATION 1: Cache LRU pour éviter recalculs
# ============================================================================

@lru_cache(maxsize=128)
def _cached_log_grid(n_levels):
    """Cache la grille logarithmique."""
    half_levels = n_levels // 2
    log_thresholds = np.logspace(-3, 2, half_levels)
    thresholds = np.concatenate([
        -log_thresholds[::-1],
        [0],
        log_thresholds
    ])
    return thresholds


# ============================================================================
# OPTIMISATION 2: Transformée KL Ultra-Rapide (vectorisée)
# ============================================================================

class FastKLTransform:
    """Version optimisée de la transformée KL."""
    
    def __init__(self, n_components=64, input_dim=256):
        self.n_components = n_components
        self.input_dim = input_dim
        
        # Initialisation optimisée avec SVD sur bruit
        np.random.seed(42)
        noise_samples = np.random.randn(max(n_components * 2, input_dim), input_dim)
        U, _, _ = np.linalg.svd(noise_samples, full_matrices=False)
        # Prend les n_components premières colonnes
        self.W = U[:, :n_components].astype(np.float32)
        
        # Statistiques vectorisées
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.std = np.ones(input_dim, dtype=np.float32)
        self.n_seen = 0
        
    def update_batch(self, X):
        """Mise à jour batch ultra-rapide."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Update stats en une passe vectorisée
        n_new = X.shape[0]
        new_mean = np.mean(X, axis=0)
        new_std = np.std(X, axis=0)
        
        # Fusion incrémentale
        total = self.n_seen + n_new
        self.mean = (self.mean * self.n_seen + new_mean * n_new) / total
        self.std = np.sqrt((self.std**2 * self.n_seen + new_std**2 * n_new) / total)
        self.n_seen = total
        
    def transform(self, x):
        """Projection ultra-rapide."""
        if len(x) != self.input_dim:
            x = np.pad(x, (0, max(0, self.input_dim - len(x))))[:self.input_dim]
        
        x_norm = (x - self.mean) / (self.std + 1e-8)
        return self.W.T @ x_norm.astype(np.float32)
    
    def transform_batch(self, X):
        """Transformation batch vectorisée."""
        # Assure que X a la bonne dimension
        if X.shape[1] != self.input_dim:
            # Pad ou tronque
            if X.shape[1] < self.input_dim:
                X = np.pad(X, ((0, 0), (0, self.input_dim - X.shape[1])))
            else:
                X = X[:, :self.input_dim]
        
        X_norm = (X - self.mean) / (self.std + 1e-8)
        return (X_norm @ self.W).astype(np.float32)
    
    def inverse_transform(self, y):
        """Reconstruction rapide."""
        x_norm = self.W @ y.astype(np.float32)
        return np.clip(x_norm * self.std + self.mean, -1e6, 1e6)
    
    def inverse_transform_batch(self, Y):
        """Reconstruction batch."""
        X_norm = Y @ self.W.T
        return np.clip(X_norm * self.std + self.mean, -1e6, 1e6)


# ============================================================================
# OPTIMISATION 3: Quantification Vectorisée Ultra-Rapide
# ============================================================================

class FastQuantizer:
    """Quantifieur ultra-optimisé."""
    
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.thresholds = _cached_log_grid(self.n_levels)
        
    def quantize_batch(self, coeffs_matrix):
        """Quantifie une matrice entière en une passe."""
        # searchsorted vectorisé sur toute la matrice
        indices = np.searchsorted(self.thresholds, coeffs_matrix.ravel())
        indices = np.clip(indices, 0, len(self.thresholds) - 1)
        return indices.reshape(coeffs_matrix.shape).astype(np.uint16)
    
    def dequantize_batch(self, indices_matrix):
        """Déquantification vectorisée."""
        return self.thresholds[indices_matrix]


# ============================================================================
# OPTIMISATION 4: Prédiction Vectorisée Complète
# ============================================================================

class FastPredictor:
    """Prédicteur ultra-rapide avec convolution."""
    
    def __init__(self, max_order=16):
        self.max_order = max_order
        self.weights = np.ones(max_order, dtype=np.float32) / max_order
        
    def predict_vectorized(self, signal):
        """
        Prédiction complète en une passe avec convolution.
        Beaucoup plus rapide que boucle!
        """
        # Convolution = prédiction linéaire vectorisée
        padded = np.pad(signal, (self.max_order, 0), mode='edge')
        
        # Convolution avec les poids inversés
        predictions = np.convolve(padded, self.weights[::-1], mode='valid')
        predictions = predictions[:len(signal)]
        
        return predictions.astype(np.float32)
    
    def compute_residual(self, signal):
        """Calcul résidu ultra-rapide."""
        predictions = self.predict_vectorized(signal)
        return (signal - predictions).astype(np.float32)


# ============================================================================
# OPTIMISATION 5: Segmentation Rapide Simplifiée
# ============================================================================

class FastSegmenter:
    """Segmenteur optimisé avec moins de calculs."""
    
    def __init__(self, window_size=512, min_segment=256, max_segment=4096):
        self.window_size = window_size
        self.min_segment = min_segment
        self.max_segment = max_segment
        
    def segment_fast(self, signal):
        """
        Segmentation rapide basée sur variance locale.
        Plus simple que Kolmogorov mais 10x plus rapide.
        """
        n_windows = len(signal) // self.window_size
        
        # Variance par fenêtre (vectorisé)
        signal_reshaped = signal[:n_windows * self.window_size].reshape(n_windows, -1)
        variances = np.var(signal_reshaped, axis=1)
        
        # Gradient de variance = transitions
        gradient = np.abs(np.diff(variances))
        threshold = np.percentile(gradient, 70)
        
        # Points de coupure
        cut_points = np.where(gradient > threshold)[0] * self.window_size
        
        # Création segments
        boundaries = np.concatenate([[0], cut_points, [len(signal)]])
        boundaries = np.unique(boundaries)
        
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = min(boundaries[i + 1], start + self.max_segment)
            
            if end - start >= self.min_segment:
                segments.append((start, end))
        
        return segments if segments else [(0, len(signal))]


# ============================================================================
# ARCHITECTURE OPTIMISÉE - VERSION 3
# ============================================================================

class OptimizedCompressor:
    """
    Compresseur optimisé pour la vitesse.
    Objectif: 5-10x plus rapide que v2.
    """
    
    def __init__(self, n_components=64, block_size=256, n_bits=8):
        self.n_components = n_components
        self.block_size = block_size
        self.n_bits = n_bits
        
        # Composants optimisés
        self.klt = FastKLTransform(n_components, block_size)
        self.quantizer = FastQuantizer(n_bits)
        self.predictor = FastPredictor(max_order=16)
        self.segmenter = FastSegmenter()
        
    def compress(self, signal, sample_rate=44100, verbose=False):
        """Compression optimisée."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"⚡ COMPRESSION OPTIMISÉE v3")
            print(f"{'='*80}")
            print(f"Signal: {len(signal):,} échantillons @ {sample_rate}Hz")
        
        t0 = time.time()
        
        # Normalisation
        signal_max = np.max(np.abs(signal))
        if signal_max > 0:
            signal_norm = (signal / signal_max).astype(np.float32)
        else:
            signal_norm = signal.astype(np.float32)
        
        # ÉTAPE 1: Segmentation rapide
        if verbose:
            print(f"\n📊 Segmentation rapide...")
        segments = self.segmenter.segment_fast(signal_norm)
        if verbose:
            print(f"   ✓ {len(segments)} segments")
        
        # ÉTAPE 2: Traitement batch par segment
        if verbose:
            print(f"\n🔬 Traitement batch...")
        
        compressed_blocks = []
        
        for seg_idx, (start, end) in enumerate(segments):
            segment = signal_norm[start:end]
            
            # Prédiction vectorisée
            residuals = self.predictor.compute_residual(segment)
            
            # Découpe en blocs (vectorisé)
            n_blocks = len(residuals) // self.block_size
            if n_blocks == 0:
                continue
                
            blocks = residuals[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
            
            # Apprentissage batch (pas à chaque bloc!)
            if seg_idx % 5 == 0:  # Seulement tous les 5 segments
                self.klt.update_batch(blocks)
            
            # Transformation batch
            coeffs_batch = self.klt.transform_batch(blocks)
            
            # Quantification batch
            quant_indices_batch = self.quantizer.quantize_batch(coeffs_batch)
            
            # Stockage compact
            for i, quant_indices in enumerate(quant_indices_batch):
                compressed_blocks.append({
                    'indices': quant_indices,
                    'segment_idx': seg_idx,
                    'block_idx': i
                })
        
        t_compress = time.time() - t0
        
        # Statistiques
        total_original_bits = len(signal) * 16
        total_compressed_bits = sum(block['indices'].nbytes * 8 for block in compressed_blocks)
        compression_ratio = total_original_bits / (total_compressed_bits + 1)
        
        if verbose:
            print(f"\n📈 RÉSULTATS:")
            print(f"   • Temps: {t_compress:.3f}s")
            print(f"   • Ratio: {compression_ratio:.2f}x")
            print(f"   • Blocs: {len(compressed_blocks)}")
        
        return {
            'blocks': compressed_blocks,
            'segments': segments,
            'signal_max': signal_max,
            'sample_rate': sample_rate,
            'klt_params': {
                'W': self.klt.W,
                'mean': self.klt.mean,
                'std': self.klt.std
            },
            'compression_ratio': compression_ratio,
            'compress_time': t_compress
        }
    
    def decompress(self, compressed_data, verbose=False):
        """Décompression optimisée."""
        if verbose:
            print(f"\n{'='*80}")
            print(f"🔓 DÉCOMPRESSION OPTIMISÉE")
            print(f"{'='*80}")
        
        t0 = time.time()
        
        # Restauration paramètres
        segments = compressed_data['segments']
        signal_max = compressed_data['signal_max']
        
        self.klt.W = compressed_data['klt_params']['W']
        self.klt.mean = compressed_data['klt_params']['mean']
        self.klt.std = compressed_data['klt_params']['std']
        
        # Reconstruction
        signal_length = segments[-1][1]
        reconstructed = np.zeros(signal_length, dtype=np.float32)
        
        # Organise par segment
        blocks_by_segment = defaultdict(list)
        for block_data in compressed_data['blocks']:
            blocks_by_segment[block_data['segment_idx']].append(block_data)
        
        # Reconstruction segment par segment
        for seg_idx, (start, end) in enumerate(segments):
            seg_blocks = sorted(blocks_by_segment[seg_idx], key=lambda x: x['block_idx'])
            
            if not seg_blocks:
                continue
            
            # Batch déquantification
            indices_batch = np.array([b['indices'] for b in seg_blocks])
            coeffs_batch = self.quantizer.dequantize_batch(indices_batch)
            
            # Batch transformation inverse
            residuals_batch = self.klt.inverse_transform_batch(coeffs_batch)
            segment_residuals = residuals_batch.ravel()[:end - start]
            
            # Reconstruction avec prédiction vectorisée
            segment_length = min(len(segment_residuals), end - start)
            segment_reconstructed = np.zeros(segment_length, dtype=np.float32)
            
            # Reconstruction itérative optimisée
            for i in range(segment_length):
                if i < self.predictor.max_order:
                    pred = 0.0
                else:
                    # Prédiction rapide sur historique
                    history = segment_reconstructed[i - self.predictor.max_order:i]
                    pred = np.dot(self.predictor.weights, history[::-1])
                
                segment_reconstructed[i] = segment_residuals[i] + pred
            
            reconstructed[start:start + segment_length] = segment_reconstructed
        
        # Dénormalisation
        reconstructed *= signal_max
        
        t_decompress = time.time() - t0
        if verbose:
            print(f"   ✓ Terminé en {t_decompress:.3f}s")
        
        return reconstructed


# ============================================================================
# UTILITAIRES
# ============================================================================

def load_wav(filename):
    """Charge un fichier WAV."""
    with wave.open(filename, 'rb') as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
        
        if params.sampwidth == 2:
            samples = np.frombuffer(frames, dtype=np.int16)
        else:
            raise ValueError(f"Sample width {params.sampwidth} non supporté")
        
        if params.nchannels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        
        return samples.astype(np.float32), params


def save_wav(filename, signal, params):
    """Sauvegarde un signal audio."""
    signal = np.nan_to_num(signal, nan=0.0, posinf=32767, neginf=-32768)
    signal_int16 = np.clip(signal, -32768, 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(params.nchannels)
        wav.setsampwidth(params.sampwidth)
        wav.setframerate(params.framerate)
        wav.writeframes(signal_int16.tobytes())


def compute_metrics(original, reconstructed):
    """Calcule les métriques de qualité."""
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len].astype(np.float32)
    recon = reconstructed[:min_len].astype(np.float32)
    
    mse = np.mean((orig - recon) ** 2)
    
    if mse > 0:
        max_val = np.max(np.abs(orig))
        psnr = 10 * np.log10(max_val ** 2 / mse)
    else:
        psnr = float('inf')
    
    signal_power = np.mean(orig ** 2)
    snr = 10 * np.log10(signal_power / (mse + 1e-10))
    
    return {'mse': mse, 'psnr': psnr, 'snr': snr}


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("⚡ NEUROSOUND V3 - OPTIMIZED EDITION ⚡")
    print("="*80)
    print("Objectif: 5-10x plus rapide que v2")
    print("="*80 + "\n")
    
    # Signal de test
    try:
        signal, params = load_wav('input.wav')
        print(f"📁 Fichier: input.wav")
    except FileNotFoundError:
        print("⚠️  Génération signal de test\n")
        
        sample_rate = 44100
        duration = 5
        t = np.linspace(0, duration, sample_rate * duration)
        
        signal = (
            10000 * np.sin(2 * np.pi * 440 * t) +
            7000 * np.sin(2 * np.pi * 880 * t) +
            5000 * np.sin(2 * np.pi * 1320 * t) +
            2000 * np.random.randn(len(t))
        ).astype(np.float32)
        
        class Params:
            nchannels = 1
            sampwidth = 2
            framerate = sample_rate
            nframes = len(signal)
        
        params = Params()
    
    # Test v3 optimisé
    print("\n" + "="*80)
    print("🚀 TEST VERSION OPTIMISÉE (v3)")
    print("="*80)
    
    compressor = OptimizedCompressor(n_components=64, block_size=256, n_bits=8)
    
    compressed = compressor.compress(signal, params.framerate, verbose=True)
    reconstructed = compressor.decompress(compressed, verbose=True)
    
    # Métriques
    metrics = compute_metrics(signal, reconstructed)
    
    print(f"\n{'='*80}")
    print("📊 MÉTRIQUES")
    print(f"{'='*80}")
    print(f"   • PSNR: {metrics['psnr']:.2f} dB")
    print(f"   • SNR:  {metrics['snr']:.2f} dB")
    print(f"   • Ratio: {compressed['compression_ratio']:.2f}x")
    print(f"   • Vitesse: {compressed['compress_time']:.3f}s")
    
    # Sauvegarde
    save_wav('output_v3_optimized.wav', reconstructed, params)
    print(f"\n💾 Sauvegardé: output_v3_optimized.wav")
    
    print(f"\n{'='*80}")
    print("✨ TERMINÉ ✨")
    print(f"{'='*80}\n")
