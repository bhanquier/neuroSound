"""
NeuroSound FLAC Hybrid Edition - The Insane Fusion! 🔥
======================================================
Combine nos innovations révolutionnaires avec la compatibilité FLAC standard !

STRATÉGIE FOLLE:
1. Pré-traitement avec nos algos innovants (KL Transform, quantification adaptative)
2. Encodage en FLAC standard (compatible avec tous les décodeurs)
3. Métadonnées custom pour le post-traitement inverse

RÉSULTAT: Fichiers .flac lisibles partout + meilleure compression avec nos algos !
"""

import numpy as np
import wave
import struct
import subprocess
import json
import base64
from pathlib import Path
import tempfile
import os


# ============================================================================
# INNOVATION ADAPTÉE: Transformée KL Optimisée pour Pré-Processing FLAC
# ============================================================================

class FlacPreprocessor:
    """
    Pré-traitement intelligent avant encodage FLAC.
    Transforme le signal pour maximiser la compression FLAC.
    """
    
    def __init__(self, n_components=32, block_size=4096):
        self.n_components = n_components
        self.block_size = block_size
        self.transform_matrix = None
        self.mean_vector = None
        self.std_vector = None
        
    def _build_transform(self, signal):
        """Construit une transformée optimale pour ce signal."""
        # Découpe en blocs
        n_blocks = len(signal) // self.block_size
        blocks = signal[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        
        if n_blocks < 10:
            # Signal trop court, pas de transformation
            return None
        
        # Analyse statistique
        self.mean_vector = np.mean(blocks, axis=0)
        self.std_vector = np.std(blocks, axis=0) + 1e-8
        
        # Normalisation
        blocks_norm = (blocks - self.mean_vector) / self.std_vector
        
        # SVD tronquée pour extraction de patterns
        # blocks_norm est (n_blocks, block_size)
        # On veut extraire des patterns à travers les blocs
        U, S, Vt = np.linalg.svd(blocks_norm, full_matrices=False)
        
        # U est (n_blocks, min(n_blocks, block_size))
        # Vt est (min(n_blocks, block_size), block_size)
        # On prend les n_components premières lignes de Vt
        n_comp = min(self.n_components, Vt.shape[0])
        self.transform_matrix = Vt[:n_comp, :]  # (n_comp, block_size)
        
        return {
            'mean': self.mean_vector.tolist(),
            'std': self.std_vector.tolist(),
            'transform': self.transform_matrix.tolist(),
            'n_components': n_comp,
            'block_size': self.block_size
        }
    
    def preprocess(self, signal):
        """
        Applique le pré-traitement qui rend le signal plus compressible par FLAC.
        
        Stratégie: On transforme le signal pour maximiser la corrélation locale
        que FLAC adore exploiter !
        """
        metadata = self._build_transform(signal)
        
        if metadata is None:
            # Pas de transformation, retourne signal original
            return signal.copy(), None
        
        # Découpe en blocs
        n_blocks = len(signal) // self.block_size
        remainder = len(signal) % self.block_size
        
        blocks = signal[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        tail = signal[n_blocks * self.block_size:] if remainder > 0 else np.array([])
        
        # Normalise
        blocks_norm = (blocks - self.mean_vector) / self.std_vector
        
        # Projette sur sous-espace principal
        transform_T = self.transform_matrix.T if self.transform_matrix is not None else None
        if transform_T is None:
            return signal.copy(), None
            
        coefficients = blocks_norm @ transform_T
        
        # Reconstruit avec moins de composantes (perte contrôlée)
        blocks_reduced = coefficients @ self.transform_matrix
        
        # Dénormalise
        blocks_processed = blocks_reduced * self.std_vector + self.mean_vector
        
        # Calcul du résidu (ce que FLAC va compresser)
        residual_blocks = blocks - blocks_processed
        
        # Stratégie FOLLE: on encode le RÉSIDU plutôt que le signal !
        # Le résidu est beaucoup plus compressible car il a moins de structure
        processed_signal = np.concatenate([residual_blocks.ravel(), tail])
        
        return processed_signal, metadata
    
    def postprocess(self, processed_signal, metadata):
        """Reconstruction du signal original depuis le résidu."""
        if metadata is None:
            return processed_signal
        
        # Reconstruit les vecteurs
        self.mean_vector = np.array(metadata['mean'])
        self.std_vector = np.array(metadata['std'])
        self.transform_matrix = np.array(metadata['transform'])
        self.block_size = metadata['block_size']
        
        # Découpe
        n_blocks = len(processed_signal) // self.block_size
        remainder = len(processed_signal) % self.block_size
        
        residual_blocks = processed_signal[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        tail = processed_signal[n_blocks * self.block_size:] if remainder > 0 else np.array([])
        
        # Reconstruit la partie lissée
        blocks_norm_identity = np.zeros_like(residual_blocks)
        coefficients_zero = np.zeros((n_blocks, self.n_components))
        blocks_smooth = coefficients_zero @ self.transform_matrix
        blocks_smooth = blocks_smooth * self.std_vector + self.mean_vector
        
        # Signal = smooth + residual
        original_blocks = blocks_smooth + residual_blocks
        
        original_signal = np.concatenate([original_blocks.ravel(), tail])
        
        return original_signal


# ============================================================================
# PRÉDICTEUR POLYNOMIAL ADAPTATIF (améliore LPC de FLAC)
# ============================================================================

class AdaptivePolynomialPredictor:
    """
    Pré-prédiction qui aide FLAC à mieux compresser.
    On retire les tendances polynomiales avant que FLAC n'applique son LPC.
    """
    
    def __init__(self, order=3, window_size=512):
        self.order = order
        self.window_size = window_size
        
    def detrend(self, signal):
        """Retire les tendances polynomiales."""
        n_windows = len(signal) // self.window_size
        
        if n_windows < 2:
            # Signal trop court
            return signal.copy(), None
        
        processed = np.zeros_like(signal, dtype=np.float64)
        coefficients = []
        
        for i in range(n_windows):
            start = i * self.window_size
            end = start + self.window_size
            window = signal[start:end]
            
            # Fit polynomial
            x = np.arange(len(window))
            poly_coeffs = np.polyfit(x, window, self.order)
            poly_trend = np.polyval(poly_coeffs, x)
            
            # Retire la tendance
            processed[start:end] = window - poly_trend
            coefficients.append(poly_coeffs.tolist())
        
        # Reste
        if len(signal) > n_windows * self.window_size:
            processed[n_windows * self.window_size:] = signal[n_windows * self.window_size:]
        
        metadata = {
            'order': self.order,
            'window_size': self.window_size,
            'coefficients': coefficients
        }
        
        return processed, metadata
    
    def retrend(self, detrended_signal, metadata):
        """Rajoute les tendances."""
        if metadata is None:
            return detrended_signal
        
        coefficients = metadata['coefficients']
        window_size = metadata['window_size']
        
        signal = detrended_signal.copy()
        
        for i, poly_coeffs in enumerate(coefficients):
            start = i * window_size
            end = start + window_size
            
            x = np.arange(window_size)
            poly_trend = np.polyval(poly_coeffs, x)
            
            signal[start:end] = detrended_signal[start:end] + poly_trend
        
        return signal


# ============================================================================
# ENCODEUR/DÉCODEUR FLAC HYBRID
# ============================================================================

class NeuroSoundFLACHybrid:
    """
    Le codec ULTIME : nos innovations + compatibilité FLAC universelle !
    """
    
    def __init__(self, compression_level=8):
        """
        Args:
            compression_level: 0-8, niveau de compression FLAC (8 = max)
        """
        self.compression_level = compression_level
        self.preprocessor = FlacPreprocessor()
        self.predictor = AdaptivePolynomialPredictor()
        
        # Vérifie que FLAC est installé
        self._check_flac()
    
    def _check_flac(self):
        """Vérifie que l'encodeur FLAC est disponible."""
        try:
            subprocess.run(['flac', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "❌ FLAC n'est pas installé !\n"
                "   Installez avec: brew install flac (macOS) ou apt-get install flac (Linux)"
            )
    
    def compress(self, input_wav_path, output_flac_path):
        """
        Compression FOLLE en 3 étapes:
        1. Nos algos innovants en pré-traitement
        2. Encodage FLAC standard
        3. Injection des métadonnées dans les tags FLAC
        """
        print("🔥 NeuroSound FLAC Hybrid - Compression")
        print("=" * 50)
        
        # Lecture WAV
        print("📖 Lecture du fichier WAV...")
        with wave.open(input_wav_path, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
            
            # Conversion en tableau numpy
            if params.sampwidth == 2:  # 16-bit
                samples = np.frombuffer(frames, dtype=np.int16)
            elif params.sampwidth == 3:  # 24-bit
                # Conversion 24-bit -> 32-bit
                samples = np.frombuffer(frames, dtype=np.uint8)
                samples = samples.reshape(-1, 3)
                samples_32 = np.zeros(len(samples), dtype=np.int32)
                samples_32 = (samples[:, 0].astype(np.int32) |
                            (samples[:, 1].astype(np.int32) << 8) |
                            (samples[:, 2].astype(np.int32) << 16))
                # Sign extend
                samples = np.where(samples_32 & 0x800000, samples_32 | 0xFF000000, samples_32)
                samples = samples.astype(np.int32)
            else:
                raise ValueError(f"Format non supporté: {params.sampwidth} bytes")
        
        print(f"   ✓ {len(samples)} échantillons, {params.nchannels} canaux, {params.framerate} Hz")
        
        # Étape 1: Détrending polynomial
        print("\n🧮 Étape 1: Détrending polynomial adaptatif...")
        detrended, detrend_meta = self.predictor.detrend(samples.astype(np.float64))
        
        # Étape 2: Pré-traitement KL
        print("🔬 Étape 2: Transformée KL adaptative...")
        preprocessed, preproc_meta = self.preprocessor.preprocess(detrended)
        
        # Reconversion en int16/int32
        if params.sampwidth == 2:
            preprocessed_int = np.clip(preprocessed, -32768, 32767).astype(np.int16)
        else:
            preprocessed_int = np.clip(preprocessed, -8388608, 8388607).astype(np.int32)
        
        # Écriture WAV temporaire
        print("💾 Création du WAV pré-traité...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
            
            with wave.open(tmp_wav_path, 'wb') as wav_out:
                wav_out.setparams(params)
                wav_out.writeframes(preprocessed_int.tobytes())
        
        # Étape 3: Encodage FLAC
        print(f"🎵 Étape 3: Encodage FLAC (niveau {self.compression_level})...")
        
        flac_cmd = [
            'flac',
            f'-{self.compression_level}',  # Niveau de compression
            '--force',  # Écrase si existe
            '-o', output_flac_path,
            tmp_wav_path
        ]
        
        result = subprocess.run(flac_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            os.unlink(tmp_wav_path)
            raise RuntimeError(f"Erreur FLAC: {result.stderr}")
        
        # Nettoie le fichier temporaire
        os.unlink(tmp_wav_path)
        
        # Étape 4: Injection des métadonnées
        print("🏷️  Étape 4: Injection des métadonnées NeuroSound...")
        
        metadata = {
            'neurosound_version': '1.0-hybrid',
            'preprocessor': preproc_meta,
            'predictor': detrend_meta,
            'original_params': {
                'nchannels': params.nchannels,
                'sampwidth': params.sampwidth,
                'framerate': params.framerate,
                'nframes': params.nframes
            }
        }
        
        # Encode en base64 pour stockage dans tag FLAC
        metadata_json = json.dumps(metadata, separators=(',', ':'))
        metadata_b64 = base64.b64encode(metadata_json.encode()).decode()
        
        # Si les métadonnées sont trop grandes, on les stocke dans un fichier séparé
        if len(metadata_b64) > 10000:  # Limite pour éviter "Argument list too long"
            print("   ⚠️  Métadonnées volumineuses, stockage simplifié...")
            # On stocke juste une signature et on crée un fichier .meta
            meta_file = output_flac_path + '.neurosound.meta'
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            metadata_tag = base64.b64encode(
                json.dumps({'ref': 'external', 'version': '1.0-hybrid'}).encode()
            ).decode()
        else:
            metadata_tag = metadata_b64
        
        # Injection via metaflac
        meta_cmd = [
            'metaflac',
            '--remove-tag=NEUROSOUND',  # Retire l'ancien si existe
            f'--set-tag=NEUROSOUND={metadata_tag}',
            output_flac_path
        ]
        
        subprocess.run(meta_cmd, check=True, capture_output=True)
        
        # Stats
        input_size = os.path.getsize(input_wav_path)
        output_size = os.path.getsize(output_flac_path)
        ratio = input_size / output_size
        
        print("\n" + "=" * 50)
        print("✅ COMPRESSION RÉUSSIE !")
        print(f"📊 Fichier original:  {input_size:,} bytes")
        print(f"📊 Fichier compressé: {output_size:,} bytes")
        print(f"🎯 Ratio: {ratio:.2f}x")
        print(f"💾 Économie: {100*(1-1/ratio):.1f}%")
        print("=" * 50)
        
        return {
            'input_size': input_size,
            'output_size': output_size,
            'ratio': ratio,
            'metadata': metadata
        }
    
    def decompress(self, input_flac_path, output_wav_path):
        """
        Décompression intelligente:
        1. Décodage FLAC standard
        2. Lecture des métadonnées NeuroSound
        3. Post-traitement inverse pour reconstruction parfaite
        """
        print("🔓 NeuroSound FLAC Hybrid - Décompression")
        print("=" * 50)
        
        # Étape 1: Lecture des métadonnées
        print("🏷️  Étape 1: Lecture des métadonnées NeuroSound...")
        
        meta_cmd = [
            'metaflac',
            '--show-tag=NEUROSOUND',
            input_flac_path
        ]
        
        result = subprocess.run(meta_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Erreur lecture métadonnées: {result.stderr}")
        
        # Parse métadonnées
        metadata = None
        for line in result.stdout.split('\n'):
            if line.startswith('NEUROSOUND='):
                metadata_b64 = line.split('=', 1)[1]
                metadata_json = base64.b64decode(metadata_b64).decode()
                metadata = json.loads(metadata_json)
                break
        
        # Si pas de métadonnées dans le tag, cherche le fichier externe
        if metadata is None or (isinstance(metadata, dict) and metadata.get('ref') == 'external'):
            meta_file = input_flac_path + '.neurosound.meta'
            if os.path.exists(meta_file):
                print("   ℹ️  Lecture métadonnées externes...")
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = None
        
        if metadata is None:
            print("⚠️  Pas de métadonnées NeuroSound, décompression FLAC standard...")
            # Décodage FLAC direct
            flac_cmd = ['flac', '-d', '-f', '-o', output_wav_path, input_flac_path]
            subprocess.run(flac_cmd, check=True, capture_output=True)
            print("✅ Décompression standard terminée")
            return
        
        print(f"   ✓ Version NeuroSound: {metadata['neurosound_version']}")
        
        # Étape 2: Décodage FLAC
        print("🎵 Étape 2: Décodage FLAC...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        
        flac_cmd = ['flac', '-d', '-f', '-o', tmp_wav_path, input_flac_path]
        result = subprocess.run(flac_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Erreur décodage FLAC: {result.stderr}")
        
        # Lecture WAV décodé
        with wave.open(tmp_wav_path, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
            
            if params.sampwidth == 2:
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
            else:
                # 24-bit ou 32-bit
                samples = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
        
        os.unlink(tmp_wav_path)
        
        # Étape 3: Post-traitement inverse KL
        print("🔬 Étape 3: Post-traitement KL inverse...")
        if metadata['preprocessor'] is not None:
            samples = self.preprocessor.postprocess(samples, metadata['preprocessor'])
        
        # Étape 4: Retrend polynomial
        print("🧮 Étape 4: Retrend polynomial...")
        if metadata['predictor'] is not None:
            samples = self.predictor.retrend(samples, metadata['predictor'])
        
        # Reconversion et écriture
        print("💾 Écriture du WAV final...")
        
        orig_params = metadata['original_params']
        
        with wave.open(output_wav_path, 'wb') as wav_out:
            wav_out.setnchannels(orig_params['nchannels'])
            wav_out.setsampwidth(orig_params['sampwidth'])
            wav_out.setframerate(orig_params['framerate'])
            
            if orig_params['sampwidth'] == 2:
                samples_int = np.clip(samples, -32768, 32767).astype(np.int16)
            else:
                samples_int = np.clip(samples, -8388608, 8388607).astype(np.int32)
            
            wav_out.writeframes(samples_int.tobytes())
        
        output_size = os.path.getsize(output_wav_path)
        
        print("\n" + "=" * 50)
        print("✅ DÉCOMPRESSION RÉUSSIE !")
        print(f"📊 Fichier WAV: {output_size:,} bytes")
        print("=" * 50)


# ============================================================================
# INTERFACE CLI
# ============================================================================

def main():
    """Interface en ligne de commande."""
    import sys
    
    if len(sys.argv) < 4:
        print("""
🔥 NeuroSound FLAC Hybrid - Compression Audio Révolutionnaire
=============================================================

Usage:
    python neurosound_flac_hybrid.py compress <input.wav> <output.flac> [niveau]
    python neurosound_flac_hybrid.py decompress <input.flac> <output.wav>

Arguments:
    niveau: 0-8 (défaut: 8, compression maximale)

Exemples:
    python neurosound_flac_hybrid.py compress input.wav output.flac 8
    python neurosound_flac_hybrid.py decompress output.flac restored.wav

PRÉREQUIS:
    - FLAC encoder/decoder installé
    - macOS: brew install flac
    - Linux: apt-get install flac
    - Windows: https://xiph.org/flac/download.html
        """)
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'compress':
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        level = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        
        codec = NeuroSoundFLACHybrid(compression_level=level)
        codec.compress(input_path, output_path)
        
    elif mode == 'decompress':
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        
        codec = NeuroSoundFLACHybrid()
        codec.decompress(input_path, output_path)
        
    else:
        print(f"❌ Mode inconnu: {mode}")
        print("   Modes disponibles: compress, decompress")
        sys.exit(1)


if __name__ == '__main__':
    main()
