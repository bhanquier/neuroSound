"""
NeuroSound FLAC Simple Lossless - Version Minimaliste GARANTIE
===============================================================
Approche simple et prouvée : Delta encoding + FLAC
100% lossless garanti, pas de transformations complexes.

PRINCIPE:
- Delta encoding (différences) pour réduire la plage dynamique
- FLAC compresse mieux les petites valeurs
- Reconstruction parfaite garantie (opérations réversibles)
"""

import numpy as np
import wave
import struct
import subprocess
import json
import zlib
import base64
import tempfile
import os


class SimpleLosslessCodec:
    """Codec ultra-simple : delta encoding + FLAC, 100% réversible."""
    
    def __init__(self, compression_level=8):
        self.compression_level = compression_level
        self._check_flac()
    
    def _check_flac(self):
        try:
            subprocess.run(['flac', '--version'], capture_output=True, check=True)
        except:
            raise RuntimeError("❌ FLAC non installé ! brew install flac")
    
    def compress(self, input_wav, output_flac, verbose=True):
        """
        Compression lossless simple.
        1. Lecture WAV
        2. Delta encoding (différences)
        3. FLAC sur les différences
        4. Métadonnées minimales
        """
        if verbose:
            print("🔥 NeuroSound Simple Lossless - Compression")
            print("=" * 60)
        
        # Lecture
        if verbose:
            print("📖 Lecture WAV...")
        
        with wave.open(input_wav, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
            
            if params.sampwidth == 2:
                samples = np.frombuffer(frames, dtype=np.int16)
            else:
                raise ValueError("Seul 16-bit supporté pour cette version simple")
        
        if verbose:
            print(f"   ✓ {len(samples)} échantillons, {params.framerate} Hz")
        
        # Delta encoding
        if verbose:
            print("🔄 Delta encoding...")
        
        deltas = np.zeros_like(samples)
        deltas[0] = samples[0]  # Premier échantillon stocké tel quel
        deltas[1:] = samples[1:] - samples[:-1]  # Différences
        
        if verbose:
            orig_range = np.max(samples) - np.min(samples)
            delta_range = np.max(deltas) - np.min(deltas)
            reduction = 100 * (1 - delta_range / orig_range)
            print(f"   ✓ Plage réduite de {reduction:.1f}%")
        
        # Écriture WAV temporaire
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_wav = tmp.name
            with wave.open(tmp_wav, 'wb') as wav_out:
                wav_out.setparams(params)
                wav_out.writeframes(deltas.tobytes())
        
        # FLAC
        if verbose:
            print(f"🎵 Encodage FLAC (niveau {self.compression_level})...")
        
        cmd = ['flac', f'-{self.compression_level}', '--force', '-o', output_flac, tmp_wav]
        subprocess.run(cmd, check=True, capture_output=True)
        os.unlink(tmp_wav)
        
        # Métadonnées minimales
        meta = {
            'version': 'simple-lossless-1.0',
            'method': 'delta',
            'params': {
                'nchannels': params.nchannels,
                'sampwidth': params.sampwidth,
                'framerate': params.framerate,
                'nframes': params.nframes
            }
        }
        
        meta_compressed = base64.b64encode(
            zlib.compress(json.dumps(meta).encode())
        ).decode()
        
        cmd = ['metaflac', '--remove-tag=SFSL', 
               f'--set-tag=SFSL={meta_compressed}', output_flac]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Stats
        in_size = os.path.getsize(input_wav)
        out_size = os.path.getsize(output_flac)
        ratio = in_size / out_size
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ COMPRESSION RÉUSSIE !")
            print(f"📊 Original:  {in_size:,} bytes")
            print(f"📊 Compressé: {out_size:,} bytes")
            print(f"🎯 Ratio:     {ratio:.2f}x")
            print(f"🔒 Lossless:  100% garanti (delta reversible)")
            print("=" * 60)
        
        return {'ratio': ratio, 'size': out_size}
    
    def decompress(self, input_flac, output_wav, verbose=True):
        """
        Décompression avec reconstruction parfaite.
        1. Lecture métadonnées
        2. FLAC decode
        3. Reconstruction depuis deltas
        """
        if verbose:
            print("🔓 NeuroSound Simple Lossless - Décompression")
            print("=" * 60)
        
        # Métadonnées
        cmd = ['metaflac', '--show-tag=SFSL', input_flac]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        meta_tag = None
        for line in result.stdout.split('\n'):
            if line.startswith('SFSL='):
                meta_tag = line.split('=', 1)[1]
                break
        
        if not meta_tag:
            raise RuntimeError("Pas de métadonnées NeuroSound Simple Lossless!")
        
        meta = json.loads(
            zlib.decompress(base64.b64decode(meta_tag)).decode()
        )
        
        if verbose:
            print(f"✓ Version: {meta['version']}")
        
        # FLAC decode
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_wav = tmp.name
        
        cmd = ['flac', '-d', '-f', '-o', tmp_wav, input_flac]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Lecture deltas
        with wave.open(tmp_wav, 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            deltas = np.frombuffer(frames, dtype=np.int16)
        
        os.unlink(tmp_wav)
        
        # Reconstruction depuis deltas
        if verbose:
            print("🔄 Reconstruction depuis deltas...")
        
        samples = np.zeros_like(deltas)
        samples[0] = deltas[0]
        for i in range(1, len(deltas)):
            samples[i] = samples[i-1] + deltas[i]
        
        # Écriture WAV
        if verbose:
            print("💾 Écriture WAV...")
        
        p = meta['params']
        with wave.open(output_wav, 'wb') as wav_out:
            wav_out.setnchannels(p['nchannels'])
            wav_out.setsampwidth(p['sampwidth'])
            wav_out.setframerate(p['framerate'])
            wav_out.writeframes(samples.tobytes())
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ DÉCOMPRESSION RÉUSSIE !")
            print("🔒 Reconstruction bit-perfect garantie !")
            print("=" * 60)


def main():
    import sys
    
    if len(sys.argv) < 4:
        print("""
🔥 NeuroSound Simple Lossless
==============================

Usage:
    python3 neurosound_flac_simple_lossless.py compress <in.wav> <out.flac>
    python3 neurosound_flac_simple_lossless.py decompress <in.flac> <out.wav>

Garantie: 100% lossless, reconstruction bit-perfect
        """)
        sys.exit(1)
    
    mode = sys.argv[1]
    codec = SimpleLosslessCodec()
    
    if mode == 'compress':
        codec.compress(sys.argv[2], sys.argv[3])
    elif mode == 'decompress':
        codec.decompress(sys.argv[2], sys.argv[3])
    else:
        print(f"❌ Mode inconnu: {mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
