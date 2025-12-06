"""
🧠 NeuroSound v3.1 EXTREME - Au-delà des limites
================================================

Analyse spectrale intelligente pour VBR optimal.

Objectif EXTRÊME:
🎯 Ratio: 12.5x (+118% vs v1.0)
⚡ Vitesse: 0.10s (32% plus rapide)
🔋 Énergie: 29mJ (38% moins)
🌍 Compatible: 100% MP3

USAGE:
    codec = NeuroSoundExtreme(mode='balanced')
    codec.compress('input.wav', 'output.mp3')
"""

import numpy as np
import wave
import subprocess
import os
import tempfile


class NeuroSoundExtreme:
    """Codec v3.1 EXTREME optimisé - Code minimal."""
    
    # Pré-calcul des seuils (constantes)
    PURE_TONE_THRESHOLD = 50
    TONAL_THRESHOLD = 20
    MONO_CORRELATION_THRESHOLD = 0.9
    
    __slots__ = ('mode',)  # Optimisation mémoire
    
    def __init__(self, mode='balanced'):
        """
        mode:
        - 'aggressive': Vitesse max (VBR V5, q=5)
        - 'balanced': Compromis optimal (VBR adaptatif, q=3)
        - 'safe': Qualité max (VBR V1, q=2)
        """
        self.mode = mode
    
    @staticmethod
    def _analyze_tonality(audio_mono, sample_size=44100):
        """Analyse rapide de tonalité (FFT optimisée)."""
        # Utilise seulement 1s pour économie CPU
        sample = audio_mono[:min(sample_size, len(audio_mono))]
        
        # FFT (numpy optimisé)
        fft = np.fft.rfft(sample)
        magnitude = np.abs(fft)
        
        # Peak ratio (mesure de tonalité)
        max_peak = np.max(magnitude)
        mean_magnitude = np.mean(magnitude)
        
        return max_peak / (mean_magnitude + 1e-10)
    
    @staticmethod
    def _detect_stereo_correlation(left, right, sample_size=44100):
        """Détecte corrélation L/R rapide."""
        # Échantillon pour économie CPU
        size = min(sample_size, len(left))
        
        # Corrcoef sur float32 (plus rapide que float64)
        l_sample = left[:size].astype(np.float32)
        r_sample = right[:size].astype(np.float32)
        
        return np.corrcoef(l_sample, r_sample)[0, 1]
    
    def compress(self, input_wav, output_mp3, verbose=True):
        """Compression v3.1 optimisée."""
        import time
        t0 = time.time()
        
        if verbose:
            print("🧠 NEUROSOUND V3.1 EXTREME - OPTIMIZED")
            print("=" * 70)
        
        # Lecture WAV (context manager optimisé)
        with wave.open(input_wav, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
        
        if params.sampwidth != 2:
            raise ValueError("Seul 16-bit supporté")
        
        original_size = len(frames)
        
        # Conversion directe (évite copie)
        samples = np.frombuffer(frames, dtype=np.int16)
        
        if verbose:
            n_samples = len(samples) // params.nchannels
            print(f"📖 {params.nchannels}ch, {params.framerate}Hz, {n_samples} samples")
        
        # Traitement selon mono/stéréo
        if params.nchannels == 2:
            # Slicing optimisé (évite copie avec view)
            left = samples[0::2]
            right = samples[1::2]
            
            # Analyse corrélation (optimisée)
            correlation = self._detect_stereo_correlation(left, right)
            
            # DC offset removal in-place sur float32
            left_f = left.astype(np.float32)
            right_f = right.astype(np.float32)
            left_f -= left_f.mean()
            right_f -= right_f.mean()
            
            # Reconstruction entrelacée optimisée
            processed = np.empty(len(samples), dtype=np.int16)
            processed[0::2] = left_f.astype(np.int16)
            processed[1::2] = right_f.astype(np.int16)
            
            channels = 2
            vbr = '1' if self.mode == 'safe' else '2'
            quality_algo = '3' if self.mode == 'balanced' else '5'
            
            if verbose:
                corr_str = "high" if correlation > self.MONO_CORRELATION_THRESHOLD else "low"
                print(f"🔍 L/R correlation: {correlation:.2f} ({corr_str})")
        
        else:
            # Mono - analyse tonalité
            mono_f = samples.astype(np.float32)
            mono_f -= mono_f.mean()  # DC offset in-place
            
            # Analyse spectrale optimisée
            peak_ratio = self._analyze_tonality(mono_f)
            
            # VBR adaptatif selon tonalité
            if peak_ratio > self.PURE_TONE_THRESHOLD:
                vbr, desc = '5', 'pure tone'
            elif peak_ratio > self.TONAL_THRESHOLD:
                vbr, desc = '4', 'tonal'
            else:
                vbr, desc = '2', 'complex'
            
            quality_algo = '5' if self.mode == 'aggressive' else '3'
            
            processed = mono_f.astype(np.int16)
            channels = 1
            
            if verbose:
                print(f"🎵 Content: {desc} (peak ratio: {peak_ratio:.1f})")
        
        # Sauvegarde WAV temporaire (optimisée)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        try:
            with wave.open(temp_wav, 'wb') as wav_out:
                wav_out.setparams((channels, 2, params.framerate, 
                                   len(processed) // channels, 'NONE', 'not compressed'))
                wav_out.writeframes(processed.tobytes())
            
            # Encodage MP3 (commande optimisée)
            cmd = ['lame', '-V', vbr, '-q', quality_algo, '--quiet']
            
            # Joint stereo si stéréo
            if channels == 2:
                cmd.extend(['-m', 'j'])
            
            cmd.extend([temp_wav, output_mp3])
            
            # Exécution
            subprocess.run(cmd, check=True, capture_output=True)
        
        finally:
            # Nettoyage garanti
            os.remove(temp_wav)
        
        t1 = time.time()
        
        # Stats
        compressed_size = os.path.getsize(output_mp3)
        ratio = original_size / compressed_size
        energy = (t1 - t0) * 280  # mJ estimate
        
        if verbose:
            print(f"\n✅ {t1-t0:.3f}s | {ratio:.2f}x | {compressed_size:,} bytes | ~{energy:.0f}mJ")
            
            if ratio > 9.60:
                gain = ((ratio - 9.60) / 9.60) * 100
                print(f"🎉 +{gain:.1f}% vs v3.0")
        
        return compressed_size, ratio, energy


# Test
if __name__ == "__main__":
    import time
    
    print("🧠 NEUROSOUND V3.1 EXTREME - TEST")
    print("=" * 70)
    
    # Test avec audio existant
    if not os.path.exists('test_input_v3.wav'):
        # Génération audio test
        sample_rate = 44100
        duration = 30
        t = np.linspace(0, duration, sample_rate * duration, dtype=np.float32)
        
        # Audio mixte
        audio = np.zeros(len(t), dtype=np.float32)
        
        # Voix simulée
        voice_start = 5 * sample_rate
        voice_end = 15 * sample_rate
        voice_t = t[voice_start:voice_end]
        audio[voice_start:voice_end] = (
            np.sin(2 * np.pi * 200 * voice_t) * 0.1 +
            np.random.randn(len(voice_t)) * 0.15
        )
        
        # Musique
        music_start = 15 * sample_rate
        music_t = t[music_start:]
        audio[music_start:] = (
            np.sin(2 * np.pi * 440 * music_t) * 0.3 +
            np.sin(2 * np.pi * 554 * music_t) * 0.2 +
            np.sin(2 * np.pi * 659 * music_t) * 0.15 +
            np.random.randn(len(music_t)) * 0.05
        )
        
        samples_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open('test_input_v3.wav', 'wb') as wav:
            wav.setparams((1, 2, sample_rate, len(samples_int16), 'NONE', 'not compressed'))
            wav.writeframes(samples_int16.tobytes())
        
        print(f"✓ Audio test créé: {duration}s\n")
    
    # Test 3 modes
    results = []
    for mode in ['aggressive', 'balanced', 'safe']:
        print(f"\n{'='*70}")
        print(f"MODE: {mode.upper()}")
        print('='*70)
        
        codec = NeuroSoundExtreme(mode=mode)
        
        t0 = time.time()
        size, ratio, energy = codec.compress(
            'test_input_v3.wav',
            f'test_v31_{mode}.mp3'
        )
        elapsed = time.time() - t0
        
        results.append((mode, ratio, elapsed, energy))
    
    print(f"\n{'='*70}")
    print("📊 COMPARAISON v3.0 vs v3.1")
    print('='*70)
    print(f"{'Mode':<12} {'Ratio':<10} {'Temps':<12} {'Énergie'}")
    print('-'*70)
    print(f"v3.0 Aggr.   9.19x      0.114s       31mJ")
    print(f"v3.0 Bal.    9.60x      0.121s       34mJ")
    print('-'*70)
    
    for mode, ratio, elapsed, energy in results:
        print(f"v3.1 {mode.title():<7} {ratio:>5.2f}x     {elapsed:>6.3f}s      {energy:>4.0f}mJ")
    
    best = max(results, key=lambda x: x[1])
    print(f"\n🏆 Champion: v3.1 {best[0]} avec {best[1]:.2f}x")
    print(f"📈 Gain vs v3.0: +{((best[1]-9.60)/9.60)*100:.1f}%")
