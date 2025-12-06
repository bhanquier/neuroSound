#!/usr/bin/env python3
"""Benchmark comparatif v1.0 vs v2.0"""

import numpy as np
import wave
import time
import os

# Import v1
from neurosound_mp3_extreme import NeuroSoundMP3

# Import v2
from neurosound_v2_perceptual import NeuroSoundV2
from multiprocessing import cpu_count


def generate_test_audio(filename, duration=30):
    """Génère audio test."""
    sample_rate = 44100
    t = np.linspace(0, duration, sample_rate * duration, dtype=np.float32)
    
    # Musique complexe
    audio = (
        np.sin(2 * np.pi * 440 * t) * 0.3 +
        np.sin(2 * np.pi * 554 * t) * 0.2 +
        np.sin(2 * np.pi * 659 * t) * 0.15 +
        np.sin(2 * np.pi * 1760 * t) * 0.1 +
        np.sin(2 * np.pi * 110 * t) * 0.2 +
        np.random.randn(len(t)) * 0.05
    )
    
    samples = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setparams((1, 2, sample_rate, len(samples), 'NONE', 'not compressed'))
        wav.writeframes(samples.tobytes())
    
    return os.path.getsize(filename)


if __name__ == "__main__":
    print("🏁 NEUROSOUND BENCHMARK: v1.0 vs v2.0")
    print("=" * 70)
    
    # Génère audio test
    duration = 30
    print(f"📝 Génération audio test ({duration}s)...")
    original_size = generate_test_audio('benchmark_input.wav', duration)
    print(f"   ✓ {original_size:,} bytes\n")
    
    # Test v1.0
    print("🔵 TEST v1.0 (MP3 Extreme)")
    print("-" * 70)
    codec_v1 = NeuroSoundMP3(quality='extreme')
    
    t0 = time.time()
    size_v1, ratio_v1 = codec_v1.compress('benchmark_input.wav', 'benchmark_v1.mp3', verbose=False)
    t1 = time.time()
    time_v1 = t1 - t0
    
    print(f"   Ratio: {ratio_v1:.2f}x")
    print(f"   Temps: {time_v1:.3f}s")
    print(f"   Taille: {size_v1:,} bytes")
    print(f"   Économie: {100*(1-1/ratio_v1):.1f}%\n")
    
    # Test v2.0
    print("🟢 TEST v2.0 (Perceptual + Multi-core)")
    print("-" * 70)
    codec_v2 = NeuroSoundV2(cores=cpu_count(), perceptual=True, adaptive=True)
    
    t0 = time.time()
    size_v2, ratio_v2 = codec_v2.compress('benchmark_input.wav', 'benchmark_v2.mp3', verbose=False)
    t1 = time.time()
    time_v2 = t1 - t0
    
    print(f"   Ratio: {ratio_v2:.2f}x")
    print(f"   Temps: {time_v2:.3f}s")
    print(f"   Taille: {size_v2:,} bytes")
    print(f"   Économie: {100*(1-1/ratio_v2):.1f}%\n")
    
    # Comparaison
    print("📊 COMPARAISON")
    print("=" * 70)
    
    ratio_gain = ((ratio_v2 - ratio_v1) / ratio_v1) * 100
    time_gain = ((time_v1 - time_v2) / time_v1) * 100
    
    if ratio_v2 > ratio_v1:
        print(f"🎯 Ratio: v2.0 compresse {ratio_gain:+.1f}% mieux")
    else:
        print(f"🎯 Ratio: v1.0 compresse {-ratio_gain:.1f}% mieux")
    
    if time_v2 < time_v1:
        print(f"⚡ Vitesse: v2.0 est {time_gain:+.1f}% plus rapide")
    else:
        print(f"⚡ Vitesse: v1.0 est {-time_gain:.1f}% plus rapide")
    
    # Gagnant
    print("\n🏆 VERDICT")
    print("-" * 70)
    
    if ratio_v2 > ratio_v1 and time_v2 < time_v1:
        print("✅ v2.0 gagne sur TOUS les critères!")
    elif ratio_v2 > ratio_v1:
        print("✅ v2.0 gagne en COMPRESSION (mais v1.0 plus rapide)")
    elif time_v2 < time_v1:
        print("✅ v2.0 gagne en VITESSE (mais v1.0 compresse mieux)")
    else:
        print("✅ v1.0 reste optimal (meilleur ratio ET plus rapide)")
    
    print(f"\nRecommandation:")
    if ratio_v2 > ratio_v1:
        print(f"  • Choisir v2.0 si priorité = compression maximale")
        print(f"  • Choisir v1.0 si priorité = vitesse")
    else:
        print(f"  • v1.0 reste le meilleur choix général")
        print(f"  • v2.0 utile uniquement si CPU multi-core disponible ET temps non critique")
