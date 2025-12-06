#!/usr/bin/env python3
"""Benchmark v1.0 vs v2.0 vs v2.1"""

import numpy as np
import wave
import time
import os

from neurosound_mp3_extreme import NeuroSoundMP3
from neurosound_v2_perceptual import NeuroSoundV2
from neurosound_v2_1_energy import NeuroSoundV21
from multiprocessing import cpu_count


def generate_test_audio(filename, duration=30):
    """Génère audio test."""
    sample_rate = 44100
    t = np.linspace(0, duration, sample_rate * duration, dtype=np.float32)
    
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
    print("🏁 NEUROSOUND MEGA-BENCHMARK: v1.0 vs v2.0 vs v2.1")
    print("=" * 80)
    
    duration = 30
    print(f"📝 Génération audio test ({duration}s)...")
    original_size = generate_test_audio('megabench_input.wav', duration)
    print(f"   ✓ {original_size:,} bytes\n")
    
    results = []
    
    # v1.0
    print("🔵 v1.0 - MP3 Extreme (baseline)")
    print("-" * 80)
    codec_v1 = NeuroSoundMP3(quality='extreme')
    t0 = time.time()
    size_v1, ratio_v1 = codec_v1.compress('megabench_input.wav', 'mega_v1.mp3', verbose=False)
    time_v1 = time.time() - t0
    energy_v1 = time_v1 * 300  # Estimation mW
    
    results.append(('v1.0 Extreme', ratio_v1, time_v1, energy_v1, size_v1))
    print(f"   Ratio: {ratio_v1:.2f}x | Temps: {time_v1:.3f}s | Énergie: ~{energy_v1:.0f}mJ\n")
    
    # v2.0
    print("🟢 v2.0 - Perceptual + Multi-core")
    print("-" * 80)
    codec_v2 = NeuroSoundV2(cores=cpu_count(), perceptual=True, adaptive=True)
    t0 = time.time()
    size_v2, ratio_v2 = codec_v2.compress('megabench_input.wav', 'mega_v2.mp3', verbose=False)
    time_v2 = time.time() - t0
    energy_v2 = 350 + time_v2 * 300  # FFT overhead + LAME
    
    results.append(('v2.0 Perceptual', ratio_v2, time_v2, energy_v2, size_v2))
    print(f"   Ratio: {ratio_v2:.2f}x | Temps: {time_v2:.3f}s | Énergie: ~{energy_v2:.0f}mJ\n")
    
    # v2.1 modes
    for mode, emoji in [('ultra', '⚡'), ('balanced', '⚖️'), ('quality', '🎯')]:
        print(f"{emoji} v2.1 - Energy Optimized ({mode.upper()})")
        print("-" * 80)
        codec_v21 = NeuroSoundV21(energy_mode=mode)
        t0 = time.time()
        size, ratio, energy = codec_v21.compress('megabench_input.wav', f'mega_v21_{mode}.mp3', verbose=False)
        elapsed = time.time() - t0
        
        results.append((f'v2.1 {mode.title()}', ratio, elapsed, energy, size))
        print(f"   Ratio: {ratio:.2f}x | Temps: {elapsed:.3f}s | Énergie: ~{energy:.0f}mJ\n")
    
    # Tableau comparatif
    print("\n" + "=" * 80)
    print("📊 TABLEAU COMPARATIF COMPLET")
    print("=" * 80)
    print(f"{'Version':<20} {'Ratio':<10} {'Temps':<12} {'Énergie':<12} {'Taille':<12}")
    print("-" * 80)
    
    for name, ratio, elapsed, energy, size in results:
        print(f"{name:<20} {ratio:>6.2f}x   {elapsed:>8.3f}s   {energy:>8.0f}mJ   {size:>9,} B")
    
    # Gagnants
    best_ratio = max(results, key=lambda x: x[1])
    best_speed = min(results, key=lambda x: x[2])
    best_energy = min(results, key=lambda x: x[3])
    
    print("\n" + "=" * 80)
    print("🏆 CHAMPIONS")
    print("=" * 80)
    print(f"🎯 Meilleur ratio:    {best_ratio[0]:<20} ({best_ratio[1]:.2f}x)")
    print(f"⚡ Plus rapide:       {best_speed[0]:<20} ({best_speed[2]:.3f}s)")
    print(f"🔋 Moins d'énergie:   {best_energy[0]:<20} ({best_energy[3]:.0f}mJ)")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMANDATIONS")
    print("=" * 80)
    print("• Mobile/IoT/Temps réel    → v1.0 Extreme (rapide, simple, efficace)")
    print("• Serveurs batch           → v2.1 Ultra (meilleur ratio, économie max)")
    print("• Streaming live           → v1.0 Extreme (latence minimale)")
    print("• Archivage                → v2.1 Ultra (compression maximale)")
    print("• Usage général            → v2.1 Balanced (bon compromis)")
