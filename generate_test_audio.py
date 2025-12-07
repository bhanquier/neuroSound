#!/usr/bin/env python3
"""
Génère des fichiers WAV de test de différents types
"""

import numpy as np
from scipy.io import wavfile
import os

def generate_test_files():
    """Génère des fichiers WAV variés pour tests réalistes"""
    
    sample_rate = 44100
    duration = 30  # 30 secondes
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    output_dir = "/Users/bhanquier/neuroSound/test_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎵 Génération de fichiers WAV de test...\n")
    
    # 1. Musique complexe (simulation orchestre)
    print("1️⃣ Musique complexe (multi-fréquences)...")
    complex_music = np.zeros(len(t))
    for freq in [220, 277, 330, 440, 554, 660]:  # Accords riches
        complex_music += 0.1 * np.sin(2 * np.pi * freq * t)
    # Ajouter variations temporelles
    envelope = np.sin(2 * np.pi * 0.5 * t) * 0.5 + 0.5
    complex_music *= envelope
    # Stéréo large
    left = complex_music + 0.05 * np.random.randn(len(t))
    right = complex_music * 0.8 + 0.05 * np.random.randn(len(t))
    stereo = np.vstack([left, right]).T
    stereo = (stereo * 32767).astype(np.int16)
    wavfile.write(f"{output_dir}/complex_music.wav", sample_rate, stereo)
    print(f"   ✅ {os.path.getsize(f'{output_dir}/complex_music.wav')/1024/1024:.1f} MB")
    
    # 2. Audio simple avec silence (podcast simulé)
    print("2️⃣ Podcast avec pauses (50% silence)...")
    podcast = np.zeros(len(t))
    # Voix simulée (fréquences 150-300 Hz)
    for i in range(0, len(t), int(sample_rate * 2)):  # Phrases de 2s
        if i % (sample_rate * 4) < sample_rate * 2:  # Pause toutes les 4s
            segment = t[i:min(i+int(sample_rate*2), len(t))] - t[i]
            podcast[i:min(i+int(sample_rate*2), len(t))] = (
                0.3 * np.sin(2 * np.pi * 200 * segment) +
                0.2 * np.sin(2 * np.pi * 250 * segment)
            )
    # Mono (quasi identique L/R)
    left = podcast
    right = podcast + 0.001 * np.random.randn(len(t))
    stereo = np.vstack([left, right]).T
    stereo = (stereo * 32767).astype(np.int16)
    wavfile.write(f"{output_dir}/podcast_silence.wav", sample_rate, stereo)
    print(f"   ✅ {os.path.getsize(f'{output_dir}/podcast_silence.wav')/1024/1024:.1f} MB")
    
    # 3. Ton pur (test optimal)
    print("3️⃣ Ton pur 440 Hz (cas optimal)...")
    pure_tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.vstack([pure_tone, pure_tone]).T
    stereo = (stereo * 32767).astype(np.int16)
    wavfile.write(f"{output_dir}/pure_tone.wav", sample_rate, stereo)
    print(f"   ✅ {os.path.getsize(f'{output_dir}/pure_tone.wav')/1024/1024:.1f} MB")
    
    # 4. Musique simple mono (quasi-mono)
    print("4️⃣ Musique simple quasi-mono...")
    simple_music = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 554 * t)
    # 99% correlation
    left = simple_music
    right = simple_music + 0.01 * np.random.randn(len(t))
    stereo = np.vstack([left, right]).T
    stereo = (stereo * 32767).astype(np.int16)
    wavfile.write(f"{output_dir}/simple_quasi_mono.wav", sample_rate, stereo)
    print(f"   ✅ {os.path.getsize(f'{output_dir}/simple_quasi_mono.wav')/1024/1024:.1f} MB")
    
    # 5. Bruit blanc (pire cas)
    print("5️⃣ Bruit blanc (pire cas)...")
    white_noise_l = np.random.randn(len(t)) * 0.1
    white_noise_r = np.random.randn(len(t)) * 0.1
    stereo = np.vstack([white_noise_l, white_noise_r]).T
    stereo = (stereo * 32767).astype(np.int16)
    wavfile.write(f"{output_dir}/white_noise.wav", sample_rate, stereo)
    print(f"   ✅ {os.path.getsize(f'{output_dir}/white_noise.wav')/1024/1024:.1f} MB")
    
    print("\n✅ 5 fichiers WAV générés!")
    print(f"📁 Dossier: {output_dir}")

if __name__ == '__main__':
    generate_test_files()
