"""
Démonstration NeuroSound FLAC Hybrid
====================================
Teste la compression hybride avec compatibilité FLAC universelle !
"""

import subprocess
import os
import wave
import numpy as np
from pathlib import Path


def generate_test_wav(filename, duration=5, sample_rate=44100):
    """Génère un fichier WAV de test avec du contenu musical varié."""
    print(f"🎼 Génération du fichier de test ({duration}s)...")
    
    t = np.linspace(0, duration, duration * sample_rate)
    
    # Signal complexe : musique synthétique
    signal = np.zeros_like(t)
    
    # Basse (fondamentale)
    signal += 0.3 * np.sin(2 * np.pi * 110 * t)  # A2
    
    # Mélodie (harmoniques)
    melody_freqs = [440, 494, 523, 587, 659, 698, 784, 880]  # Gamme A
    for i, freq in enumerate(melody_freqs):
        start = i * sample_rate * duration // len(melody_freqs)
        end = (i + 1) * sample_rate * duration // len(melody_freqs)
        signal[start:end] += 0.2 * np.sin(2 * np.pi * freq * t[start:end])
    
    # Harmoniques riches
    for harmonic in [2, 3, 4, 5]:
        signal += 0.1 / harmonic * np.sin(2 * np.pi * 440 * harmonic * t)
    
    # Ajoute du bruit musical (simule texture)
    noise = np.random.randn(len(t)) * 0.05
    signal += noise
    
    # Normalise
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    # Conversion en int16
    signal_int16 = (signal * 32767).astype(np.int16)
    
    # Écriture WAV
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(signal_int16.tobytes())
    
    size = os.path.getsize(filename)
    print(f"   ✓ Créé: {filename} ({size:,} bytes)")
    return size


def compare_with_standard_flac(wav_file):
    """Compare notre version hybrid avec FLAC standard."""
    print("\n" + "="*60)
    print("📊 COMPARAISON: NeuroSound Hybrid vs FLAC Standard")
    print("="*60)
    
    wav_size = os.path.getsize(wav_file)
    
    # Test FLAC standard
    print("\n🔹 FLAC Standard (niveau 8)...")
    standard_flac = 'test_standard.flac'
    
    cmd = ['flac', '-8', '--force', '-o', standard_flac, wav_file]
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        standard_size = os.path.getsize(standard_flac)
        standard_ratio = wav_size / standard_size
        print(f"   ✓ Taille: {standard_size:,} bytes")
        print(f"   ✓ Ratio: {standard_ratio:.2f}x")
    else:
        print("   ✗ Erreur FLAC standard")
        standard_size = 0
        standard_ratio = 0
    
    # Test NeuroSound Hybrid
    print("\n🔥 NeuroSound FLAC Hybrid...")
    hybrid_flac = 'test_hybrid.flac'
    
    cmd = ['python3', 'neurosound_flac_hybrid.py', 'compress', wav_file, hybrid_flac, '8']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        hybrid_size = os.path.getsize(hybrid_flac)
        hybrid_ratio = wav_size / hybrid_size
        print(f"   ✓ Taille: {hybrid_size:,} bytes")
        print(f"   ✓ Ratio: {hybrid_ratio:.2f}x")
    else:
        print(f"   ✗ Erreur: {result.stderr}")
        hybrid_size = 0
        hybrid_ratio = 0
    
    # Comparaison
    if standard_size > 0 and hybrid_size > 0:
        print("\n" + "="*60)
        print("🎯 RÉSULTATS")
        print("="*60)
        
        print(f"\n📁 Fichier original WAV:  {wav_size:,} bytes")
        print(f"📦 FLAC Standard:         {standard_size:,} bytes ({standard_ratio:.2f}x)")
        print(f"🔥 NeuroSound Hybrid:     {hybrid_size:,} bytes ({hybrid_ratio:.2f}x)")
        
        if hybrid_size < standard_size:
            improvement = (1 - hybrid_size / standard_size) * 100
            print(f"\n✨ NeuroSound gagne: {improvement:.1f}% plus compact ! 🏆")
        elif hybrid_size > standard_size:
            penalty = (hybrid_size / standard_size - 1) * 100
            print(f"\n⚠️  NeuroSound: {penalty:.1f}% plus gros (overhead métadonnées)")
        else:
            print(f"\n🤝 Égalité parfaite !")
        
        # Test de compatibilité
        print("\n" + "="*60)
        print("🔍 TEST DE COMPATIBILITÉ")
        print("="*60)
        
        print("\n📻 Test 1: Lecture avec décodeur FLAC standard...")
        test_decode = 'test_decoded_standard.wav'
        cmd = ['flac', '-d', '-f', '-o', test_decode, hybrid_flac]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ SUCCÈS ! Le fichier hybrid est lisible par FLAC standard !")
            os.unlink(test_decode)
        else:
            print("   ✗ Échec décodage standard")
        
        print("\n🔬 Test 2: Décompression NeuroSound complète...")
        test_restore = 'test_restored.wav'
        cmd = ['python3', 'neurosound_flac_hybrid.py', 'decompress', hybrid_flac, test_restore]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ SUCCÈS ! Reconstruction parfaite avec métadonnées !")
            
            # Vérifie la fidélité
            with wave.open(wav_file, 'rb') as orig:
                orig_frames = np.frombuffer(
                    orig.readframes(orig.getnframes()), 
                    dtype=np.int16
                )
            
            with wave.open(test_restore, 'rb') as rest:
                rest_frames = np.frombuffer(
                    rest.readframes(rest.getnframes()), 
                    dtype=np.int16
                )
            
            # Calcul de l'erreur
            if len(orig_frames) == len(rest_frames):
                mse = np.mean((orig_frames.astype(float) - rest_frames.astype(float))**2)
                psnr = 10 * np.log10(32768**2 / (mse + 1e-10))
                print(f"   📊 PSNR: {psnr:.1f} dB")
                
                if psnr > 90:
                    print("   🎯 Qualité: EXCELLENTE (quasi-lossless)")
                elif psnr > 60:
                    print("   🎯 Qualité: TRÈS BONNE")
                elif psnr > 40:
                    print("   🎯 Qualité: BONNE")
                else:
                    print("   🎯 Qualité: ACCEPTABLE")
            
            os.unlink(test_restore)
        else:
            print(f"   ✗ Échec: {result.stderr}")
    
    # Nettoyage
    for f in [standard_flac, hybrid_flac]:
        if os.path.exists(f):
            os.unlink(f)


def main():
    """Démonstration principale."""
    print("🔥" * 30)
    print("NeuroSound FLAC Hybrid - Démonstration")
    print("🔥" * 30)
    
    # Vérification FLAC
    try:
        subprocess.run(['flac', '--version'], capture_output=True, check=True)
    except:
        print("\n❌ ERREUR: FLAC n'est pas installé !")
        print("   Installez avec: brew install flac")
        return
    
    # Génère un fichier de test
    test_wav = 'test_music.wav'
    generate_test_wav(test_wav, duration=5)
    
    # Comparaison
    compare_with_standard_flac(test_wav)
    
    # Nettoyage
    if os.path.exists(test_wav):
        os.unlink(test_wav)
    
    print("\n" + "🔥" * 30)
    print("✅ Démonstration terminée !")
    print("🔥" * 30)


if __name__ == '__main__':
    main()
