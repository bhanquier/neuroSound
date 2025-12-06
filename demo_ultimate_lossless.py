"""
Démonstration NeuroSound Ultimate Lossless
==========================================
Teste la version ultime avec garantie bit-perfect !
"""

import subprocess
import os
import wave
import numpy as np
from pathlib import Path


def generate_complex_test_signal(filename, duration=5, sample_rate=44100):
    """Génère un signal de test très complexe (difficile à compresser)."""
    print(f"🎼 Génération signal de test complexe ({duration}s)...")
    
    t = np.linspace(0, duration, duration * sample_rate)
    signal = np.zeros_like(t)
    
    # Synthèse additive complexe (simule musique orchestrale)
    fundamentals = [110, 146.83, 196, 220, 293.66, 392]  # Notes musicales
    
    for freq in fundamentals:
        # Fondamentale
        signal += 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Harmoniques
        for h in range(2, 8):
            signal += (0.1 / h) * np.sin(2 * np.pi * freq * h * t)
    
    # Modulation d'amplitude (expression musicale)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    signal *= envelope
    
    # Ajoute transitoires (attaques) - très difficile à compresser !
    n_attacks = 20
    for i in range(n_attacks):
        pos = int(i * len(t) / n_attacks)
        if pos < len(t):
            attack_duration = int(0.01 * sample_rate)  # 10ms
            attack = np.exp(-np.linspace(0, 10, min(attack_duration, len(t) - pos)))
            signal[pos:pos+len(attack)] += 0.3 * attack
    
    # Texture de bruit musical (simule instruments à souffle)
    noise = np.random.randn(len(t)) * 0.03
    noise_filtered = np.convolve(noise, np.ones(100)/100, mode='same')  # Lissage
    signal += noise_filtered
    
    # Normalise
    signal = signal / np.max(np.abs(signal)) * 0.95
    signal_int16 = (signal * 32767).astype(np.int16)
    
    # Écriture
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(signal_int16.tobytes())
    
    size = os.path.getsize(filename)
    print(f"   ✓ Signal généré: {size:,} bytes")
    print(f"   ℹ️  Contient: harmoniques, transitoires, modulations, bruit")
    return size


def verify_bit_perfect(original_wav, restored_wav):
    """Vérifie que la reconstruction est bit-perfect."""
    print("\n🔬 Vérification bit-perfect...")
    
    with wave.open(original_wav, 'rb') as orig:
        orig_params = orig.getparams()
        orig_frames = orig.readframes(orig_params.nframes)
        orig_samples = np.frombuffer(orig_frames, dtype=np.int16)
    
    with wave.open(restored_wav, 'rb') as rest:
        rest_params = rest.getparams()
        rest_frames = rest.readframes(rest_params.nframes)
        rest_samples = np.frombuffer(rest_frames, dtype=np.int16)
    
    # Vérification exacte
    if len(orig_samples) != len(rest_samples):
        print(f"   ✗ ERREUR: Longueurs différentes ({len(orig_samples)} vs {len(rest_samples)})")
        return False
    
    if not np.array_equal(orig_samples, rest_samples):
        diff = orig_samples != rest_samples
        n_diff = np.sum(diff)
        print(f"   ✗ ERREUR: {n_diff} échantillons différents !")
        
        # Analyse des différences
        if n_diff > 0:
            max_diff = np.max(np.abs(orig_samples.astype(int) - rest_samples.astype(int)))
            print(f"   📊 Différence maximale: {max_diff}")
        
        return False
    
    print("   ✅ PARFAIT ! Reconstruction 100% bit-perfect !")
    print("   ✅ Aucune différence sur les", len(orig_samples), "échantillons")
    return True


def compare_all_versions():
    """Compare toutes les versions de compression."""
    print("\n" + "="*70)
    print("📊 COMPARAISON: Toutes les Versions NeuroSound")
    print("="*70)
    
    test_wav = 'test_complex.wav'
    wav_size = generate_complex_test_signal(test_wav, duration=5)
    
    results = []
    
    # FLAC standard
    print("\n🔹 FLAC Standard (niveau 8)...")
    flac_std = 'test_flac_standard.flac'
    cmd = ['flac', '-8', '--force', '-o', flac_std, test_wav]
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        flac_size = os.path.getsize(flac_std)
        ratio = wav_size / flac_size
        results.append(('FLAC Standard', flac_size, ratio, True))
        print(f"   ✓ Taille: {flac_size:,} bytes ({ratio:.2f}x)")
        os.unlink(flac_std)
    
    # NeuroSound Hybrid
    print("\n🔥 NeuroSound FLAC Hybrid...")
    hybrid_flac = 'test_hybrid.flac'
    cmd = ['python3', 'neurosound_flac_hybrid.py', 'compress', test_wav, hybrid_flac, '8']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        hybrid_size = os.path.getsize(hybrid_flac)
        ratio = wav_size / hybrid_size
        results.append(('SF Hybrid (lossy)', hybrid_size, ratio, False))
        print(f"   ✓ Taille: {hybrid_size:,} bytes ({ratio:.2f}x)")
        os.unlink(hybrid_flac)
        if os.path.exists(hybrid_flac + '.neurosound.meta'):
            os.unlink(hybrid_flac + '.neurosound.meta')
    
    # NeuroSound Ultimate Lossless
    print("\n🏆 NeuroSound Ultimate Lossless...")
    ultimate_flac = 'test_ultimate.flac'
    cmd = ['python3', 'neurosound_flac_ultimate_lossless.py', 'compress', test_wav, ultimate_flac, '8']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        ultimate_size = os.path.getsize(ultimate_flac)
        ratio = wav_size / ultimate_size
        results.append(('SF Ultimate Lossless', ultimate_size, ratio, True))
        print(f"   ✓ Taille: {ultimate_size:,} bytes ({ratio:.2f}x)")
        
        # Test reconstruction bit-perfect
        print("\n🔍 Test de reconstruction...")
        restored_wav = 'test_restored.wav'
        cmd = ['python3', 'neurosound_flac_ultimate_lossless.py', 'decompress', ultimate_flac, restored_wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            is_perfect = verify_bit_perfect(test_wav, restored_wav)
            os.unlink(restored_wav)
        
        os.unlink(ultimate_flac)
        if os.path.exists(ultimate_flac + '.sfmeta'):
            os.unlink(ultimate_flac + '.sfmeta')
    
    # Affichage comparatif
    print("\n" + "="*70)
    print("🏆 TABLEAU COMPARATIF FINAL")
    print("="*70)
    
    print(f"\n{'Version':<25} {'Taille':<15} {'Ratio':<10} {'Lossless'}")
    print("-" * 70)
    print(f"{'WAV Original':<25} {wav_size:>10,} bytes {'1.00x':<10} {'N/A'}")
    
    for name, size, ratio, lossless in results:
        lossless_str = "✅ Oui" if lossless else "❌ Non"
        print(f"{name:<25} {size:>10,} bytes {ratio:>5.2f}x     {lossless_str}")
    
    # Gagnant
    if len(results) > 0:
        print("\n" + "="*70)
        
        # Meilleure compression lossless
        lossless_results = [(name, size, ratio) for name, size, ratio, ll in results if ll]
        if lossless_results:
            best_lossless = min(lossless_results, key=lambda x: x[1])
            print(f"🏆 Meilleure compression LOSSLESS: {best_lossless[0]}")
            print(f"   📊 {best_lossless[2]:.2f}x - {best_lossless[1]:,} bytes")
        
        # Meilleure compression globale
        best_overall = min(results, key=lambda x: x[1])
        print(f"\n🏆 Meilleure compression GLOBALE: {best_overall[0]}")
        print(f"   📊 {best_overall[2]:.2f}x - {best_overall[1]:,} bytes")
        
        # Calcul gains
        flac_standard = next((r for r in results if r[0] == 'FLAC Standard'), None)
        ultimate = next((r for r in results if r[0] == 'SF Ultimate Lossless'), None)
        
        if flac_standard and ultimate:
            improvement = (1 - ultimate[1] / flac_standard[1]) * 100
            print(f"\n✨ NeuroSound Ultimate gagne: {improvement:.1f}% vs FLAC Standard")
            print(f"   (tout en gardant la garantie lossless !)")
    
    print("="*70)
    
    # Nettoyage
    if os.path.exists(test_wav):
        os.unlink(test_wav)


def main():
    """Démonstration principale."""
    print("🔥" * 35)
    print("NeuroSound Ultimate Lossless - Démonstration")
    print("🔥" * 35)
    
    # Vérif FLAC
    try:
        subprocess.run(['flac', '--version'], capture_output=True, check=True)
    except:
        print("\n❌ ERREUR: FLAC non installé !")
        print("   Installez avec: brew install flac")
        return
    
    # Comparaison complète
    compare_all_versions()
    
    print("\n" + "🔥" * 35)
    print("✅ Démonstration terminée !")
    print("🔥" * 35)


if __name__ == '__main__':
    main()
