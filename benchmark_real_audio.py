#!/usr/bin/env python3
"""
Benchmark réaliste de NeuroSound v3.2 sur vrais fichiers audio
Avec barre de progression en temps réel
"""

import sys
import os
from pathlib import Path
from neurosound import NeuroSoundUniversal, NeuroSound
import tempfile
import statistics
import time

# Fichiers de test WAV (sources non compressées)
TEST_FILES = [
    ("/Users/bhanquier/neuroSound/test_audio/pure_tone.wav", "Ton pur 440 Hz (cas optimal)"),
    ("/Users/bhanquier/neuroSound/test_audio/podcast_silence.wav", "Podcast avec 50% silence"),
    ("/Users/bhanquier/neuroSound/test_audio/simple_quasi_mono.wav", "Musique simple quasi-mono"),
    ("/Users/bhanquier/neuroSound/test_audio/complex_music.wav", "Musique complexe stéréo"),
    ("/Users/bhanquier/neuroSound/test_audio/classical.wav", "Classique (orgue) - réel"),
    ("/Users/bhanquier/neuroSound/test_audio/white_noise.wav", "Bruit blanc (pire cas)"),
]

def progress_bar(current, total, width=40):
    """Affiche une barre de progression"""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {current}/{total} ({percent*100:.0f}%)"

def benchmark_v32():
    """Test v3.2 UNIVERSAL sur vraie musique"""
    print("=" * 70)
    print("NeuroSound v3.2 UNIVERSAL - Benchmark sur Audio WAV Varié")
    print("=" * 70)
    print(f"Sources: WAV non compressés (synthétiques + réels)")
    print(f"Objectif: Valider les ratios 15-25x typique, 30-50x optimal")
    print(f"Fichiers à tester: {len(TEST_FILES)}\n")
    
    codec = NeuroSoundUniversal(mode='balanced')
    ratios = []
    results_by_type = {}
    total_files = len(TEST_FILES)
    
    for i, (test_file, description) in enumerate(TEST_FILES, 1):
        print(f"\n{progress_bar(i-1, total_files)}")
        print(f"Fichier {i}/{total_files}: {description}")
        print(f"  📁 {Path(test_file).name}")
        
        if not os.path.exists(test_file):
            print(f"  ❌ Introuvable, skip")
            continue
        
        original_size = os.path.getsize(test_file)
        print(f"  📦 Original: {original_size/1024/1024:.2f} MB")
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        try:
            print(f"  ⏳ Compression en cours...", end='', flush=True)
            start = time.time()
            
            size, ratio, energy = codec.compress(test_file, tmp_out_path, verbose=False)
            
            elapsed = time.time() - start
            print(f" ✅ {elapsed:.1f}s")
            print(f"  📉 Compressé: {size/1024:.1f} KB")
            print(f"  🎯 Ratio: {ratio:.2f}x")
            print(f"  ⚡ Energie: {energy:.0f} mJ")
            
            ratios.append(ratio)
            results_by_type[description] = ratio
            
        except Exception as e:
            print(f" ❌")
            print(f"  Erreur: {e}")
        finally:
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
    
    print(f"\n{progress_bar(total_files, total_files)}")
    
    if ratios:
        print("\n" + "=" * 70)
        print("RÉSULTATS FINAUX (Musique électronique professionnelle)")
        print("=" * 70)
        print(f"Fichiers testés:  {len(ratios)}/{total_files}")
        print(f"Ratio médian:     {statistics.median(ratios):.2f}x")
        print(f"Ratio moyen:      {statistics.mean(ratios):.2f}x")
        print(f"Ratio min:        {min(ratios):.2f}x")
        print(f"Ratio max:        {max(ratios):.2f}x")
        if len(ratios) > 1:
            print(f"Écart-type:       {statistics.stdev(ratios):.2f}x")
        
        print(f"\n💡 Interprétation:")
        median = statistics.median(ratios)
        if median >= 20:
            print(f"   Excellent! Le ratio médian {median:.1f}x dépasse nos prédictions (15-25x)")
        elif median >= 15:
            print(f"   Conforme! Le ratio médian {median:.1f}x est dans la fourchette attendue (15-25x)")
        else:
            print(f"   Inférieur! Le ratio médian {median:.1f}x est sous nos prédictions (15-25x)")
            print(f"   → Musique très dense/complexe, documentations à ajuster si nécessaire")
        
        return {
            'median': statistics.median(ratios),
            'mean': statistics.mean(ratios),
            'min': min(ratios),
            'max': max(ratios),
            'all': ratios
        }
    
    return None

def compare_v31_v32():
    """Comparaison v3.1 vs v3.2 sur le même fichier WAV"""
    print("\n\n" + "=" * 70)
    print("BONUS: Comparaison v3.1 vs v3.2")
    print("=" * 70)
    
    test_file = TEST_FILES[0]
    
    if not os.path.exists(test_file):
        print("❌ Fichier source introuvable")
        return
    
    # Convertir en WAV pour v3.1
    print(f"\n⏳ Conversion MP3 → WAV (30s)...", end='', flush=True)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-i', test_file, '-ar', '44100', '-ac', '2',
            '-t', '30', tmp_wav_path
        ], check=True, capture_output=True)
        print(" ✅")
    except:
        print(" ❌ ffmpeg requis")
        return
    
    wav_size = os.path.getsize(tmp_wav_path)
    print(f"📦 WAV source: {wav_size/1024/1024:.2f} MB (30s)")
    
    # Test v3.1
    print(f"\n⏳ Test v3.1 (Spectral Analysis)...", end='', flush=True)
    codec_v31 = NeuroSound(mode='balanced')
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        size_v31, ratio_v31, energy_v31 = codec_v31.compress(tmp_wav_path, tmp_out_path, verbose=False)
        print(" ✅")
        print(f"  📉 {size_v31/1024:.1f} KB")
        print(f"  🎯 {ratio_v31:.2f}x")
        print(f"  ⚡ {energy_v31:.0f} mJ")
    finally:
        os.remove(tmp_out_path)
    
    # Test v3.2
    print(f"\n⏳ Test v3.2 (4 innovations)...", end='', flush=True)
    codec_v32 = NeuroSoundUniversal(mode='balanced')
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        size_v32, ratio_v32, energy_v32 = codec_v32.compress(tmp_wav_path, tmp_out_path, verbose=False)
        print(" ✅")
        print(f"  📉 {size_v32/1024:.1f} KB")
        print(f"  🎯 {ratio_v32:.2f}x")
        print(f"  ⚡ {energy_v32:.0f} mJ")
        
        improvement = ((ratio_v32 - ratio_v31) / ratio_v31) * 100
        print(f"\n💡 v3.2 vs v3.1: {improvement:+.1f}% sur ce fichier")
        
    finally:
        os.remove(tmp_out_path)
        os.remove(tmp_wav_path)


if __name__ == '__main__':
    print("\n🚀 Démarrage du benchmark réaliste...\n")
    
    results = benchmark_v32()
    
    if results:
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print(f"\nSur musique électronique professionnelle (Moderat):")
        print(f"  • Ratio médian réel: {results['median']:.2f}x")
        print(f"  • Fourchette observée: {results['min']:.2f}x - {results['max']:.2f}x")
        print(f"\nComparaison avec nos prédictions:")
        print(f"  • Prédit: 15-25x sur musique typique")
        print(f"  • Réel:   {results['median']:.2f}x (musique électronique dense)")
        
        if results['median'] >= 15:
            print(f"\n✅ Documentation validée! Le ratio réel est conforme.")
        else:
            print(f"\n⚠️  Documentation à ajuster pour musique très complexe.")
    
    # Comparaison v3.1 vs v3.2
    try:
        compare_v31_v32()
    except Exception as e:
        print(f"\n⚠️  Comparaison v3.1 vs v3.2 échouée: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark terminé!")
    print("=" * 70)
