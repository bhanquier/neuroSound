#!/usr/bin/env python3
"""
Benchmark réaliste de NeuroSound v3.2 sur divers types d'audio
"""

import sys
import os
from pathlib import Path
from neurosound import NeuroSoundUniversal, NeuroSound
import tempfile
import statistics

# Fichiers de test (musique électronique complexe)
TEST_FILES = [
    "/Users/bhanquier/Music/Music/Media.localized/Music/Moderat/III/1-01 Eating Hooks.mp3",
    "/Users/bhanquier/Music/Music/Media.localized/Music/Moderat/III/1-04 Ghostmother.mp3",
    "/Users/bhanquier/Music/Music/Media.localized/Music/Moderat/III/1-05 Reminder.mp3",
    "/Users/bhanquier/Music/Music/Media.localized/Music/Moderat/III/1-06 The Fool.mp3",
    "/Users/bhanquier/Music/Music/Media.localized/Music/Moderat/III/1-08 Animal Trails.mp3",
]

def benchmark_v32():
    """Test v3.2 UNIVERSAL sur vraie musique"""
    print("=" * 60)
    print("NeuroSound v3.2 UNIVERSAL - Benchmark Réaliste")
    print("=" * 60)
    print(f"\nType: Musique électronique (Moderat)")
    print(f"Caractéristiques: Stéréo large, dense, complexe\n")
    
    codec = NeuroSoundUniversal(mode='balanced')
    ratios = []
    
    for i, test_file in enumerate(TEST_FILES, 1):
        if not os.path.exists(test_file):
            print(f"❌ Fichier {i} introuvable: {Path(test_file).name}")
            continue
        
        filename = Path(test_file).name
        original_size = os.path.getsize(test_file)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        try:
            print(f"\n[{i}/5] {filename[:50]}...")
            print(f"  Original: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
            
            size, ratio, energy = codec.compress(test_file, tmp_out_path, verbose=False)
            
            print(f"  Compressed: {size:,} bytes ({size/1024:.1f} KB)")
            print(f"  Ratio: {ratio:.2f}x")
            print(f"  Energy: {energy:.0f} mJ")
            
            ratios.append(ratio)
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
        finally:
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
    
    if ratios:
        print("\n" + "=" * 60)
        print("RÉSULTATS AGRÉGÉS (Musique électronique complexe)")
        print("=" * 60)
        print(f"Médiane:  {statistics.median(ratios):.2f}x")
        print(f"Moyenne:  {statistics.mean(ratios):.2f}x")
        print(f"Min:      {min(ratios):.2f}x")
        print(f"Max:      {max(ratios):.2f}x")
        print(f"Écart-type: {statistics.stdev(ratios):.2f}x" if len(ratios) > 1 else "")
        
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
    print("\n\n" + "=" * 60)
    print("Comparaison v3.1 vs v3.2")
    print("=" * 60)
    
    # Créer un fichier WAV de test simple
    test_file = TEST_FILES[0]
    
    # Convertir en WAV pour v3.1
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    
    # ffmpeg pour convertir
    import subprocess
    subprocess.run([
        'ffmpeg', '-i', test_file, '-ar', '44100', '-ac', '2',
        '-t', '30',  # Limiter à 30s pour le test
        tmp_wav_path
    ], check=True, capture_output=True)
    
    wav_size = os.path.getsize(tmp_wav_path)
    
    # Test v3.1
    print(f"\nTest sur 30s de musique électronique")
    print(f"WAV source: {wav_size:,} bytes ({wav_size/1024/1024:.2f} MB)")
    
    codec_v31 = NeuroSound(mode='balanced')
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        size_v31, ratio_v31, energy_v31 = codec_v31.compress(tmp_wav_path, tmp_out_path, verbose=False)
        print(f"\nv3.1 (Spectral Analysis):")
        print(f"  Ratio: {ratio_v31:.2f}x")
        print(f"  Size: {size_v31:,} bytes")
        print(f"  Energy: {energy_v31:.0f} mJ")
    finally:
        os.remove(tmp_out_path)
    
    # Test v3.2
    codec_v32 = NeuroSoundUniversal(mode='balanced')
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        size_v32, ratio_v32, energy_v32 = codec_v32.compress(tmp_wav_path, tmp_out_path, verbose=False)
        print(f"\nv3.2 UNIVERSAL (4 innovations):")
        print(f"  Ratio: {ratio_v32:.2f}x")
        print(f"  Size: {size_v32:,} bytes")
        print(f"  Energy: {energy_v32:.0f} mJ")
        
        improvement = ((ratio_v32 - ratio_v31) / ratio_v31) * 100
        print(f"\nAmélioration v3.2 vs v3.1: {improvement:+.1f}%")
        
    finally:
        os.remove(tmp_out_path)
        os.remove(tmp_wav_path)


if __name__ == '__main__':
    results = benchmark_v32()
    
    if results:
        print("\n" + "=" * 60)
        print("CONCLUSION SCIENTIFIQUE")
        print("=" * 60)
        print(f"\nSur musique électronique complexe (stéréo large, dense):")
        print(f"  • Ratio médian: {results['median']:.2f}x")
        print(f"  • Fourchette: {results['min']:.2f}x - {results['max']:.2f}x")
        print(f"\nLe ratio 80.94x était obtenu sur:")
        print(f"  • Audio simple avec 50% de silence")
        print(f"  • Stéréo quasi-mono (correlation 100%)")
        print(f"  • Cas optimal, NON représentatif")
        print(f"\nRatio réaliste attendu sur audio varié: ~{results['median']:.0f}-{results['mean']:.0f}x")
    
    # Comparaison
    try:
        compare_v31_v32()
    except Exception as e:
        print(f"\n⚠️  Comparaison v3.1 vs v3.2 échouée: {e}")
