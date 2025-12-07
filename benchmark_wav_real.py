#!/usr/bin/env python3
"""
Benchmark réaliste de NeuroSound v3.2 sur vrais fichiers audio WAV
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
    """Test v3.2 UNIVERSAL sur audio WAV varié"""
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
            print(f"  ⏳ Compression...", end='', flush=True)
            start = time.time()
            
            size, ratio, energy = codec.compress(test_file, tmp_out_path, verbose=False)
            
            elapsed = time.time() - start
            print(f" ✅ {elapsed:.1f}s")
            print(f"  📉 {size/1024:.1f} KB | 🎯 {ratio:.2f}x | ⚡ {energy:.0f} mJ")
            
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
        print("RÉSULTATS DÉTAILLÉS PAR TYPE D'AUDIO")
        print("=" * 70)
        for desc, ratio in results_by_type.items():
            print(f"{desc:45} {ratio:.2f}x")
        
        print("\n" + "=" * 70)
        print("STATISTIQUES GLOBALES")
        print("=" * 70)
        print(f"Fichiers testés:  {len(ratios)}/{total_files}")
        print(f"Ratio médian:     {statistics.median(ratios):.2f}x")
        print(f"Ratio moyen:      {statistics.mean(ratios):.2f}x")
        print(f"Ratio min:        {min(ratios):.2f}x")
        print(f"Ratio max:        {max(ratios):.2f}x")
        if len(ratios) > 1:
            print(f"Écart-type:       {statistics.stdev(ratios):.2f}x")
        
        print(f"\n💡 VALIDATION DES PRÉDICTIONS:")
        
        # Séparer optimal vs typique
        optimal_files = ["Ton pur", "Podcast avec"]
        typical_files = ["Musique simple", "Musique complexe", "Classique"]
        
        optimal_ratios = [r for d, r in results_by_type.items() if any(o in d for o in optimal_files)]
        typical_ratios = [r for d, r in results_by_type.items() if any(t in d for t in typical_files)]
        
        if optimal_ratios:
            opt_median = statistics.median(optimal_ratios)
            print(f"\n   Audio optimal (silence/quasi-mono):")
            print(f"     Mesuré: {opt_median:.1f}x médian")
            print(f"     Prédit: 30-50x")
            if opt_median >= 30:
                print(f"     ✅ CONFORME")
            elif opt_median >= 20:
                print(f"     ⚠️  Légèrement sous la cible")
            else:
                print(f"     ❌ SOUS LA PRÉDICTION")
        
        if typical_ratios:
            typ_median = statistics.median(typical_ratios)
            print(f"\n   Audio typique (musique réelle):")
            print(f"     Mesuré: {typ_median:.1f}x médian")
            print(f"     Prédit: 15-25x")
            if typ_median >= 15:
                print(f"     ✅ CONFORME")
            elif typ_median >= 10:
                print(f"     ⚠️  Légèrement sous la cible")
            else:
                print(f"     ❌ SOUS LA PRÉDICTION")
        
        return {
            'median': statistics.median(ratios),
            'mean': statistics.mean(ratios),
            'min': min(ratios),
            'max': max(ratios),
            'all': ratios,
            'by_type': results_by_type,
            'optimal_median': statistics.median(optimal_ratios) if optimal_ratios else None,
            'typical_median': statistics.median(typical_ratios) if typical_ratios else None,
        }
    
    return None


if __name__ == '__main__':
    print("\n🚀 Démarrage du benchmark réaliste sur WAV...\n")
    
    results = benchmark_v32()
    
    if results:
        print("\n" + "=" * 70)
        print("CONCLUSION FINALE")
        print("=" * 70)
        print(f"\nRésultats sur WAV non compressés:")
        print(f"  • Global médian: {results['median']:.2f}x")
        print(f"  • Fourchette: {results['min']:.2f}x - {results['max']:.2f}x")
        
        if results['optimal_median']:
            print(f"  • Optimal (silence/mono): {results['optimal_median']:.2f}x")
        if results['typical_median']:
            print(f"  • Typique (musique): {results['typical_median']:.2f}x")
        
        print(f"\nRecommandation documentation:")
        if results.get('typical_median', 0) >= 15 and results.get('optimal_median', 0) >= 30:
            print(f"  ✅ Les prédictions 15-25x (typique) et 30-50x (optimal) sont VALIDÉES")
        elif results.get('typical_median', 0) >= 10:
            print(f"  ⚠️  Ajuster à : 10-{results['typical_median']:.0f}x (typique), {results.get('optimal_median', 20):.0f}-{results.get('optimal_median', 20)*1.5:.0f}x (optimal)")
        else:
            print(f"  ❌ Réviser complètement les claims. Médian réel: {results['median']:.1f}x")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark terminé!")
    print("=" * 70)
