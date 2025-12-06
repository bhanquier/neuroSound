"""
Benchmark: NeuroSound v2 vs FLAC
================================
Comparaison complète des performances
"""

import numpy as np
import time
import wave
import struct
import subprocess
import os
from v2_pure_innovation import (
    UltimatePureCompressor,
    load_wav,
    save_wav,
    compute_metrics
)


def generate_test_signals():
    """Génère différents types de signaux de test."""
    sample_rate = 44100
    duration = 5  # 5 secondes
    
    signals = {}
    
    # 1. Musique synthétique (harmoniques riches)
    print("  Génération: Musique synthétique...")
    t = np.linspace(0, duration, sample_rate * duration)
    music = (
        10000 * np.sin(2 * np.pi * 440 * t) +      # La4 (fondamentale)
        7000 * np.sin(2 * np.pi * 880 * t) +       # La5 (octave)
        5000 * np.sin(2 * np.pi * 1320 * t) +      # Mi6 (quinte)
        3000 * np.sin(2 * np.pi * 660 * t) +       # Mi5
        2000 * np.sin(2 * np.pi * 220 * t) +       # La3 (sous-octave)
        1500 * np.sin(2 * np.pi * 1760 * t) +      # La6
        1000 * np.random.randn(len(t))              # Bruit de fond
    )
    # Enveloppe ADSR simple
    envelope = np.concatenate([
        np.linspace(0, 1, len(t)//10),              # Attack
        np.ones(len(t)//2),                         # Sustain
        np.linspace(1, 0.3, len(t) - len(t)//10 - len(t)//2)  # Release
    ])
    signals['music'] = music * envelope
    
    # 2. Parole simulée (modulation AM)
    print("  Génération: Parole simulée...")
    t2 = np.linspace(0, duration, sample_rate * duration)
    formants = (
        8000 * np.sin(2*np.pi*800*t2) +   # F1
        6000 * np.sin(2*np.pi*1200*t2) +  # F2
        4000 * np.sin(2*np.pi*2500*t2)    # F3
    )
    modulation = 1 + 0.7 * np.sin(2*np.pi*4*t2)  # Modulation 4Hz
    signals['speech'] = formants * modulation
    
    # 3. Silence avec bruit léger (pire cas pour compression)
    print("  Génération: Silence...")
    signals['silence'] = np.random.randn(sample_rate * duration) * 100
    
    # 4. Bruit blanc pur (non compressible)
    print("  Génération: Bruit blanc...")
    signals['noise'] = np.random.randn(sample_rate * duration) * 15000
    
    # 5. Tonalité pure (meilleur cas)
    print("  Génération: Tonalité pure...")
    signals['tone'] = 20000 * np.sin(2 * np.pi * 1000 * t)
    
    return signals, sample_rate


def test_flac(signal, sample_rate, test_name):
    """Test de compression FLAC."""
    # Sauvegarde temporaire
    class Params:
        nchannels = 1
        sampwidth = 2
        framerate = sample_rate
        nframes = len(signal)
    
    params = Params()
    
    # Fichiers temporaires
    wav_file = f'temp_{test_name}.wav'
    flac_file = f'temp_{test_name}.flac'
    decoded_file = f'temp_{test_name}_decoded.wav'
    
    try:
        # Sauvegarde WAV original
        save_wav(wav_file, signal, params)
        original_size = os.path.getsize(wav_file)
        
        # Compression FLAC
        t0 = time.time()
        result = subprocess.run(
            ['flac', '--silent', '--best', '-f', wav_file, '-o', flac_file],
            capture_output=True,
            timeout=30
        )
        t_compress = time.time() - t0
        
        if result.returncode != 0:
            print(f"    ⚠️  FLAC non disponible ou erreur")
            return None
        
        compressed_size = os.path.getsize(flac_file)
        
        # Décompression
        t0 = time.time()
        subprocess.run(
            ['flac', '--silent', '-d', '-f', flac_file, '-o', decoded_file],
            capture_output=True,
            timeout=30
        )
        t_decompress = time.time() - t0
        
        # Charge le résultat
        reconstructed, _ = load_wav(decoded_file)
        
        # Métriques
        ratio = original_size / compressed_size
        metrics = compute_metrics(signal, reconstructed)
        
        return {
            'name': 'FLAC',
            'compress_time': t_compress,
            'decompress_time': t_decompress,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': ratio,
            'psnr': metrics['psnr'],
            'snr': metrics['snr'],
            'mse': metrics['mse']
        }
        
    except subprocess.TimeoutExpired:
        print(f"    ⚠️  FLAC timeout")
        return None
    except FileNotFoundError:
        print(f"    ⚠️  FLAC non installé (brew install flac)")
        return None
    finally:
        # Nettoyage
        for f in [wav_file, flac_file, decoded_file]:
            if os.path.exists(f):
                os.remove(f)


def test_neurosound(signal, sample_rate, test_name, config='balanced'):
    """Test de compression NeuroSound."""
    configs = {
        'fast': {'n_components': 32, 'block_size': 128, 'n_bits': 7},
        'balanced': {'n_components': 64, 'block_size': 256, 'n_bits': 8},
        'quality': {'n_components': 128, 'block_size': 512, 'n_bits': 10}
    }
    
    params = configs[config]
    compressor = UltimatePureCompressor(**params)
    
    # Compression
    t0 = time.time()
    compressed = compressor.compress(signal, sample_rate)
    t_compress = time.time() - t0
    
    # Décompression
    t0 = time.time()
    reconstructed = compressor.decompress(compressed)
    t_decompress = time.time() - t0
    
    # Taille estimée (en bytes)
    # Compte les données principales
    blocks_data = compressed.get('blocks', [])
    
    # Estime la taille totale
    if blocks_data:
        # Compte indices + norms
        total_indices = sum(block['indices'].nbytes for block in blocks_data)
        compressed_size = total_indices + 2048  # + overhead métadonnées
    else:
        # Fallback sur ratio
        compressed_size = int(len(signal) * 2 / compressed.get('compression_ratio', 1))
    
    original_size = len(signal) * 2  # 16-bit audio
    
    # Métriques
    metrics = compute_metrics(signal, reconstructed)
    
    return {
        'name': f'NeuroSound ({config})',
        'compress_time': t_compress,
        'decompress_time': t_decompress,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': compressed.get('compression_ratio', original_size / compressed_size),
        'psnr': metrics['psnr'],
        'snr': metrics['snr'],
        'mse': metrics['mse']
    }


def print_comparison_table(results):
    """Affiche un tableau comparatif."""
    print("\n" + "="*100)
    print(" "*35 + "TABLEAU COMPARATIF")
    print("="*100)
    
    for test_name, test_results in results.items():
        print(f"\n📊 TEST: {test_name.upper()}")
        print("-"*100)
        print(f"{'Compresseur':<20} {'Temps C.':<12} {'Temps D.':<12} {'Ratio':<10} "
              f"{'PSNR':<12} {'SNR':<12} {'Taille':<15}")
        print("-"*100)
        
        for res in test_results:
            if res is None:
                continue
            
            size_str = f"{res['compressed_size']/1024:.1f}KB"
            
            print(f"{res['name']:<20} "
                  f"{res['compress_time']:>8.3f}s    "
                  f"{res['decompress_time']:>8.3f}s    "
                  f"{res['ratio']:>6.2f}x    "
                  f"{res['psnr']:>8.1f}dB   "
                  f"{res['snr']:>8.1f}dB   "
                  f"{size_str:>12}")
        print("-"*100)


def compute_winner(results):
    """Détermine le vainqueur pour chaque catégorie."""
    print("\n" + "="*100)
    print(" "*40 + "🏆 VAINQUEURS 🏆")
    print("="*100)
    
    categories = {
        'Vitesse Compression': lambda r: -r['compress_time'],
        'Vitesse Décompression': lambda r: -r['decompress_time'],
        'Ratio Compression': lambda r: r['ratio'],
        'Qualité (PSNR)': lambda r: r['psnr'] if r['psnr'] != float('inf') else 0
    }
    
    for test_name, test_results in results.items():
        valid_results = [r for r in test_results if r is not None]
        if not valid_results:
            continue
        
        print(f"\n📊 {test_name.upper()}:")
        print("-"*100)
        
        for cat_name, key_func in categories.items():
            winner = max(valid_results, key=key_func)
            print(f"  {cat_name:<30} → {winner['name']}")


def main():
    print("\n" + "="*100)
    print(" "*30 + "🎵 BENCHMARK: NEUROSOUND vs FLAC 🎵")
    print("="*100)
    print("\nGénération des signaux de test...")
    print("-"*100)
    
    signals, sample_rate = generate_test_signals()
    
    print(f"\n✅ {len(signals)} signaux générés ({5}s @ {sample_rate}Hz)\n")
    
    # Tests
    results = {}
    
    for signal_name, signal in signals.items():
        print(f"\n{'='*100}")
        print(f"🔬 TEST: {signal_name.upper()}")
        print(f"{'='*100}")
        
        test_results = []
        
        # Test FLAC
        print("\n  [1/4] FLAC compression...")
        flac_res = test_flac(signal, sample_rate, signal_name)
        if flac_res:
            test_results.append(flac_res)
            print(f"    ✓ Ratio: {flac_res['ratio']:.2f}x | Temps: {flac_res['compress_time']:.3f}s")
        
        # Test NeuroSound (3 configs)
        for i, config in enumerate(['fast', 'balanced', 'quality'], 2):
            print(f"\n  [{i}/4] NeuroSound ({config})...")
            ns_res = test_neurosound(signal, sample_rate, signal_name, config)
            test_results.append(ns_res)
            print(f"    ✓ Ratio: {ns_res['ratio']:.2f}x | Temps: {ns_res['compress_time']:.3f}s")
        
        results[signal_name] = test_results
    
    # Affichage comparatif
    print_comparison_table(results)
    compute_winner(results)
    
    # Résumé global
    print("\n" + "="*100)
    print(" "*35 + "📈 RÉSUMÉ GLOBAL")
    print("="*100)
    
    all_flac = [r for test in results.values() for r in test if r and r['name'] == 'FLAC']
    all_ns_balanced = [r for test in results.values() for r in test 
                       if r and 'balanced' in r['name']]
    
    if all_flac and all_ns_balanced:
        print("\nMoyennes (FLAC vs NeuroSound Balanced):")
        print("-"*100)
        
        flac_avg_ratio = np.mean([r['ratio'] for r in all_flac])
        ns_avg_ratio = np.mean([r['ratio'] for r in all_ns_balanced])
        
        flac_avg_time = np.mean([r['compress_time'] for r in all_flac])
        ns_avg_time = np.mean([r['compress_time'] for r in all_ns_balanced])
        
        print(f"  Ratio moyen:")
        print(f"    • FLAC:       {flac_avg_ratio:.2f}x")
        print(f"    • NeuroSound: {ns_avg_ratio:.2f}x")
        print(f"    → Différence: {(ns_avg_ratio - flac_avg_ratio):.2f}x "
              f"({((ns_avg_ratio/flac_avg_ratio - 1)*100):+.1f}%)")
        
        print(f"\n  Temps compression moyen:")
        print(f"    • FLAC:       {flac_avg_time:.3f}s")
        print(f"    • NeuroSound: {ns_avg_time:.3f}s")
        print(f"    → NeuroSound est {flac_avg_time/ns_avg_time:.1f}x "
              f"{'plus rapide' if ns_avg_time < flac_avg_time else 'plus lent'}")
    
    print("\n" + "="*100)
    print(" "*30 + "✨ BENCHMARK TERMINÉ ✨")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
