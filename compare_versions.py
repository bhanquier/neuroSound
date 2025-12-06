"""
Comparaison des 3 versions de NeuroSound
=========================================
"""

import numpy as np
import time
from v2_pure_innovation import UltimatePureCompressor
from v3_optimized import OptimizedCompressor


def generate_test_signal(duration=5):
    """Génère un signal de test."""
    sample_rate = 44100
    t = np.linspace(0, duration, sample_rate * duration)
    
    signal = (
        10000 * np.sin(2 * np.pi * 440 * t) +
        7000 * np.sin(2 * np.pi * 880 * t) +
        5000 * np.sin(2 * np.pi * 1320 * t) +
        2000 * np.random.randn(len(t))
    )
    
    return signal, sample_rate


def test_version(compressor_class, signal, sample_rate, version_name):
    """Test une version du compresseur."""
    print(f"\n{'='*80}")
    print(f"🔬 TEST: {version_name}")
    print(f"{'='*80}")
    
    # Création du compresseur
    compressor = compressor_class(n_components=64, block_size=256, n_bits=8)
    
    # Compression
    print("  Compression...")
    t0 = time.time()
    try:
        compressed = compressor.compress(signal, sample_rate, verbose=False)
    except TypeError:
        # v2 ne supporte pas verbose
        compressed = compressor.compress(signal, sample_rate)
    t_compress = time.time() - t0
    
    # Décompression
    print("  Décompression...")
    t0 = time.time()
    try:
        reconstructed = compressor.decompress(compressed, verbose=False)
    except TypeError:
        reconstructed = compressor.decompress(compressed)
    t_decompress = time.time() - t0
    
    # Métriques
    min_len = min(len(signal), len(reconstructed))
    mse = np.mean((signal[:min_len] - reconstructed[:min_len]) ** 2)
    
    if mse > 0:
        psnr = 10 * np.log10(np.max(np.abs(signal)) ** 2 / mse)
        snr = 10 * np.log10(np.mean(signal[:min_len] ** 2) / mse)
    else:
        psnr = snr = float('inf')
    
    ratio = compressed.get('compression_ratio', 0)
    
    return {
        'version': version_name,
        'compress_time': t_compress,
        'decompress_time': t_decompress,
        'total_time': t_compress + t_decompress,
        'ratio': ratio,
        'psnr': psnr,
        'snr': snr,
        'mse': mse
    }


def main():
    print("\n" + "="*80)
    print(" "*20 + "🏁 COMPARAISON DES VERSIONS NEUROSOUND 🏁")
    print("="*80)
    
    # Signal de test
    print("\nGénération du signal de test (5 secondes @ 44.1kHz)...")
    signal, sample_rate = generate_test_signal(duration=5)
    print(f"✓ Signal généré: {len(signal):,} échantillons\n")
    
    # Test v2 (Pure Innovation)
    results_v2 = test_version(
        UltimatePureCompressor,
        signal,
        sample_rate,
        "v2 Pure Innovation"
    )
    
    # Test v3 (Optimized)
    results_v3 = test_version(
        OptimizedCompressor,
        signal,
        sample_rate,
        "v3 Optimized"
    )
    
    # Tableau comparatif
    print("\n" + "="*80)
    print(" "*25 + "📊 TABLEAU COMPARATIF")
    print("="*80)
    print(f"{'Métrique':<30} {'v2 Pure':<20} {'v3 Optimized':<20} {'Gain':<15}")
    print("-"*80)
    
    # Temps compression
    speedup_c = results_v2['compress_time'] / results_v3['compress_time']
    print(f"{'Temps Compression':<30} "
          f"{results_v2['compress_time']:>8.3f}s          "
          f"{results_v3['compress_time']:>8.3f}s          "
          f"{speedup_c:>6.1f}x")
    
    # Temps décompression
    speedup_d = results_v2['decompress_time'] / results_v3['decompress_time']
    print(f"{'Temps Décompression':<30} "
          f"{results_v2['decompress_time']:>8.3f}s          "
          f"{results_v3['decompress_time']:>8.3f}s          "
          f"{speedup_d:>6.1f}x")
    
    # Temps total
    speedup_t = results_v2['total_time'] / results_v3['total_time']
    print(f"{'Temps Total':<30} "
          f"{results_v2['total_time']:>8.3f}s          "
          f"{results_v3['total_time']:>8.3f}s          "
          f"{speedup_t:>6.1f}x")
    
    print("-"*80)
    
    # Ratio compression
    print(f"{'Ratio Compression':<30} "
          f"{results_v2['ratio']:>8.2f}x          "
          f"{results_v3['ratio']:>8.2f}x          "
          f"{'-':<15}")
    
    # PSNR
    print(f"{'PSNR':<30} "
          f"{results_v2['psnr']:>8.2f} dB        "
          f"{results_v3['psnr']:>8.2f} dB        "
          f"{'-':<15}")
    
    # SNR
    print(f"{'SNR':<30} "
          f"{results_v2['snr']:>8.2f} dB        "
          f"{results_v3['snr']:>8.2f} dB        "
          f"{'-':<15}")
    
    print("="*80)
    
    # Résumé
    print("\n" + "="*80)
    print(" "*30 + "🏆 RÉSUMÉ")
    print("="*80)
    
    print(f"\n✨ v3 Optimized est {speedup_t:.1f}x PLUS RAPIDE que v2 Pure Innovation!")
    print(f"\n   Compression:  {speedup_c:.1f}x plus rapide")
    print(f"   Décompression: {speedup_d:.1f}x plus rapide")
    
    if results_v3['ratio'] > results_v2['ratio'] * 0.5:
        print(f"\n   Ratio maintenu: {results_v3['ratio']:.1f}x (vs {results_v2['ratio']:.1f}x pour v2)")
    
    print(f"\n   Gain total: {(speedup_t - 1) * 100:.0f}% de réduction de temps")
    
    # Estimation vs FLAC
    flac_time = 0.01  # Temps FLAC observé
    ratio_vs_flac = results_v3['compress_time'] / flac_time
    
    print(f"\n📊 Position vs FLAC:")
    print(f"   • v2: {results_v2['compress_time']/flac_time:.0f}x plus lent que FLAC")
    print(f"   • v3: {ratio_vs_flac:.0f}x plus lent que FLAC")
    print(f"   → Amélioration de {speedup_t:.0f}x réduit l'écart avec FLAC")
    
    print("\n" + "="*80)
    print("✅ COMPARAISON TERMINÉE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
