"""
Comparaison Finale : Toutes les Versions NeuroSound
====================================================
"""

import subprocess
import os
import wave
import numpy as np


def generate_test_file():
    """Génère un fichier de test musical complexe."""
    print("🎼 Génération fichier test (5s, musical complexe)...")
    
    t = np.linspace(0, 5, 5*44100)
    signal = np.zeros_like(t)
    
    # Musique synthétique riche
    freqs = [110, 146.83, 196, 220, 293.66, 392, 440]
    for freq in freqs:
        signal += 0.15 * np.sin(2 * np.pi * freq * t)
        for h in [2, 3, 4]:
            signal += 0.05/h * np.sin(2 * np.pi * freq * h * t)
    
    # Enveloppe
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    signal *= envelope
    
    # Bruit musical
    noise = np.random.randn(len(t)) * 0.03
    signal += noise
    
    # Normalise
    signal = signal / np.max(np.abs(signal)) * 0.95
    signal_int16 = (signal * 32767).astype(np.int16)
    
    filename = 'test_final.wav'
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(signal_int16.tobytes())
    
    size = os.path.getsize(filename)
    print(f"   ✓ {size:,} bytes\n")
    return filename, size


def test_codec(name, compress_cmd, decompress_cmd, test_wav, is_lossless):
    """Teste un codec."""
    print(f"{'='*70}")
    print(f"🔹 {name}")
    print(f"{'='*70}")
    
    # Compression
    result = subprocess.run(compress_cmd, capture_output=True, text=True, shell=True)
    
    if result.returncode != 0:
        print(f"   ❌ Échec compression")
        return None
    
    # Trouve le fichier de sortie
    flac_file = compress_cmd.split()[-1] if not isinstance(compress_cmd, str) else compress_cmd.split()[-1]
    
    if not os.path.exists(flac_file):
        print(f"   ❌ Fichier {flac_file} non trouvé")
        return None
    
    size = os.path.getsize(flac_file)
    wav_size = os.path.getsize(test_wav)
    ratio = wav_size / size
    
    print(f"   ✅ Compressé: {size:,} bytes ({ratio:.2f}x)")
    
    # Décompression et test
    if decompress_cmd and is_lossless:
        restored = f'test_restored_{name.replace(" ", "_")}.wav'
        if isinstance(decompress_cmd, str):
            dec_cmd = decompress_cmd.replace('OUTPUT', restored)
        else:
            dec_cmd = decompress_cmd + [restored]
        
        result = subprocess.run(dec_cmd, capture_output=True, text=True, shell=True if isinstance(dec_cmd, str) else False)
        
        if result.returncode == 0 and os.path.exists(restored):
            # Vérification bit-perfect
            with wave.open(test_wav, 'rb') as w:
                orig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            with wave.open(restored, 'rb') as w:
                rest = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            
            if np.array_equal(orig, rest):
                print(f"   ✅ LOSSLESS: Bit-perfect confirmé !")
            else:
                diff = np.sum(orig != rest)
                print(f"   ⚠️  {diff:,} différences")
            
            os.unlink(restored)
    
    # Nettoyage
    if os.path.exists(flac_file):
        os.unlink(flac_file)
        # Nettoie aussi les fichiers .meta si existants
        if os.path.exists(flac_file + '.neurosound.meta'):
            os.unlink(flac_file + '.neurosound.meta')
        if os.path.exists(flac_file + '.sfmeta'):
            os.unlink(flac_file + '.sfmeta')
    
    print()
    return {'name': name, 'size': size, 'ratio': ratio, 'lossless': is_lossless}


def main():
    print("🔥" * 35)
    print("NeuroSound - Comparaison Finale de TOUTES les Versions")
    print("🔥" * 35)
    print()
    
    # Génère fichier test
    test_wav, wav_size = generate_test_file()
    
    results = []
    
    # FLAC Standard
    r = test_codec(
        "FLAC Standard",
        f"flac -8 --force -o test_std.flac {test_wav}",
        f"flac -d -f -o OUTPUT test_std.flac",
        test_wav,
        True
    )
    if r: results.append(r)
    
    # NeuroSound Hybrid (lossy)
    r = test_codec(
        "NeuroSound Hybrid (lossy)",
        f"python3 neurosound_flac_hybrid.py compress {test_wav} test_hybrid.flac 8",
        None,
        test_wav,
        False
    )
    if r: results.append(r)
    
    # NeuroSound Simple Lossless
    r = test_codec(
        "NeuroSound Simple Lossless",
        f"python3 neurosound_flac_simple_lossless.py compress {test_wav} test_simple.flac",
        f"python3 neurosound_flac_simple_lossless.py decompress test_simple.flac OUTPUT",
        test_wav,
        True
    )
    if r: results.append(r)
    
    # Tableau final
    print("="*70)
    print("🏆 TABLEAU RÉCAPITULATIF FINAL")
    print("="*70)
    print()
    print(f"{'Version':<35} {'Taille':<15} {'Ratio':<10} {'Lossless'}")
    print("-"*70)
    print(f"{'WAV Original':<35} {wav_size:>10,} bytes {'1.00x':<10} {'N/A'}")
    
    for r in results:
        ll = "✅ Oui" if r['lossless'] else "❌ Non"
        print(f"{r['name']:<35} {r['size']:>10,} bytes {r['ratio']:>5.2f}x     {ll}")
    
    print()
    print("="*70)
    
    # Gagnants
    lossless_only = [r for r in results if r['lossless']]
    if lossless_only:
        best_lossless = min(lossless_only, key=lambda x: x['size'])
        print(f"🏆 MEILLEUR LOSSLESS: {best_lossless['name']}")
        print(f"   📊 {best_lossless['ratio']:.2f}x - {best_lossless['size']:,} bytes")
        
        flac_std = next((r for r in results if 'FLAC Standard' in r['name']), None)
        if flac_std and best_lossless['name'] != 'FLAC Standard':
            gain = (1 - best_lossless['size'] / flac_std['size']) * 100
            print(f"   ✨ {gain:+.1f}% vs FLAC Standard")
    
    best_overall = min(results, key=lambda x: x['size'])
    print(f"\n🏆 MEILLEUR GLOBAL: {best_overall['name']}")
    print(f"   📊 {best_overall['ratio']:.2f}x - {best_overall['size']:,} bytes")
    
    print("="*70)
    
    # Nettoyage
    if os.path.exists(test_wav):
        os.unlink(test_wav)
    
    print("\n✅ Comparaison terminée !")


if __name__ == '__main__':
    main()
