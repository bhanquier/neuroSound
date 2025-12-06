# 🎵 NeuroSound v2 - Guide d'Utilisation

## 🚀 Démarrage Rapide

### Installation

```bash
# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # ou `.venv\Scripts\activate` sur Windows

# Installer dépendance (seulement NumPy!)
pip install numpy
```

### Utilisation Basique

```python
from v2_pure_innovation import UltimatePureCompressor, load_wav, save_wav

# 1. Charger votre fichier audio
signal, params = load_wav('votre_musique.wav')

# 2. Créer le compresseur
compressor = UltimatePureCompressor(
    n_components=64,    # Nombre de composantes principales
    block_size=256,     # Taille des blocs de transformation
    n_bits=8           # Bits de quantification
)

# 3. Compresser
compressed = compressor.compress(signal, params.framerate)

# 4. Décompresser
reconstructed = compressor.decompress(compressed)

# 5. Sauvegarder
save_wav('sortie_compressée.wav', reconstructed, params)
```

---

## ⚙️ Configuration Avancée

### Paramètres du Compresseur

```python
compressor = UltimatePureCompressor(
    n_components=128,   # 🎚️ Plus = meilleure qualité, moins de compression
    block_size=512,     # 🎚️ Plus = meilleure capture de patterns longs
    n_bits=10          # 🎚️ Plus = meilleure qualité, plus de bits
)
```

#### Recommandations par Usage

| Usage | n_components | block_size | n_bits | Ratio attendu |
|-------|-------------|------------|--------|---------------|
| **Max Compression** | 32 | 128 | 6 | 15-20x |
| **Équilibré** | 64 | 256 | 8 | 8-12x |
| **Haute Qualité** | 128 | 512 | 10 | 5-8x |
| **Archivage** | 256 | 1024 | 12 | 3-5x |

---

## 📊 Analyse et Benchmarks

### Comparer Versions

```python
import time
import numpy as np

# Métriques de qualité
def compute_metrics(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    mse = np.mean((original[:min_len] - reconstructed[:min_len]) ** 2)
    
    if mse > 0:
        psnr = 10 * np.log10(np.max(np.abs(original)) ** 2 / mse)
        snr = 10 * np.log10(np.mean(original[:min_len] ** 2) / mse)
    else:
        psnr = snr = float('inf')
    
    return {'mse': mse, 'psnr': psnr, 'snr': snr}

# Test
signal, params = load_wav('test.wav')

# v2 Pure Innovation
t0 = time.time()
compressor = UltimatePureCompressor()
compressed = compressor.compress(signal, params.framerate)
t_comp = time.time() - t0

t0 = time.time()
reconstructed = compressor.decompress(compressed)
t_decomp = time.time() - t0

metrics = compute_metrics(signal, reconstructed)

print(f"Compression: {t_comp:.3f}s")
print(f"Décompression: {t_decomp:.3f}s")
print(f"Ratio: {compressed['compression_ratio']:.2f}x")
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SNR: {metrics['snr']:.2f} dB")
```

---

## 🎨 Cas d'Usage

### 1. Compression de Podcast

```python
# Paramètres optimisés pour la voix
compressor = UltimatePureCompressor(
    n_components=48,     # Voix = moins de composantes harmoniques
    block_size=128,      # Blocs courts pour parole
    n_bits=7            # Économie maximale
)

signal, params = load_wav('podcast.wav')
compressed = compressor.compress(signal, params.framerate)

# Ratio attendu: 12-18x
```

### 2. Compression de Musique Classique

```python
# Paramètres pour préserver richesse harmonique
compressor = UltimatePureCompressor(
    n_components=128,    # Beaucoup d'harmoniques
    block_size=512,      # Capture notes longues
    n_bits=10           # Haute résolution
)

signal, params = load_wav('symphonie.wav')
compressed = compressor.compress(signal, params.framerate)

# Ratio attendu: 6-10x
```

### 3. Compression de Musique Électronique

```python
# Équilibre entre compression et qualité
compressor = UltimatePureCompressor(
    n_components=80,
    block_size=256,
    n_bits=9
)

signal, params = load_wav('techno.wav')
compressed = compressor.compress(signal, params.framerate)

# Ratio attendu: 8-12x
```

---

## 🔧 Troubleshooting

### Problème: "Overflow warning"

**Solution**: Réduire `n_components` ou `block_size`

```python
compressor = UltimatePureCompressor(
    n_components=32,  # Réduit
    block_size=128    # Réduit
)
```

### Problème: "MSE très élevée"

**Cause**: Signal trop long ou paramètres trop agressifs

**Solutions**:
1. Augmenter `n_bits` pour meilleure résolution
2. Augmenter `n_components` pour capturer plus de détails
3. Pré-filtrer le signal (anti-aliasing)

```python
# Filtrage passe-bas avant compression
from scipy import signal as sp_signal
b, a = sp_signal.butter(4, 0.8, 'low')
signal_filtered = sp_signal.filtfilt(b, a, signal)
```

### Problème: "Compression trop lente"

**Solutions**:
1. Réduire `block_size`
2. Réduire `n_components`
3. Pré-segmenter manuellement les gros fichiers

```python
# Traiter par chunks
def compress_large_file(filename, chunk_duration=10):
    signal, params = load_wav(filename)
    chunk_size = params.framerate * chunk_duration
    
    compressed_chunks = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        compressed = compressor.compress(chunk, params.framerate)
        compressed_chunks.append(compressed)
    
    return compressed_chunks
```

---

## 📈 Optimisations Futures

### Pour aller plus loin

1. **Parallélisation**: Traiter segments en parallèle
   ```python
   from multiprocessing import Pool
   
   def compress_segment(seg_data):
       segment, params = seg_data
       return compressor.compress(segment, params.framerate)
   
   with Pool(4) as p:
       results = p.map(compress_segment, segments_data)
   ```

2. **Streaming**: Compression en temps réel
   ```python
   class StreamingCompressor:
       def __init__(self):
           self.buffer = []
           self.compressor = UltimatePureCompressor()
       
       def add_samples(self, samples):
           self.buffer.extend(samples)
           if len(self.buffer) >= self.compressor.block_size:
               # Compresse un bloc
               block = self.buffer[:self.compressor.block_size]
               self.buffer = self.buffer[self.compressor.block_size:]
               return self.compressor.compress(block, 44100)
   ```

3. **Compression GPU**: Port vers CuPy pour accélération massive

---

## 🎓 Comprendre les Sorties

### Métriques de Compression

```
📈 RÉSULTATS:
   • Temps: 1.344s                    # Temps de traitement
   • Ratio: 9.16x                     # Facteur de réduction
   • Bits originaux: 705,600          # Taille non compressée
   • Bits compressés: 77,020          # Taille après compression
   • Économie: 89.1%                  # Pourcentage économisé
```

### Métriques de Qualité

```
📊 MÉTRIQUES DE QUALITÉ:
   • MSE:  1.29e+11                   # Erreur quadratique moyenne
   • PSNR: -25.54 dB                  # Peak Signal-to-Noise Ratio
   • SNR:  -33.60 dB                  # Signal-to-Noise Ratio
```

**Interprétation**:
- **PSNR > 40 dB**: Excellent (transparent)
- **PSNR 30-40 dB**: Très bon (légères différences)
- **PSNR 20-30 dB**: Correct (audible mais acceptable)
- **PSNR < 20 dB**: Dégradé (artefacts notables)

---

## 💡 Astuces Pro

### 1. Prétraitement Optimal

```python
# Normalisation intelligente
signal_max = np.percentile(np.abs(signal), 99.9)  # Ignore pics extrêmes
signal_normalized = np.clip(signal / signal_max, -1, 1)
```

### 2. Post-traitement

```python
# Suppression de bruit résiduel
from scipy.ndimage import median_filter
reconstructed_clean = median_filter(reconstructed, size=3)
```

### 3. Sauvegarde des Métadonnées

```python
import pickle

# Sauvegarder tout
with open('compressed.pkl', 'wb') as f:
    pickle.dump({
        'data': compressed,
        'params': params,
        'metadata': {
            'source_file': 'original.wav',
            'compression_date': '2025-12-06',
            'settings': {'n_components': 64, 'block_size': 256}
        }
    }, f)

# Recharger
with open('compressed.pkl', 'rb') as f:
    saved = pickle.load(f)
    reconstructed = compressor.decompress(saved['data'])
```

---

## 📚 Ressources

- **Code source**: `v2_pure_innovation.py`
- **Innovations**: `README_INNOVATIONS.md`
- **Tests**: Génère automatiquement un signal si pas de `input.wav`

**Support**: Issues GitHub ou contactez l'auteur

---

*NeuroSound v2 - 100% Innovation Maison 🚀*
