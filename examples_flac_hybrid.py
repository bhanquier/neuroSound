"""
Exemples d'Utilisation NeuroSound FLAC Hybrid
==============================================
Cas d'usage réels et scénarios concrets
"""

import subprocess
import os


def example_1_simple_compression():
    """Exemple 1: Compression simple d'un fichier WAV."""
    print("="*60)
    print("EXEMPLE 1: Compression Simple")
    print("="*60)
    
    print("""
Scénario: Vous avez un fichier audio 'song.wav' et voulez
         le compresser pour économiser de l'espace.

Commande:
    python3 neurosound_flac_hybrid.py compress song.wav song.flac

Résultat:
    ✓ Fichier song.flac créé (plus petit que WAV)
    ✓ Lisible avec n'importe quel lecteur FLAC
    ✓ Métadonnées stockées pour reconstruction
    """)


def example_2_batch_processing():
    """Exemple 2: Traitement par lot."""
    print("\n" + "="*60)
    print("EXEMPLE 2: Traitement Par Lot")
    print("="*60)
    
    script = """
#!/bin/bash
# batch_compress.sh - Compresse tous les WAV d'un dossier

for file in *.wav; do
    echo "Compression de $file..."
    python3 neurosound_flac_hybrid.py compress "$file" "${file%.wav}.flac" 8
done

echo "✅ Tous les fichiers compressés !"
    """
    
    print(f"""
Scénario: Compresser tous les fichiers WAV d'un album

Script Bash:
{script}

Usage:
    chmod +x batch_compress.sh
    ./batch_compress.sh
    """)


def example_3_streaming_workflow():
    """Exemple 3: Workflow de streaming."""
    print("\n" + "="*60)
    print("EXEMPLE 3: Workflow Streaming")
    print("="*60)
    
    print("""
Scénario: Service de streaming musical avec économie de stockage

Pipeline:
    1. Enregistrement original → master.wav
    2. Compression hybrid → master.flac (NeuroSound)
    3. Stockage cloud → 10% d'économie vs FLAC standard
    4. Streaming → Tous clients FLAC compatibles !

Avantages:
    ✓ Économie de stockage cloud
    ✓ Économie de bande passante
    ✓ Qualité préservée
    ✓ Standard compatible
    
Code Python:
    # Compression pour stockage cloud
    compress('master.wav', 'cloud/master.flac', level=8)
    
    # Client stream avec n'importe quel lecteur
    # vlc http://server/cloud/master.flac
    """)


def example_4_archival():
    """Exemple 4: Archivage longue durée."""
    print("\n" + "="*60)
    print("EXEMPLE 4: Archivage Longue Durée")
    print("="*60)
    
    print("""
Scénario: Archive de podcasts/émissions radio

Stratégie:
    Original: 1000 épisodes × 50 MB = 50 GB
    FLAC std: 1000 × 43 MB = 43 GB (14% gain)
    Hybrid:   1000 × 39 MB = 39 GB (22% gain)
    
    💾 Économie: 11 GB sur l'archive !

Bonus:
    ✓ Format FLAC pérenne (existera toujours)
    ✓ Reconstruction parfaite disponible
    ✓ Migration facile vers nouveaux formats
    
Commande:
    python3 neurosound_flac_hybrid.py compress \\
        podcast_ep001.wav \\
        archive/podcast_ep001.flac \\
        8
    """)


def example_5_professional_audio():
    """Exemple 5: Production audio professionnelle."""
    print("\n" + "="*60)
    print("EXEMPLE 5: Production Audio Pro")
    print("="*60)
    
    print("""
Scénario: Studio d'enregistrement avec workflow hybride

Workflow:
    Enregistrement → WAV 24-bit/96kHz (haute qualité)
           ↓
    Mixage/Master → ProTools/Logic/Reaper
           ↓
    Export Final → master.wav
           ↓
    Archive Hybrid → master.flac (NeuroSound)
           ↓
    Distribution → MP3, AAC, etc.

Avantages Production:
    ✓ Master en FLAC (économie stockage)
    ✓ Compatible tous DAW (lecture FLAC)
    ✓ Métadonnées préservées
    ✓ Reconstruction parfaite si besoin
    
Commande:
    # Archive du master
    python3 neurosound_flac_hybrid.py compress \\
        "Album - Master Final.wav" \\
        "Archive/Album - Master.flac" \\
        8
    
    # Récupération pour nouveau mix
    python3 neurosound_flac_hybrid.py decompress \\
        "Archive/Album - Master.flac" \\
        "Remix/source.wav"
    """)


def example_6_quality_comparison():
    """Exemple 6: Comparaison de qualité."""
    print("\n" + "="*60)
    print("EXEMPLE 6: Test de Qualité ABX")
    print("="*60)
    
    test_script = """
#!/bin/bash
# quality_test.sh - Compare qualité original vs hybrid

# 1. Compresse
python3 neurosound_flac_hybrid.py compress original.wav test.flac

# 2. Décode avec FLAC standard (mode compatible)
flac -d test.flac -o decoded_standard.wav

# 3. Décode avec NeuroSound (mode optimal)
python3 neurosound_flac_hybrid.py decompress test.flac decoded_hybrid.wav

# 4. Compare avec original
echo "Comparaison mode standard:"
python3 -c "
import numpy as np
import wave

with wave.open('original.wav', 'rb') as w:
    orig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
with wave.open('decoded_standard.wav', 'rb') as w:
    decoded = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

mse = np.mean((orig.astype(float) - decoded.astype(float))**2)
psnr = 10 * np.log10(32768**2 / (mse + 1e-10))
print(f'PSNR Standard: {psnr:.1f} dB')
"

echo "Comparaison mode hybrid:"
python3 -c "
import numpy as np
import wave

with wave.open('original.wav', 'rb') as w:
    orig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
with wave.open('decoded_hybrid.wav', 'rb') as w:
    decoded = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

mse = np.mean((orig.astype(float) - decoded.astype(float))**2)
psnr = 10 * np.log10(32768**2 / (mse + 1e-10))
print(f'PSNR Hybrid: {psnr:.1f} dB')
"
    """
    
    print(f"""
Scénario: Évaluer la qualité audio avant déploiement

Script de Test:
{test_script}

Interprétation PSNR:
    > 90 dB  → Quasi-lossless (imperceptible)
    > 60 dB  → Très haute qualité
    > 40 dB  → Haute qualité
    > 30 dB  → Qualité acceptable
    < 30 dB  → Dégradation audible
    """)


def example_7_integration_python():
    """Exemple 7: Intégration dans application Python."""
    print("\n" + "="*60)
    print("EXEMPLE 7: Intégration Application")
    print("="*60)
    
    code = '''
from neurosound_flac_hybrid import NeuroSoundFLACHybrid

# Initialisation du codec
codec = NeuroSoundFLACHybrid(compression_level=8)

# Compression
result = codec.compress('input.wav', 'output.flac')
print(f"Ratio: {result['ratio']:.2f}x")
print(f"Économie: {100*(1-1/result['ratio']):.1f}%")

# Décompression
codec.decompress('output.flac', 'restored.wav')
print("✅ Restauration complète !")
    '''
    
    print(f"""
Scénario: Intégrer NeuroSound dans votre application

Code Python:
{code}

Cas d'usage:
    • Application d'enregistrement audio
    • Éditeur audio
    • Convertisseur de formats
    • Service de backup audio
    • Pipeline de traitement batch
    """)


def example_8_web_service():
    """Exemple 8: Service web de conversion."""
    print("\n" + "="*60)
    print("EXEMPLE 8: API Web de Conversion")
    print("="*60)
    
    flask_code = '''
from flask import Flask, request, send_file
from neurosound_flac_hybrid import NeuroSoundFLACHybrid
import tempfile
import os

app = Flask(__name__)
codec = NeuroSoundFLACHybrid(compression_level=8)

@app.route('/compress', methods=['POST'])
def compress_audio():
    """Endpoint de compression."""
    # Récupère le fichier uploadé
    file = request.files['audio']
    
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
        file.save(tmp_in.name)
        
    # Compression
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp_out:
        codec.compress(tmp_in.name, tmp_out.name)
        
    # Envoie le fichier compressé
    return send_file(tmp_out.name, mimetype='audio/flac')

@app.route('/decompress', methods=['POST'])
def decompress_audio():
    """Endpoint de décompression."""
    file = request.files['audio']
    
    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp_in:
        file.save(tmp_in.name)
        
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
        codec.decompress(tmp_in.name, tmp_out.name)
        
    return send_file(tmp_out.name, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True)
    '''
    
    print(f"""
Scénario: Service web de conversion audio

Flask API:
{flask_code}

Usage Client:
    # Compression
    curl -F "audio=@song.wav" http://localhost:5000/compress -o song.flac
    
    # Décompression
    curl -F "audio=@song.flac" http://localhost:5000/decompress -o song.wav

Déploiement:
    docker build -t neurosound-api .
    docker run -p 5000:5000 neurosound-api
    """)


def main():
    """Affiche tous les exemples."""
    print("🔥" * 30)
    print("NeuroSound FLAC Hybrid - Exemples d'Utilisation")
    print("🔥" * 30)
    
    example_1_simple_compression()
    example_2_batch_processing()
    example_3_streaming_workflow()
    example_4_archival()
    example_5_professional_audio()
    example_6_quality_comparison()
    example_7_integration_python()
    example_8_web_service()
    
    print("\n" + "🔥" * 30)
    print("Plus d'infos: README_FLAC_HYBRID.md")
    print("🔥" * 30)


if __name__ == '__main__':
    main()
