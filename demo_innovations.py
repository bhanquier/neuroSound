"""
NeuroSound - Démonstration Interactive des Innovations
======================================================
Visualise et compare chaque innovation individuellement
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans GUI
import matplotlib.pyplot as plt
from v2_pure_innovation import (
    IncrementalKLTransform,
    LogarithmicHypercubicQuantizer,
    ContextualMultiOrderEncoder,
    AdaptivePolynomialPredictor,
    KolmogorovComplexitySegmenter
)


def demo_klt_learning():
    """Démontre l'apprentissage adaptatif de la KLT."""
    print("\n" + "="*80)
    print("🧬 DÉMO 1: Transformée de Karhunen-Loève Adaptative")
    print("="*80)
    
    # Génère un signal avec structure
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*13*t) + 0.2*np.random.randn(len(t))
    
    # Découpe en blocs
    block_size = 64
    blocks = signal[:len(signal)//block_size*block_size].reshape(-1, block_size)
    
    # KLT adaptative
    klt = IncrementalKLTransform(n_components=8, input_dim=block_size, learning_rate=0.05)
    
    # Apprentissage progressif
    reconstruction_errors = []
    for i, block in enumerate(blocks):
        # Transforme
        coeffs = klt.transform(block)
        reconstructed = klt.inverse_transform(coeffs)
        
        # Erreur
        error = np.mean((block - reconstructed)**2)
        reconstruction_errors.append(error)
        
        # Apprend
        if i % 2 == 0:  # Update tous les 2 blocs
            klt.update(block)
        
        if i % 5 == 0:
            print(f"  Bloc {i:3d}: MSE = {error:.4f}")
    
    print(f"\n  ✅ Erreur initiale: {reconstruction_errors[0]:.4f}")
    print(f"  ✅ Erreur finale:   {reconstruction_errors[-1]:.4f}")
    print(f"  ✅ Amélioration:    {(1-reconstruction_errors[-1]/reconstruction_errors[0])*100:.1f}%")
    
    # Visualisation
    plt.figure(figsize=(12, 4))
    plt.plot(reconstruction_errors, linewidth=2)
    plt.xlabel('Numéro de bloc', fontsize=12)
    plt.ylabel('MSE de reconstruction', fontsize=12)
    plt.title('Apprentissage Adaptatif de la KLT', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('demo_klt_learning.png', dpi=150, bbox_inches='tight')
    print(f"  📊 Graphique sauvegardé: demo_klt_learning.png")


def demo_logarithmic_quantization():
    """Compare quantification uniforme vs logarithmique."""
    print("\n" + "="*80)
    print("📐 DÉMO 2: Quantification Logarithmique vs Uniforme")
    print("="*80)
    
    # Génère des coefficients avec distribution de loi de puissance
    np.random.seed(42)
    coeffs = np.random.randn(10000)
    coeffs = np.sign(coeffs) * np.abs(coeffs) ** 1.5  # Loi de puissance
    
    # Quantification uniforme (naïve)
    n_levels = 256
    min_val, max_val = coeffs.min(), coeffs.max()
    uniform_levels = np.linspace(min_val, max_val, n_levels)
    uniform_indices = np.searchsorted(uniform_levels, coeffs)
    uniform_quantized = uniform_levels[np.clip(uniform_indices, 0, n_levels-1)]
    
    # Quantification logarithmique
    log_quantizer = LogarithmicHypercubicQuantizer(n_bits=8)
    log_indices = log_quantizer.quantize(coeffs)
    log_quantized = log_quantizer.dequantize(log_indices)
    
    # Erreurs
    uniform_error = np.mean((coeffs - uniform_quantized)**2)
    log_error = np.mean((coeffs - log_quantized)**2)
    
    print(f"\n  Quantification Uniforme:")
    print(f"    • MSE: {uniform_error:.6f}")
    print(f"    • MAE: {np.mean(np.abs(coeffs - uniform_quantized)):.6f}")
    
    print(f"\n  Quantification Logarithmique:")
    print(f"    • MSE: {log_error:.6f}")
    print(f"    • MAE: {np.mean(np.abs(coeffs - log_quantized)):.6f}")
    
    print(f"\n  ✅ Amélioration MSE: {(1 - log_error/uniform_error)*100:.1f}%")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Distribution originale
    axes[0].hist(coeffs, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Valeur')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution Originale\n(Loi de puissance)', fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Erreur uniforme
    axes[1].scatter(coeffs[:1000], uniform_quantized[:1000] - coeffs[:1000], 
                   alpha=0.3, s=1, c='red')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Valeur originale')
    axes[1].set_ylabel('Erreur de quantification')
    axes[1].set_title(f'Quantification Uniforme\nMSE={uniform_error:.6f}', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Erreur logarithmique
    axes[2].scatter(coeffs[:1000], log_quantized[:1000] - coeffs[:1000], 
                   alpha=0.3, s=1, c='green')
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Valeur originale')
    axes[2].set_ylabel('Erreur de quantification')
    axes[2].set_title(f'Quantification Logarithmique\nMSE={log_error:.6f}', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_quantization.png', dpi=150, bbox_inches='tight')
    print(f"  📊 Graphique sauvegardé: demo_quantization.png")


def demo_contextual_encoding():
    """Démontre le codage contextuel vs ordre 0."""
    print("\n" + "="*80)
    print("🎯 DÉMO 3: Codage Contextuel Multi-Ordre")
    print("="*80)
    
    # Génère une séquence avec structure
    np.random.seed(42)
    sequence = []
    state = 128
    for _ in range(5000):
        # Marche aléatoire avec momentum
        state = int(np.clip(state + np.random.randint(-10, 11), 0, 255))
        sequence.append(state)
    
    sequence = np.array(sequence)
    
    # Codeur ordre 0 (Huffman naïf)
    from collections import Counter
    counts = Counter(sequence)
    total = len(sequence)
    entropy_order0 = -sum((count/total) * np.log2(count/total) for count in counts.values())
    
    # Codeur contextuel multi-ordre
    encoder = ContextualMultiOrderEncoder(max_order=4, vocab_size=256)
    encoder.update(sequence)
    estimated_bits = encoder.encode_length_estimate(sequence)
    entropy_contextual = estimated_bits / len(sequence)
    
    print(f"\n  Ordre 0 (sans contexte):")
    print(f"    • Entropie: {entropy_order0:.3f} bits/symbole")
    print(f"    • Total:    {entropy_order0 * len(sequence):.0f} bits")
    
    print(f"\n  Multi-ordre (avec contexte):")
    print(f"    • Entropie: {entropy_contextual:.3f} bits/symbole")
    print(f"    • Total:    {estimated_bits:.0f} bits")
    
    print(f"\n  ✅ Économie: {(1 - entropy_contextual/entropy_order0)*100:.1f}%")
    
    # Test prédiction
    print(f"\n  Exemples de prédiction:")
    for i in range(100, 105):
        context = sequence[i-4:i].tolist()
        true_symbol = sequence[i]
        prob = encoder.get_probability(true_symbol, context)
        print(f"    Contexte {context} → symbole {true_symbol}: P = {prob:.4f}")


def demo_polynomial_prediction():
    """Compare prédiction linéaire vs polynomiale."""
    print("\n" + "="*80)
    print("📈 DÉMO 4: Prédiction Polynomiale Adaptative")
    print("="*80)
    
    # Signal avec tendance non-linéaire
    t = np.linspace(0, 10, 1000)
    signal = 100*np.sin(2*np.pi*0.5*t) + 20*t + 0.1*t**2 + 5*np.random.randn(len(t))
    
    # Prédicteur polynomial adaptatif
    predictor = AdaptivePolynomialPredictor(max_order=16, poly_degree=2)
    predictions, residuals = predictor.predict_sequence(signal)
    
    # Prédiction linéaire simple (pour comparaison)
    linear_predictions = np.zeros(len(signal))
    for i in range(1, len(signal)):
        linear_predictions[i] = signal[i-1]
    linear_residuals = signal - linear_predictions
    
    # Métriques
    poly_mse = np.mean(residuals**2)
    linear_mse = np.mean(linear_residuals**2)
    
    print(f"\n  Prédiction Linéaire Simple:")
    print(f"    • MSE résidu: {linear_mse:.2f}")
    print(f"    • Std résidu: {np.std(linear_residuals):.2f}")
    
    print(f"\n  Prédiction Polynomiale Adaptative:")
    print(f"    • MSE résidu: {poly_mse:.2f}")
    print(f"    • Std résidu: {np.std(residuals):.2f}")
    
    print(f"\n  ✅ Réduction résidu: {(1 - poly_mse/linear_mse)*100:.1f}%")
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Signal original
    axes[0, 0].plot(t[:200], signal[:200], 'b-', linewidth=2, label='Signal')
    axes[0, 0].plot(t[:200], predictions[:200], 'r--', linewidth=1.5, label='Prédiction poly')
    axes[0, 0].plot(t[:200], linear_predictions[:200], 'g:', linewidth=1.5, label='Prédiction lin')
    axes[0, 0].set_xlabel('Temps')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Signal et Prédictions', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Résidus comparés
    axes[0, 1].plot(t[:200], linear_residuals[:200], 'g-', alpha=0.7, label='Résidu linéaire')
    axes[0, 1].plot(t[:200], residuals[:200], 'r-', alpha=0.7, label='Résidu polynomial')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Temps')
    axes[0, 1].set_ylabel('Résidu')
    axes[0, 1].set_title('Comparaison des Résidus', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution résidus linéaires
    axes[1, 0].hist(linear_residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Résidu')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title(f'Distribution Résidu Linéaire\nMSE={linear_mse:.2f}', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Distribution résidus polynomiaux
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Résidu')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].set_title(f'Distribution Résidu Polynomial\nMSE={poly_mse:.2f}', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('demo_prediction.png', dpi=150, bbox_inches='tight')
    print(f"  📊 Graphique sauvegardé: demo_prediction.png")


def demo_kolmogorov_segmentation():
    """Visualise la segmentation par complexité."""
    print("\n" + "="*80)
    print("🔬 DÉMO 5: Segmentation par Complexité de Kolmogorov")
    print("="*80)
    
    # Signal composite: silence + musique + parole + bruit
    sample_rate = 8000
    
    # Silence
    silence = np.random.randn(2000) * 10
    
    # Musique (structure harmonique)
    t1 = np.linspace(0, 2, 4000)
    music = 1000 * np.sin(2*np.pi*440*t1) + 500*np.sin(2*np.pi*880*t1)
    
    # Parole simulée (modulation AM)
    t2 = np.linspace(0, 2, 3000)
    speech = 800 * np.sin(2*np.pi*200*t2) * (1 + 0.5*np.sin(2*np.pi*10*t2))
    
    # Bruit
    noise = np.random.randn(3000) * 500
    
    # Concaténation
    signal = np.concatenate([silence, music, speech, noise])
    
    # Segmentation
    segmenter = KolmogorovComplexitySegmenter(window_size=400, min_segment=500, max_segment=5000)
    
    # Profil de complexité
    complexity = segmenter.compute_complexity_profile(signal)
    
    # Segments
    segments = segmenter.segment(signal)
    
    print(f"\n  Signal total: {len(signal)} échantillons")
    print(f"  Nombre de segments: {len(segments)}")
    print(f"\n  Détails des segments:")
    for i, (start, end) in enumerate(segments):
        print(f"    Segment {i+1}: [{start:5d}, {end:5d}] - {end-start:4d} échantillons")
    
    # Visualisation
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Signal
    t = np.arange(len(signal)) / sample_rate
    axes[0].plot(t, signal, linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Signal Composite', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Annotations des régions
    axes[0].axvspan(0, len(silence)/sample_rate, alpha=0.2, color='blue', label='Silence')
    axes[0].axvspan(len(silence)/sample_rate, 
                   (len(silence)+len(music))/sample_rate, alpha=0.2, color='green', label='Musique')
    axes[0].axvspan((len(silence)+len(music))/sample_rate,
                   (len(silence)+len(music)+len(speech))/sample_rate, alpha=0.2, color='yellow', label='Parole')
    axes[0].axvspan((len(silence)+len(music)+len(speech))/sample_rate,
                   len(signal)/sample_rate, alpha=0.2, color='red', label='Bruit')
    axes[0].legend(loc='upper right')
    
    # Complexité
    t_complexity = np.linspace(0, len(signal)/sample_rate, len(complexity))
    axes[1].plot(t_complexity, complexity, linewidth=2, color='purple')
    axes[1].set_ylabel('Complexité K(x)')
    axes[1].set_title('Profil de Complexité de Kolmogorov', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Segments détectés
    axes[2].plot(t, signal, linewidth=0.5, alpha=0.5)
    for i, (start, end) in enumerate(segments):
        color = plt.cm.tab10(i % 10)
        axes[2].axvspan(start/sample_rate, end/sample_rate, alpha=0.3, color=color)
        mid = (start + end) / 2 / sample_rate
        axes[2].text(mid, 0, f'S{i+1}', ha='center', va='center', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_xlabel('Temps (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Segmentation Adaptative', fontweight='bold', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_segmentation.png', dpi=150, bbox_inches='tight')
    print(f"  📊 Graphique sauvegardé: demo_segmentation.png")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎨 DÉMONSTRATIONS INTERACTIVES - INNOVATIONS MATHÉMATIQUES")
    print("="*80)
    print("Chaque démo génère des graphiques pour visualiser les concepts")
    print("="*80)
    
    # Exécute toutes les démos
    demo_klt_learning()
    demo_logarithmic_quantization()
    demo_contextual_encoding()
    demo_polynomial_prediction()
    demo_kolmogorov_segmentation()
    
    print("\n" + "="*80)
    print("✨ TOUTES LES DÉMOS TERMINÉES ✨")
    print("="*80)
    print("\nFichiers générés:")
    print("  • demo_klt_learning.png - Apprentissage adaptatif KLT")
    print("  • demo_quantization.png - Quantification logarithmique")
    print("  • demo_prediction.png - Prédiction polynomiale")
    print("  • demo_segmentation.png - Segmentation Kolmogorov")
    print("\n" + "="*80 + "\n")
