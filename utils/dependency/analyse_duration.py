import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Exemple de timestamps de messages (en secondes depuis le début)
timestamps = np.array([
    0, 5, 12, 18, 25, 30,           # Conversation 1
    350, 355, 360, 368, 375,        # Conversation 2 (après 5.8 min)
    700, 705, 710, 720, 725, 730,   # Conversation 3 (après 5.4 min)
    1500, 1505, 1510,               # Conversation 4 (après 12.8 min - sera coupé!)
    1900, 1905, 1910                # Conversation 5 (après 6.5 min)
])

def segment_with_hard_cut_and_dbscan(timestamps, hard_threshold_seconds=420, dbscan_method='iqr'):
    """
    Segmente les conversations en deux étapes:
    1. Coupe automatiquement après hard_threshold_seconds d'inactivité
    2. Applique DBSCAN avec IQR sur chaque segment
    
    Parameters:
    - timestamps: array de timestamps en secondes
    - hard_threshold_seconds: seuil dur en secondes (défaut: 420s = 7 min)
    - dbscan_method: 'iqr', 'median', 'percentile', 'gap'
    
    Returns:
    - final_clusters: labels des clusters finaux
    - hard_cuts: indices où les coupures dures ont été faites
    - eps_used: epsilon utilisé par DBSCAN
    """
    timestamps = np.array(timestamps)
    intervals = np.diff(timestamps)
    
    # Étape 1: Identifier les coupures dures (> 7 minutes)
    hard_cuts = np.where(intervals >= hard_threshold_seconds)[0] + 1
    hard_cuts = np.concatenate([[0], hard_cuts, [len(timestamps)]])
    
    print(f"=== Étape 1: Coupures dures après {hard_threshold_seconds}s ({hard_threshold_seconds/60:.1f} min) ===")
    print(f"Nombre de segments créés: {len(hard_cuts) - 1}")
    print(f"Indices de coupure: {hard_cuts[1:-1]}\n")
    
    # Étape 2: Appliquer DBSCAN sur chaque segment
    print(f"=== Étape 2: DBSCAN avec méthode '{dbscan_method}' sur chaque segment ===\n")
    
    final_clusters = np.full(len(timestamps), -1, dtype=int)
    cluster_counter = 0
    all_eps = []
    
    for i in range(len(hard_cuts) - 1):
        start_idx = hard_cuts[i]
        end_idx = hard_cuts[i + 1]
        segment_timestamps = timestamps[start_idx:end_idx]
        
        print(f"Segment {i+1}: indices {start_idx} à {end_idx-1} ({len(segment_timestamps)} messages)")
        print(f"  Période: {segment_timestamps[0]:.0f}s à {segment_timestamps[-1]:.0f}s")
        
        # Si le segment a moins de 2 messages, marquer comme bruit
        if len(segment_timestamps) < 2:
            print(f"  → Trop petit, marqué comme bruit\n")
            continue
        
        # Calculer les intervalles du segment
        segment_intervals = np.diff(segment_timestamps)
        
        if len(segment_intervals) == 0:
            print(f"  → Pas d'intervalles, marqué comme bruit\n")
            continue
        
        # Calculer eps selon la méthode choisie
        if dbscan_method == 'iqr':
            if len(segment_intervals) >= 4:  # IQR nécessite au moins 4 valeurs
                q1 = np.percentile(segment_intervals, 25)
                q3 = np.percentile(segment_intervals, 75)
                iqr = q3 - q1
                eps = q3 + 1.5 * iqr
            else:
                eps = np.median(segment_intervals) * 2  # Fallback
        elif dbscan_method == 'median':
            median = np.median(segment_intervals)
            mad = np.median(np.abs(segment_intervals - median))
            eps = median + 3 * mad
        elif dbscan_method == 'percentile':
            eps = np.percentile(segment_intervals, 75)
        elif dbscan_method == 'gap':
            sorted_intervals = np.sort(segment_intervals)
            gaps = np.diff(sorted_intervals)
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                eps = sorted_intervals[max_gap_idx]
            else:
                eps = np.median(segment_intervals)
        else:
            raise ValueError("Méthode inconnue")
        
        all_eps.append(eps)
        print(f"  Intervalles: min={segment_intervals.min():.1f}s, max={segment_intervals.max():.1f}s, médiane={np.median(segment_intervals):.1f}s")
        print(f"  eps calculé: {eps:.1f}s")
        
        # Appliquer DBSCAN
        X = segment_timestamps.reshape(-1, 1)
        dbscan = DBSCAN(eps=eps, min_samples=2)
        segment_clusters = dbscan.fit_predict(X)
        
        # Renommer les clusters pour éviter les conflits
        for j in range(len(segment_clusters)):
            if segment_clusters[j] != -1:
                segment_clusters[j] += cluster_counter
        
        # Mettre à jour le compteur de clusters
        n_clusters_in_segment = len(set(segment_clusters[segment_clusters != -1]))
        cluster_counter += n_clusters_in_segment
        
        print(f"  → {n_clusters_in_segment} sous-conversations détectées\n")
        
        # Stocker les résultats
        final_clusters[start_idx:end_idx] = segment_clusters
    
    n_total_conversations = len(set(final_clusters[final_clusters != -1]))
    print(f"=== Résultat final: {n_total_conversations} conversations au total ===\n")
    
    return final_clusters, hard_cuts, all_eps

# Application de la segmentation hybride
print("=" * 70)
print("SEGMENTATION HYBRIDE: Coupure dure (7 min) + DBSCAN (IQR)")
print("=" * 70 + "\n")

clusters, hard_cuts, eps_list = segment_with_hard_cut_and_dbscan(
    timestamps, 
    hard_threshold_seconds=420,  # 7 minutes
    dbscan_method='iqr'
)

# Affichage détaillé des conversations
print("\n=== Détail des conversations finales ===")
for cluster_id in sorted(set(clusters)):
    if cluster_id == -1:
        noise_timestamps = timestamps[clusters == -1]
        if len(noise_timestamps) > 0:
            print(f"\nMessages isolés (bruit): {len(noise_timestamps)} messages")
            print(f"  Timestamps: {noise_timestamps}")
    else:
        conv_timestamps = timestamps[clusters == cluster_id]
        print(f"\nConversation {cluster_id + 1}:")
        print(f"  Nombre de messages: {len(conv_timestamps)}")
        print(f"  Période: {conv_timestamps[0]:.0f}s à {conv_timestamps[-1]:.0f}s")
        print(f"  Durée totale: {conv_timestamps[-1] - conv_timestamps[0]:.0f}s ({(conv_timestamps[-1] - conv_timestamps[0])/60:.1f} min)")
        print(f"  Timestamps: {conv_timestamps}")

# Visualisation
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']

# Subplot 1: Timeline avec segmentation finale
ax = axes[0]
for cluster_id in set(clusters):
    mask = clusters == cluster_id
    if cluster_id == -1:
        ax.scatter(timestamps[mask], [0]*sum(mask), c='gray', marker='x', 
                  s=150, label='Bruit', zorder=5)
    else:
        ax.scatter(timestamps[mask], [0]*sum(mask), 
                  c=colors[cluster_id % len(colors)], 
                  s=150, label=f'Conv {cluster_id + 1}', zorder=5)

# Marquer les coupures dures
for cut in hard_cuts[1:-1]:
    cut_time = timestamps[cut]
    ax.axvline(cut_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(cut_time, 0.03, '7 min', ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

ax.set_ylabel('Messages', fontsize=11)
ax.set_title('Segmentation Hybride: Coupure dure (7 min) + DBSCAN (IQR)', 
            fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 0.05)

# Subplot 2: Intervalles entre messages
ax = axes[1]
intervals = np.diff(timestamps)
x_positions = timestamps[:-1] + intervals / 2  # Position au milieu de chaque intervalle
bars = ax.bar(x_positions, intervals, width=np.minimum(intervals, 20), 
              color='steelblue', alpha=0.7, edgecolor='black')

# Colorer différemment les intervalles au-dessus du seuil
for i, interval in enumerate(intervals):
    if interval >= 420:
        bars[i].set_color('red')
        bars[i].set_alpha(0.9)

ax.axhline(y=420, color='red', linestyle='--', linewidth=2, label='Seuil dur (7 min)')
ax.set_ylabel('Intervalle (s)', fontsize=11)
ax.set_xlabel('Position temporelle', fontsize=11)
ax.set_title('Intervalles entre messages consécutifs', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Subplot 3: Distribution des intervalles
ax = axes[2]
# Séparer les intervalles normaux des pauses longues
normal_intervals = intervals[intervals < 420]
long_intervals = intervals[intervals >= 420]

if len(normal_intervals) > 0:
    ax.hist(normal_intervals, bins=20, color='steelblue', alpha=0.7, 
           edgecolor='black', label=f'< 7 min ({len(normal_intervals)} intervalles)')
if len(long_intervals) > 0:
    ax.hist(long_intervals, bins=10, color='red', alpha=0.7, 
           edgecolor='black', label=f'≥ 7 min ({len(long_intervals)} intervalles)')

# Marquer les eps utilisés par DBSCAN
for i, eps in enumerate(eps_list):
    ax.axvline(eps, color=colors[i % len(colors)], linestyle=':', 
              linewidth=2, alpha=0.8, label=f'eps segment {i+1}: {eps:.1f}s')

ax.set_xlabel('Intervalle (s)', fontsize=11)
ax.set_ylabel('Fréquence', fontsize=11)
ax.set_title('Distribution des intervalles avec seuils', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_segmentation.png', dpi=150, bbox_inches='tight')
print("\n" + "="*70)
print("Graphique sauvegardé: hybrid_segmentation.png")
print("="*70)