# %% [markdown]
# # 🧬 RNA 3D Folding - Pipeline v5 (Ultra-Optimisé)
#
# Optimisations avancées pour dépasser 400:
# - Validation data comme templates supplémentaires
# - Ensemble de templates avec superposition Kabsch
# - Moyenne pondérée par similarité
# - Prédiction de structure secondaire (base-pairing)
# - Raffinement itératif des contraintes
# - Stratégies de prédiction diversifiées

# %% [markdown]
# ## 📦 1. Installation BioPython

# %%
import os
import sys
import subprocess
import warnings

print("📦 Installation BioPython...")

whl_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        if file.endswith('.whl') and 'bio' in file.lower():
            whl_path = os.path.join(root, file)
            break
    if whl_path:
        break

if whl_path:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path, "--no-deps", "--no-index", "-q"])
        print(f"✅ BioPython installé: {whl_path}")
    except:
        whl_path = None

BIOPYTHON_OK = False
try:
    from Bio import pairwise2
    from Bio.Seq import Seq
    BIOPYTHON_OK = True
    print("✅ BioPython chargé!")
except:
    print("⚠️ BioPython non disponible")

# %% [markdown]
# ## 📦 2. Imports

# %%
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd
import random
import time
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings('ignore')

DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
print("✅ Imports OK")

# %% [markdown]
# ## 📊 3. Chargement des données

# %%
print("📂 Chargement...")
train_seqs = pd.read_csv(DATA_PATH / 'train_sequences.csv')
valid_seqs = pd.read_csv(DATA_PATH / 'validation_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH / 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')
valid_labels = pd.read_csv(DATA_PATH / 'validation_labels.csv')

print(f"✅ Train: {len(train_seqs)}, Valid: {len(valid_seqs)}, Test: {len(test_seqs)}")

# %% [markdown]
# ## 🔧 4. Traitement des labels

# %%
def process_labels(labels_df):
    """Parse labels avec split('_')[0]."""
    coords_dict = {}
    for id_prefix, group in labels_df.groupby(lambda x: labels_df['ID'][x].split('_')[0]):
        coords = []
        for _, row in group.sort_values('resid').iterrows():
            coords.append([row['x_1'], row['y_1'], row['z_1']])
        coords_dict[id_prefix] = np.array(coords)
    return coords_dict

print("📊 Traitement des labels...")
train_coords = process_labels(train_labels)
valid_coords = process_labels(valid_labels)

# FUSION: Train + Validation comme templates
all_seqs = pd.concat([train_seqs, valid_seqs], ignore_index=True)
all_coords = {**train_coords, **valid_coords}

print(f"✅ {len(all_coords)} structures totales (train + valid)")

# %% [markdown]
# ## 🔬 5. Superposition Kabsch

# %%
def kabsch_superpose(mobile, target):
    """
    Superpose 'mobile' sur 'target' avec l'algorithme de Kabsch.
    Retourne les coordonnées alignées.
    """
    if len(mobile) != len(target) or len(mobile) < 3:
        return mobile.copy()

    # Centrer
    mobile_center = np.mean(mobile, axis=0)
    target_center = np.mean(target, axis=0)

    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Matrice de covariance
    H = mobile_centered.T @ target_centered

    # SVD
    U, S, Vt = svd(H)

    # Rotation optimale
    R = Vt.T @ U.T

    # Correction si réflexion
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Appliquer transformation
    aligned = (mobile_centered @ R) + target_center

    return aligned

def rmsd(coords1, coords2):
    """Calcule le RMSD entre deux ensembles de coordonnées."""
    if len(coords1) != len(coords2):
        return float('inf')
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

print("✅ Superposition Kabsch définie")

# %% [markdown]
# ## 🧬 6. Prédiction structure secondaire

# %%
def predict_base_pairs(sequence, min_loop=3):
    """
    Prédit les paires de bases potentielles (A-U, G-C, G-U wobble).
    Retourne liste de (i, j, strength).
    """
    n = len(sequence)
    pairs = []

    # Paires canoniques et wobble
    pairing = {
        ('A', 'U'): 1.0, ('U', 'A'): 1.0,
        ('G', 'C'): 1.2, ('C', 'G'): 1.2,  # G-C plus fort
        ('G', 'U'): 0.6, ('U', 'G'): 0.6,  # Wobble
    }

    # Chercher paires possibles
    for i in range(n):
        for j in range(i + min_loop + 1, n):
            key = (sequence[i], sequence[j])
            if key in pairing:
                # Bonus si flanqué par d'autres paires (stacking)
                strength = pairing[key]

                # Check stacking
                if i > 0 and j < n - 1:
                    key_stack = (sequence[i-1], sequence[j+1])
                    if key_stack in pairing:
                        strength *= 1.3

                pairs.append((i, j, strength))

    # Trier par force
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Filtrer les conflits (greedy)
    used = set()
    filtered = []
    for i, j, s in pairs:
        if i not in used and j not in used:
            filtered.append((i, j, s))
            used.add(i)
            used.add(j)

    return filtered

print("✅ Prédiction base pairs définie")

# %% [markdown]
# ## 🔍 7. Alignement

# %%
def align_sequences(seq1, seq2):
    """Alignement BioPython ou fallback."""
    if BIOPYTHON_OK:
        alignments = pairwise2.align.globalms(
            Seq(seq1), seq2, 2, -1, -10, -0.5,
            one_alignment_only=True
        )
        if alignments:
            a = alignments[0]
            score = a.score / (2 * min(len(seq1), len(seq2)))
            return max(0, score), str(a.seqA), str(a.seqB)

    # Fallback NW simplifié
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0, seq1, seq2

    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i, 0] = i * -2
    for j in range(m + 1): dp[0, j] = j * -2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1, j-1] + (2 if seq1[i-1] == seq2[j-1] else -1)
            dp[i, j] = max(match, dp[i-1, j] - 2, dp[i, j-1] - 2)

    # Traceback simple
    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (2 if seq1[i-1] == seq2[j-1] else -1):
            aligned1.append(seq1[i-1]); aligned2.append(seq2[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] - 2:
            aligned1.append(seq1[i-1]); aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-'); aligned2.append(seq2[j-1])
            j -= 1

    score = dp[n, m] / (2 * min(n, m)) if min(n, m) > 0 else 0
    return max(0, score), ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

print(f"✅ Alignement: {'BioPython' if BIOPYTHON_OK else 'Fallback NW'}")

# %% [markdown]
# ## 🔍 8. Recherche de templates (OPTIMISÉE avec k-mer)

# %%
def kmer_similarity(seq1, seq2, k=4):
    """Similarité k-mer ultra-rapide O(n+m)."""
    if len(seq1) < k or len(seq2) < k:
        return sum(a == b for a, b in zip(seq1, seq2)) / max(len(seq1), len(seq2), 1)

    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))

    if not kmers1 or not kmers2:
        return 0.0

    return len(kmers1 & kmers2) / len(kmers1 | kmers2)

def find_templates(query_seq, seqs_df, coords_dict, temporal_cutoff=None, top_n=10):
    """
    Recherche RAPIDE avec pré-filtrage k-mer.
    1. Filtre longueur
    2. Pré-filtre k-mer (top 25 candidats)
    3. Alignement complet sur les meilleurs seulement
    """
    # Filtre temporal
    if temporal_cutoff is not None and 'temporal_cutoff' in seqs_df.columns:
        filtered = seqs_df[seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered = seqs_df

    # ÉTAPE 1: Pré-filtrage rapide par k-mer
    candidates = []
    for _, row in filtered.iterrows():
        tid = row['target_id']
        tseq = row['sequence']

        if tid not in coords_dict:
            continue

        # Filtre longueur (40%)
        len_ratio = min(len(query_seq), len(tseq)) / max(len(query_seq), len(tseq))
        if len_ratio < 0.4:
            continue

        # K-mer rapide
        kmer_sim = kmer_similarity(query_seq, tseq, k=4)
        candidates.append((tid, tseq, kmer_sim))

    # ÉTAPE 2: Garder top 25 par k-mer
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:25]

    # ÉTAPE 3: Alignement complet sur les meilleurs
    results = []
    for tid, tseq, _ in candidates:
        sim, aq, at = align_sequences(query_seq, tseq)
        results.append({
            'target_id': tid,
            'sequence': tseq,
            'similarity': sim,
            'coords': coords_dict[tid],
            'aligned_query': aq,
            'aligned_template': at
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

print("✅ Recherche templates OPTIMISÉE (k-mer + top 25)")

# %% [markdown]
# ## 🎯 9. Adaptation template

# %%
def adapt_template(query_seq, template):
    """Adapte template avec interpolation pandas."""
    aq = template['aligned_query']
    at = template['aligned_template']
    tc = template['coords']

    n = len(query_seq)
    if n == 0:
        return np.zeros((0, 3))

    coords = np.full((n, 3), np.nan)
    qi, ti = 0, 0

    for i in range(len(aq)):
        if qi >= n:
            break
        if aq[i] != '-' and at[i] != '-':
            if ti < len(tc):
                coords[qi] = tc[ti]
            ti += 1
            qi += 1
        elif aq[i] != '-':
            qi += 1
        else:
            ti += 1

    # Interpolation pandas
    df = pd.DataFrame(coords)
    df = df.interpolate(method='linear', limit_direction='both')
    coords = df.values

    if np.isnan(coords).any():
        coords = np.nan_to_num(coords)

    return coords

print("✅ Adaptation template")

# %% [markdown]
# ## 🔧 10. Contraintes avancées

# %%
def apply_constraints(coords, sequence, base_pairs=None, confidence=1.0, iterations=3):
    """
    Contraintes itératives avec base-pairing.
    """
    refined = coords.copy()
    n = len(sequence)

    if n <= 1:
        return refined

    strength = 0.6 * (1.0 - min(confidence, 0.85))

    for _ in range(iterations):
        # 1. Distance séquentielle (5.5-6.5Å)
        for i in range(n - 1):
            vec = refined[i+1] - refined[i]
            dist = np.linalg.norm(vec)

            if dist < 0.1:
                vec = np.random.normal(0, 1, 3)
                dist = np.linalg.norm(vec)

            vec = vec / dist

            if dist < 5.5 or dist > 6.5:
                target = 6.0
                adj = (target - dist) * strength
                refined[i+1] = refined[i] + vec * (dist + adj)

        # 2. Contraintes base-pairing
        if base_pairs and strength > 0.1:
            bp_dist = 10.5  # Distance C1'-C1' typique

            for i, j, s in base_pairs[:15]:  # Top 15 paires
                if i >= n or j >= n:
                    continue

                vec = refined[j] - refined[i]
                dist = np.linalg.norm(vec)

                if dist < 0.1:
                    continue

                vec = vec / dist

                # Attirer vers distance idéale
                if dist < 8.0 or dist > 13.0:
                    adj = (bp_dist - dist) * strength * s * 0.3
                    refined[i] -= vec * adj * 0.5
                    refined[j] += vec * adj * 0.5

        # 3. Éviter clashes
        for i in range(n):
            for j in range(i + 2, min(i + 10, n)):
                vec = refined[j] - refined[i]
                dist = np.linalg.norm(vec)

                if 0.1 < dist < 3.5:
                    vec = vec / dist
                    push = (3.5 - dist) * 0.4
                    refined[i] -= vec * push
                    refined[j] += vec * push

    return refined

print("✅ Contraintes avancées avec base-pairing")

# %% [markdown]
# ## 🧬 11. Génération de novo avancée

# %%
def generate_denovo(sequence, seed=None, use_bp=True):
    """Structure de novo avec base-pairing."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))

    coords = np.zeros((n, 3))

    # Initialisation hélicale
    for i in range(min(5, n)):
        theta = i * 0.6
        coords[i] = [10 * np.cos(theta), 10 * np.sin(theta), i * 2.8]

    if n <= 5:
        return coords

    # Paires de bases
    if use_bp:
        bp = predict_base_pairs(sequence)
        bp_dict = {i: j for i, j, _ in bp}
        bp_dict.update({j: i for i, j, _ in bp})
    else:
        bp_dict = {}

    direction = np.array([0.0, 0.0, 1.0])
    complement = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}

    for i in range(5, n):
        # Si partenaire déjà placé
        if i in bp_dict and bp_dict[i] < i:
            partner = bp_dict[i]
            partner_pos = coords[partner]

            # Positionner à ~10.5Å du partenaire
            center = np.mean(coords[:i], axis=0)
            to_center = center - partner_pos
            norm = np.linalg.norm(to_center)
            if norm > 0:
                to_center = to_center / norm

            coords[i] = partner_pos + to_center * (10.5 + random.uniform(-1, 1))
            coords[i] += np.random.normal(0, 1.0, 3)
        else:
            # Continuer la chaîne
            if random.random() < 0.2:
                angle = random.uniform(0.2, 0.5)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                c, s = np.cos(angle), np.sin(angle)
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                R = np.eye(3) + s * K + (1 - c) * (K @ K)
                direction = R @ direction

            step = random.uniform(4.5, 6.5)
            coords[i] = coords[i-1] + step * direction
            direction += np.random.normal(0, 0.1, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

    return coords

print("✅ Génération de novo avancée")

# %% [markdown]
# ## 🎯 12. Ensemble de templates

# %%
def ensemble_prediction(query_seq, templates, base_pairs):
    """
    Combine plusieurs templates avec moyenne pondérée.
    """
    if not templates:
        return None

    n = len(query_seq)
    weights = []
    adapted_list = []

    # Adapter tous les templates
    for t in templates:
        adapted = adapt_template(query_seq, t)
        if len(adapted) == n:
            sim = t['similarity']
            # Poids = similarité au carré (favorise les meilleurs)
            weight = sim ** 2
            weights.append(weight)
            adapted_list.append(adapted)

    if not adapted_list:
        return None

    # Normaliser les poids
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / total_weight for w in weights]

    # Superposer sur le premier (meilleur) template
    reference = adapted_list[0]

    for i in range(1, len(adapted_list)):
        adapted_list[i] = kabsch_superpose(adapted_list[i], reference)

    # Moyenne pondérée
    ensemble = np.zeros((n, 3))
    for coords, w in zip(adapted_list, weights):
        ensemble += coords * w

    # Appliquer contraintes
    confidence = templates[0]['similarity'] if templates else 0.5
    refined = apply_constraints(ensemble, query_seq, base_pairs, confidence)

    return refined

print("✅ Ensemble de templates")

# %% [markdown]
# ## 🎯 13. Pipeline de prédiction

# %%
def predict_structures(sequence, target_id, seqs_df, coords_dict, n_preds=5, temporal_cutoff=None):
    """
    Génère 5 prédictions diversifiées.
    """
    predictions = []
    n = len(sequence)

    if n == 0:
        return [np.zeros((0, 3)) for _ in range(n_preds)]

    # Prédire base pairs
    base_pairs = predict_base_pairs(sequence)

    # Trouver templates
    templates = find_templates(sequence, seqs_df, coords_dict, temporal_cutoff, top_n=10)

    # Stratégie 1: Ensemble des meilleurs templates
    if templates:
        ensemble = ensemble_prediction(sequence, templates[:5], base_pairs)
        if ensemble is not None and len(ensemble) == n:
            # Version avec peu de bruit
            noise = max(0.02, 0.4 - templates[0]['similarity'])
            predictions.append(ensemble + np.random.normal(0, noise, ensemble.shape))

    # Stratégie 2: Templates individuels avec variations
    for t in templates:
        if len(predictions) >= n_preds:
            break

        adapted = adapt_template(sequence, t)
        if len(adapted) != n:
            continue

        sim = t['similarity']
        refined = apply_constraints(adapted, sequence, base_pairs, sim)

        # Bruit proportionnel à l'incertitude
        noise = max(0.03, 0.6 - sim)
        predictions.append(refined + np.random.normal(0, noise, refined.shape))

    # Stratégie 3: De novo avec base-pairing
    seed_base = hash(target_id) % 100000

    while len(predictions) < n_preds:
        seed = seed_base + len(predictions) * 7777
        denovo = generate_denovo(sequence, seed, use_bp=True)
        refined = apply_constraints(denovo, sequence, base_pairs, 0.3, iterations=5)
        predictions.append(refined + np.random.normal(0, 0.8, refined.shape))

    return predictions[:n_preds]

print("✅ Pipeline de prédiction")

# %% [markdown]
# ## 📝 14. Génération soumission

# %%
def generate_submission(test_df, seqs_df, coords_dict, output='submission.csv'):
    """Génère la soumission."""
    print("=" * 60)
    print("     🚀 GÉNÉRATION DE LA SOUMISSION")
    print("=" * 60)

    start = time.time()
    all_rows = []
    total = len(test_df)

    for idx, row in test_df.iterrows():
        tid = row['target_id']
        seq = row['sequence']
        cutoff = row.get('temporal_cutoff', None)

        if idx % 5 == 0:
            elapsed = time.time() - start
            eta = (total - idx) * elapsed / (idx + 1) if idx > 0 else 0
            print(f"   [{idx+1}/{total}] {tid[:25]:25s} | {len(seq):4d} nt | {elapsed:6.1f}s | ETA: {eta:.0f}s", flush=True)

        preds = predict_structures(seq, tid, seqs_df, coords_dict, 5, cutoff)

        # Vérifier dimensions
        for i in range(5):
            if len(preds[i]) != len(seq):
                preds[i] = generate_denovo(seq, seed=i)

        for j in range(len(seq)):
            row_data = {'ID': f"{tid}_{j+1}", 'resname': seq[j], 'resid': j + 1}
            for i in range(5):
                row_data[f'x_{i+1}'] = round(float(preds[i][j][0]), 3)
                row_data[f'y_{i+1}'] = round(float(preds[i][j][1]), 3)
                row_data[f'z_{i+1}'] = round(float(preds[i][j][2]), 3)
            all_rows.append(row_data)

    df = pd.DataFrame(all_rows)
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1, 6) for c in ['x', 'y', 'z']]
    df = df[cols]
    df.to_csv(output, index=False)

    print("=" * 60)
    print(f"✅ {output} | {total} séquences | {time.time()-start:.1f}s")
    print("=" * 60)

    return df

# %% [markdown]
# ## 🚀 15. Exécution

# %%
print("=" * 60)
print("     🧬 RNA 3D FOLDING - PIPELINE v5 (Ultra-Optimisé)")
print("=" * 60)
print(f"""
📋 DONNÉES:
   • Templates: {len(all_coords)} (train + valid fusionnés)
   • Test: {len(test_seqs)} séquences

🔧 OPTIMISATIONS v5:
   • BioPython: {'✅' if BIOPYTHON_OK else '⚠️ Fallback'}
   • Templates: train + validation fusionnés
   • Ensemble: moyenne pondérée + superposition Kabsch
   • Base-pairing: A-U, G-C, G-U wobble
   • Contraintes: itératives avec BP
   • top_n=10, bruit ultra-réduit

🎯 Score attendu: 400+
""")
print("=" * 60)

# %%
submission = generate_submission(test_seqs, all_seqs, all_coords, 'submission.csv')
print("\n📊 Aperçu:")
print(submission.head())
