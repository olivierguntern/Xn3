# %% [markdown]
# # 🧬 RNA 3D Folding - Pipeline v4 (Optimisé)
#
# Basé sur l'analyse du notebook scoring 352:
# - BioPython via .whl (ajouter comme dataset)
# - Parsing ID corrigé
# - Interpolation pandas robuste
# - Paramètres optimisés

# %% [markdown]
# ## 📦 1. Installation BioPython (via .whl)

# %%
import os
import sys
import subprocess
import warnings

print("📦 Recherche du fichier BioPython .whl...")

whl_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        if file.endswith('.whl') and 'bio' in file.lower():
            whl_path = os.path.join(root, file)
            print(f"✅ Fichier trouvé: {whl_path}")
            break
    if whl_path:
        break

if whl_path:
    try:
        print("Installation en cours...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path, "--no-deps", "--no-index", "-q"])
        print("🎉 BioPython installé avec succès!")
    except Exception as e:
        print(f"❌ Erreur installation: {e}")
        whl_path = None
else:
    print("⚠️ Fichier .whl non trouvé - utilisation du fallback")

# Tester l'import
BIOPYTHON_AVAILABLE = False
try:
    from Bio import pairwise2
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython chargé!")
except ImportError:
    print("⚠️ BioPython non disponible - fallback activé")

# %% [markdown]
# ## 📦 2. Imports standards

# %%
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
warnings.filterwarnings('ignore')

DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
print("✅ Imports OK")

# %% [markdown]
# ## 📊 3. Chargement des données

# %%
print("📂 Chargement des données...")
train_seqs = pd.read_csv(DATA_PATH / 'train_sequences.csv')
valid_seqs = pd.read_csv(DATA_PATH / 'validation_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH / 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')
valid_labels = pd.read_csv(DATA_PATH / 'validation_labels.csv')

print(f"✅ Train: {len(train_seqs)}, Valid: {len(valid_seqs)}, Test: {len(test_seqs)}")

# %% [markdown]
# ## 🔧 4. Traitement des labels (CORRIGÉ)

# %%
def process_labels(labels_df):
    """
    Crée un dict target_id -> coordonnées 3D.
    IMPORTANT: Utilise split('_')[0] pour correspondre aux target_id!
    """
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
print(f"✅ {len(train_coords)} structures train, {len(valid_coords)} structures valid")

# %% [markdown]
# ## 🔍 5. Alignement (BioPython ou Fallback)

# %%
if BIOPYTHON_AVAILABLE:
    def align_sequences(seq1, seq2):
        """Alignement avec BioPython pairwise2."""
        alignments = pairwise2.align.globalms(
            Seq(seq1), seq2,
            2, -1, -10, -0.5,
            one_alignment_only=True
        )
        if alignments:
            a = alignments[0]
            score = a.score / (2 * min(len(seq1), len(seq2)))
            return score, a.seqA, a.seqB
        return 0.0, seq1, seq2
    print("✅ Alignement: BioPython pairwise2")
else:
    def align_sequences(seq1, seq2):
        """Alignement Needleman-Wunsch fallback."""
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return 0.0, seq1, seq2

        # Matrice de scores
        dp = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            dp[i, 0] = i * -2
        for j in range(m + 1):
            dp[0, j] = j * -2

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i-1, j-1] + (2 if seq1[i-1] == seq2[j-1] else -1)
                delete = dp[i-1, j] - 2
                insert = dp[i, j-1] - 2
                dp[i, j] = max(match, delete, insert)

        # Traceback
        aligned1, aligned2 = [], []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (2 if seq1[i-1] == seq2[j-1] else -1):
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i, j] == dp[i-1, j] - 2:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1

        score = dp[n, m] / (2 * min(n, m)) if min(n, m) > 0 else 0
        return max(0, score), ''.join(reversed(aligned1)), ''.join(reversed(aligned2))
    print("✅ Alignement: Needleman-Wunsch fallback")

# %% [markdown]
# ## 🔍 6. Recherche de templates

# %%
def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, temporal_cutoff=None, top_n=8):
    """
    Trouve les séquences les plus similaires.
    top_n=8 comme dans le code qui score 352.
    """
    similar_seqs = []

    # Filtre temporal
    if temporal_cutoff is not None and 'temporal_cutoff' in train_seqs_df.columns:
        filtered_df = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered_df = train_seqs_df

    for _, row in filtered_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        if target_id not in train_coords_dict:
            continue

        # Filtre longueur (50%)
        len_diff = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
        if len_diff > 0.5:
            continue

        # Alignement
        similarity, aligned_q, aligned_t = align_sequences(query_seq, train_seq)

        similar_seqs.append({
            'target_id': target_id,
            'sequence': train_seq,
            'similarity': similarity,
            'coords': train_coords_dict[target_id],
            'aligned_query': aligned_q,
            'aligned_template': aligned_t
        })

    similar_seqs.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_seqs[:top_n]

print("✅ Recherche de templates définie (top_n=8)")

# %% [markdown]
# ## 🎯 7. Adaptation template → query

# %%
def adapt_template_to_query(query_seq, template_data):
    """
    Adapte les coordonnées avec interpolation pandas (robuste).
    """
    aligned_query = template_data['aligned_query']
    aligned_template = template_data['aligned_template']
    template_coords = template_data['coords']

    query_len = len(query_seq)
    if query_len == 0:
        return np.zeros((0, 3))

    # Initialiser avec NaN
    query_coords = np.zeros((query_len, 3))
    query_coords.fill(np.nan)

    query_idx = 0
    template_idx = 0

    for i in range(len(aligned_query)):
        if query_idx >= query_len:
            break

        q_char = aligned_query[i]
        t_char = aligned_template[i]

        if q_char != '-' and t_char != '-':
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif q_char != '-' and t_char == '-':
            query_idx += 1
        elif q_char == '-' and t_char != '-':
            template_idx += 1

    # Interpolation PANDAS (comme le code 352)
    df_coords = pd.DataFrame(query_coords)
    df_coords = df_coords.interpolate(method='linear', limit_direction='both')
    query_coords = df_coords.values

    # Fallback pour NaN restants
    if np.isnan(query_coords).any():
        query_coords = np.nan_to_num(query_coords)

    return query_coords

def generate_basic_structure(sequence):
    """Structure hélicale de base."""
    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))

    coords = np.zeros((n, 3))
    for i in range(n):
        angle = i * 0.6
        coords[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    return coords

print("✅ Adaptation template définie (interpolation pandas)")

# %% [markdown]
# ## 🔧 8. Contraintes adaptatives

# %%
def adaptive_constraints(coords, sequence, confidence=1.0):
    """
    Contraintes géométriques adaptatives.
    Force réduite si haute confiance.
    """
    refined = coords.copy()
    n = len(sequence)

    if n <= 1:
        return refined

    # Force inversement proportionnelle à la confiance
    strength = 0.8 * (1.0 - min(confidence, 0.8))

    # Distance séquentielle (5.5-6.5Å, cible 6.0Å)
    for i in range(n - 1):
        curr = refined[i]
        next_pos = refined[i + 1]
        dist = np.linalg.norm(next_pos - curr)

        if dist < 5.5 or dist > 6.5:
            target = 6.0
            direction = next_pos - curr
            norm = np.linalg.norm(direction)

            if norm < 1e-10:
                direction = np.random.normal(0, 1, 3)
                norm = np.linalg.norm(direction)

            direction = direction / norm
            adjustment = (target - dist) * strength
            refined[i + 1] = curr + direction * (dist + adjustment)

    return refined

print("✅ Contraintes adaptatives définies")

# %% [markdown]
# ## 🎯 9. Pipeline de prédiction

# %%
def predict_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_preds=5, temporal_cutoff=None):
    """
    Prédit 5 structures diversifiées.
    """
    predictions = []

    # Edge case
    if len(sequence) == 0:
        return [np.zeros((0, 3)) for _ in range(n_preds)]

    # Trouver templates (top_n=8)
    similar = find_similar_sequences(
        sequence, train_seqs_df, train_coords_dict,
        temporal_cutoff, top_n=8
    )

    # Utiliser les templates
    for template in similar:
        adapted = adapt_template_to_query(sequence, template)

        if adapted is not None and len(adapted) == len(sequence):
            sim = template['similarity']
            refined = adaptive_constraints(adapted, sequence, confidence=sim)

            # Bruit RÉDUIT (comme le code 352)
            noise_scale = max(0.03, 0.6 - sim)
            final = refined + np.random.normal(0, noise_scale, refined.shape)

            predictions.append(final)

            if len(predictions) >= n_preds:
                break

    # Compléter avec structures de base
    while len(predictions) < n_preds:
        base = generate_basic_structure(sequence)
        base += np.random.normal(0, 1.0, base.shape)
        predictions.append(base)

    return predictions[:n_preds]

print("✅ Pipeline de prédiction défini")

# %% [markdown]
# ## 📝 10. Génération de la soumission

# %%
def generate_submission(test_df, train_seqs_df, train_coords_dict, output='submission.csv'):
    """Génère le fichier de soumission."""
    print("=" * 50)
    print("📝 GÉNÉRATION DE LA SOUMISSION")
    print("=" * 50)

    start = time.time()
    all_rows = []
    total = len(test_df)

    for idx, row in test_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal = row.get('temporal_cutoff', None)

        if idx % 5 == 0:
            elapsed = time.time() - start
            print(f"   [{idx+1}/{total}] {target_id} ({len(sequence)} nt) - {elapsed:.1f}s", flush=True)

        # Prédire
        preds = predict_structures(
            sequence, target_id, train_seqs_df, train_coords_dict,
            n_preds=5, temporal_cutoff=temporal
        )

        # Créer les lignes
        for j in range(len(sequence)):
            row_data = {
                'ID': f"{target_id}_{j+1}",
                'resname': sequence[j],
                'resid': j + 1
            }

            for p_idx in range(5):
                row_data[f'x_{p_idx+1}'] = preds[p_idx][j][0]
                row_data[f'y_{p_idx+1}'] = preds[p_idx][j][1]
                row_data[f'z_{p_idx+1}'] = preds[p_idx][j][2]

            all_rows.append(row_data)

    # DataFrame final
    df = pd.DataFrame(all_rows)
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1, 6) for c in ['x', 'y', 'z']]
    df = df[cols]
    df.to_csv(output, index=False)

    total_time = time.time() - start
    print("=" * 50)
    print(f"✅ Soumission: {output}")
    print(f"   {total} séquences en {total_time:.1f}s")
    print("=" * 50)

    return df

# %% [markdown]
# ## 🚀 11. Exécution

# %%
print("=" * 60)
print("     🧬 RNA 3D FOLDING - PIPELINE v4 (Optimisé)")
print("=" * 60)
print(f"""
📋 CONFIGURATION:
   • Train: {len(train_seqs)} séquences
   • Templates: {len(train_coords)} structures
   • Test: {len(test_seqs)} séquences

🔧 OPTIMISATIONS (basées sur code 352):
   • BioPython: {'✅ Actif' if BIOPYTHON_AVAILABLE else '⚠️ Fallback NW'}
   • ID parsing: split('_')[0] (corrigé)
   • Interpolation: pandas (robuste)
   • Templates: top_n=8
   • Bruit: max(0.03, 0.6-sim)

🚀 Génération en cours...
""")
print("=" * 60)

# %%
submission = generate_submission(test_seqs, train_seqs, train_coords, 'submission.csv')
print(submission.head())
