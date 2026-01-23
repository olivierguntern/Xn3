# %% [markdown]
# # 🧬 RNA 3D Folding - Code 352 SANS BioPython

# %% [markdown]
# ## 📦 1. Imports (aucune dépendance externe)

# %%
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
print("✅ Imports OK (sans BioPython)")

# %% [markdown]
# ## 🔧 2. Alignement Needleman-Wunsch (remplace BioPython)

# %%
def align_sequences(seq1, seq2):
    """
    Needleman-Wunsch avec mêmes paramètres que BioPython:
    match=2, mismatch=-1, gap_open=-10, gap_extend=-0.5
    Simplifié: gap=-2 (approximation)
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0, seq1, seq2

    # Matrice DP
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i, 0] = i * -2
    for j in range(m + 1):
        dp[0, j] = j * -2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                match = dp[i-1, j-1] + 2
            else:
                match = dp[i-1, j-1] - 1
            delete = dp[i-1, j] - 2
            insert = dp[i, j-1] - 2
            dp[i, j] = max(match, delete, insert)

    # Traceback
    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                score_diag = dp[i-1, j-1] + 2
            else:
                score_diag = dp[i-1, j-1] - 1

            if dp[i, j] == score_diag:
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
        elif i > 0:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1

    score = dp[n, m] / (2 * min(n, m)) if min(n, m) > 0 else 0
    return max(0, min(1, score)), ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

print("✅ Alignement NW défini")

# %% [markdown]
# ## 📊 3. Chargement

# %%
print("📂 Chargement...")
train_seqs = pd.read_csv(DATA_PATH / 'train_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH / 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')

print(f"✅ Train: {len(train_seqs)}, Test: {len(test_seqs)}")

# %% [markdown]
# ## 🔧 4. Traitement labels

# %%
def process_labels(labels_df):
    coords_dict = {}
    for id_prefix, group in labels_df.groupby(lambda x: labels_df['ID'][x].split('_')[0]):
        coords = []
        for _, row in group.sort_values('resid').iterrows():
            coords.append([row['x_1'], row['y_1'], row['z_1']])
        coords_dict[id_prefix] = np.array(coords)
    return coords_dict

print("📊 Traitement labels...")
train_coords_dict = process_labels(train_labels)
print(f"✅ {len(train_coords_dict)} structures")

# %% [markdown]
# ## 🔍 5. Recherche templates

# %%
def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, temporal_cutoff=None, top_n=8):
    similar_seqs = []

    if temporal_cutoff is not None and 'temporal_cutoff' in train_seqs_df.columns:
        filtered = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered = train_seqs_df

    for _, row in filtered.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        if target_id not in train_coords_dict:
            continue

        # Filtre longueur 50%
        if abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq)) > 0.5:
            continue

        sim, aligned_q, aligned_t = align_sequences(query_seq, train_seq)
        similar_seqs.append((target_id, train_seq, sim, train_coords_dict[target_id], aligned_q, aligned_t))

    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]

print("✅ find_similar_sequences")

# %% [markdown]
# ## 🎯 6. Adaptation template

# %%
def adapt_template_to_query(query_seq, template_seq, template_coords, aligned_q=None, aligned_t=None):
    # Utiliser l'alignement pré-calculé ou en faire un nouveau
    if aligned_q is None or aligned_t is None:
        _, aligned_q, aligned_t = align_sequences(query_seq, template_seq)

    if len(aligned_q) == 0:
        return generate_basic_structure(query_seq)

    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)

    query_idx = 0
    template_idx = 0

    for i in range(len(aligned_q)):
        q_char = aligned_q[i]
        t_char = aligned_t[i]

        if q_char != '-' and t_char != '-':
            if template_idx < len(template_coords) and query_idx < len(query_seq):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif q_char != '-' and t_char == '-':
            query_idx += 1
        elif q_char == '-' and t_char != '-':
            template_idx += 1

    # Interpolation pandas
    df_coords = pd.DataFrame(query_coords)
    df_coords = df_coords.interpolate(method='linear', limit_direction='both')
    query_coords = df_coords.values

    if np.isnan(query_coords).any():
        query_coords = np.nan_to_num(query_coords)

    return query_coords

def generate_basic_structure(sequence):
    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))
    coords = np.zeros((n, 3))
    for i in range(n):
        angle = i * 0.6
        coords[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    return coords

print("✅ adapt_template_to_query")

# %% [markdown]
# ## 🔧 7. Contraintes

# %%
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    refined = coordinates.copy()
    n = len(sequence)

    if n <= 1:
        return refined

    strength = 0.8 * (1.0 - min(confidence, 0.8))

    for i in range(n - 1):
        curr = refined[i]
        next_pos = refined[i + 1]
        dist = np.linalg.norm(next_pos - curr)

        if dist < 5.5 or dist > 6.5:
            target = 6.0
            direc = next_pos - curr
            norm = np.linalg.norm(direc)

            if norm < 1e-10:
                direc = np.random.normal(0, 1, 3)
                norm = np.linalg.norm(direc)

            direc = direc / norm
            adj = (target - dist) * strength
            refined[i + 1] = curr + direc * (dist + adj)

    return refined

print("✅ adaptive_rna_constraints")

# %% [markdown]
# ## 🎯 8. Prédiction

# %%
def predict_rna_structures(sequence, target_id, train_seqs, train_coords, n_preds=5, temporal_cutoff=None):
    predictions = []

    if len(sequence) == 0:
        return [np.zeros((0, 3)) for _ in range(n_preds)]

    similar = find_similar_sequences(sequence, train_seqs, train_coords, temporal_cutoff, top_n=8)

    if similar:
        for templ_id, templ_seq, sim, templ_coords, aligned_q, aligned_t in similar:
            try:
                adapted = adapt_template_to_query(sequence, templ_seq, templ_coords, aligned_q, aligned_t)

                if len(adapted) != len(sequence):
                    continue

                refined = adaptive_rna_constraints(adapted, sequence, confidence=sim)
                rand_scale = max(0.03, 0.6 - sim)
                final = refined + np.random.normal(0, rand_scale, refined.shape)
                predictions.append(final)

                if len(predictions) >= n_preds:
                    break
            except Exception as e:
                continue

    while len(predictions) < n_preds:
        base = generate_basic_structure(sequence)
        base = base + np.random.normal(0, 1.0, base.shape)
        predictions.append(base)

    return predictions[:n_preds]

print("✅ predict_rna_structures")

# %% [markdown]
# ## 📝 9. Génération soumission

# %%
all_predictions = []
start_time = time.time()
total_targets = len(test_seqs)

print("=" * 60)
print("🚀 GÉNÉRATION (sans BioPython)")
print("=" * 60)

for idx, row in test_seqs.iterrows():
    t_id = row['target_id']
    seq = row['sequence']
    cutoff = row.get('temporal_cutoff', None)

    if idx % 5 == 0:
        elapsed = time.time() - start_time
        eta = (total_targets - idx) * elapsed / (idx + 1) if idx > 0 else 0
        print(f"   [{idx+1}/{total_targets}] {t_id} | {elapsed:.1f}s | ETA: {eta:.0f}s", flush=True)

    try:
        preds = predict_rna_structures(seq, t_id, train_seqs, train_coords_dict,
                                       n_preds=5, temporal_cutoff=cutoff)
    except Exception as e:
        print(f"   ⚠️ Erreur {t_id}: {e}")
        preds = [generate_basic_structure(seq) for _ in range(5)]

    # Vérifier dimensions
    for i in range(5):
        if len(preds[i]) != len(seq):
            preds[i] = generate_basic_structure(seq)

    for j in range(len(seq)):
        pred_row = {
            'ID': f"{t_id}_{j+1}",
            'resname': seq[j],
            'resid': j + 1
        }

        for i in range(5):
            pred_row[f'x_{i+1}'] = float(preds[i][j][0])
            pred_row[f'y_{i+1}'] = float(preds[i][j][1])
            pred_row[f'z_{i+1}'] = float(preds[i][j][2])

        all_predictions.append(pred_row)

submission_df = pd.DataFrame(all_predictions)

cols = ['ID', 'resname', 'resid']
for i in range(1, 6):
    cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

submission_df = submission_df[cols]
submission_df.to_csv('submission.csv', index=False)

print("=" * 60)
print(f"🎉 TERMINÉ! Temps: {time.time() - start_time:.1f}s")
print("=" * 60)
print(submission_df.head())
