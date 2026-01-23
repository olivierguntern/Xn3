# %% [markdown]
# # 🧬 RNA 3D Folding - Copie exacte du code 352

# %% [markdown]
# ## 📦 1. Installation BioPython

# %%
import subprocess
import sys

print("📦 Installation BioPython...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "biopython", "-q"])
    from Bio import pairwise2
    from Bio.Seq import Seq
    BIOPYTHON_OK = True
    print("✅ BioPython installé!")
except:
    BIOPYTHON_OK = False
    print("⚠️ Fallback activé")

# %% [markdown]
# ## 📦 2. Imports

# %%
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
print("✅ Imports OK")

# %% [markdown]
# ## 📊 3. Chargement

# %%
print("📂 Chargement...")
train_seqs = pd.read_csv(DATA_PATH / 'train_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH / 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')

print(f"✅ Train: {len(train_seqs)}, Test: {len(test_seqs)}")

# %% [markdown]
# ## 🔧 4. Traitement labels (EXACTEMENT comme code 352)

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
# ## 🔍 5. Recherche templates (EXACTEMENT comme code 352)

# %%
def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, temporal_cutoff=None, top_n=8):
    similar_seqs = []
    query_seq_obj = Seq(query_seq)

    if temporal_cutoff:
        filtered_train_seqs = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered_train_seqs = train_seqs_df

    for _, row in filtered_train_seqs.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        if target_id not in train_coords_dict:
            continue

        if abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq)) > 0.5:
            continue

        alignments = pairwise2.align.globalms(query_seq_obj, train_seq, 2, -1, -10, -0.5, one_alignment_only=True)

        if alignments:
            alignment = alignments[0]
            similarity_score = alignment.score / (2 * min(len(query_seq), len(train_seq)))
            similar_seqs.append((target_id, train_seq, similarity_score, train_coords_dict[target_id]))

    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]

print("✅ find_similar_sequences (code 352)")

# %% [markdown]
# ## 🎯 6. Adaptation template (EXACTEMENT comme code 352)

# %%
def adapt_template_to_query(query_seq, template_seq, template_coords):
    query_seq_obj = Seq(query_seq)
    template_seq_obj = Seq(template_seq)
    alignments = pairwise2.align.globalms(query_seq_obj, template_seq_obj, 2, -1, -10, -0.5, one_alignment_only=True)

    if not alignments:
        return generate_basic_structure(query_seq)

    alignment = alignments[0]
    aligned_query = alignment.seqA
    aligned_template = alignment.seqB

    query_coords = np.zeros((len(query_seq), 3))
    query_coords.fill(np.nan)

    query_idx = 0
    template_idx = 0

    for i in range(len(aligned_query)):
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

    df_coords = pd.DataFrame(query_coords)
    df_coords = df_coords.interpolate(method='linear', limit_direction='both')
    query_coords = df_coords.values
    if np.isnan(query_coords).any():
        query_coords = np.nan_to_num(query_coords)
    return query_coords

def generate_basic_structure(sequence):
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        angle = i * 0.6
        coords[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]
    return coords

print("✅ adapt_template_to_query (code 352)")

# %% [markdown]
# ## 🔧 7. Contraintes (EXACTEMENT comme code 352)

# %%
def adaptive_rna_constraints(coordinates, sequence, confidence=1.0):
    refined = coordinates.copy()
    n = len(sequence)
    strength = 0.8 * (1.0 - min(confidence, 0.8))
    for i in range(n - 1):
        curr, next_pos = refined[i], refined[i+1]
        dist = np.linalg.norm(next_pos - curr)
        if dist < 5.5 or dist > 6.5:
            target = 6.0
            direc = (next_pos - curr)
            norm = np.linalg.norm(direc)
            if norm < 1e-10:
                direc = np.random.normal(0, 1, 3)
            else:
                direc = direc / norm
            adj = (target - dist) * strength
            refined[i+1] = curr + direc * (dist + adj)
    return refined

print("✅ adaptive_rna_constraints (code 352)")

# %% [markdown]
# ## 🎯 8. Prédiction (EXACTEMENT comme code 352)

# %%
def predict_rna_structures(sequence, target_id, train_seqs, train_coords, n_preds=5, temporal_cutoff=None):
    predictions = []
    similar = find_similar_sequences(sequence, train_seqs, train_coords, temporal_cutoff, top_n=8)

    if similar:
        for templ_id, templ_seq, sim, templ_coords in similar:
            adapted = adapt_template_to_query(sequence, templ_seq, templ_coords)
            refined = adaptive_rna_constraints(adapted, sequence, confidence=sim)

            rand_scale = max(0.03, 0.6 - sim)

            final = refined + np.random.normal(0, rand_scale, refined.shape)
            predictions.append(final)
            if len(predictions) >= n_preds:
                break

    while len(predictions) < n_preds:
        base = generate_basic_structure(sequence)
        base += np.random.normal(0, 1.0, base.shape)
        predictions.append(base)

    return predictions[:n_preds]

print("✅ predict_rna_structures (code 352)")

# %% [markdown]
# ## 📝 9. Génération soumission

# %%
all_predictions = []
start_time = time.time()
total_targets = len(test_seqs)

print("=" * 60)
print("🚀 GÉNÉRATION (copie exacte code 352)")
print("=" * 60)

for idx, row in test_seqs.iterrows():
    t_id = row['target_id']
    seq = row['sequence']
    cutoff = row['temporal_cutoff'] if 'temporal_cutoff' in row else None

    if idx % 5 == 0:
        elapsed = time.time() - start_time
        eta = (total_targets - idx) * elapsed / (idx + 1) if idx > 0 else 0
        print(f"   [{idx+1}/{total_targets}] {t_id} | {elapsed:.1f}s | ETA: {eta:.0f}s", flush=True)

    preds = predict_rna_structures(seq, t_id, train_seqs, train_coords_dict,
                                   n_preds=5, temporal_cutoff=cutoff)

    for j in range(len(seq)):
        pred_row = {
            'ID': f"{t_id}_{j+1}",
            'resname': seq[j],
            'resid': j + 1
        }

        for i in range(5):
            pred_row[f'x_{i+1}'] = preds[i][j][0]
            pred_row[f'y_{i+1}'] = preds[i][j][1]
            pred_row[f'z_{i+1}'] = preds[i][j][2]

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
