# %% [markdown]
# # 🧬 RNA 3D Folding - Pipeline v5 (Ultra-Optimisé)

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
    print("⚠️ BioPython non disponible - fallback activé")

# %% [markdown]
# ## 📦 2. Imports

# %%
import pandas as pd
import numpy as np
from scipy.linalg import svd
import random
import time
from pathlib import Path
import warnings
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

# FUSION Train + Valid
all_seqs = pd.concat([train_seqs, valid_seqs], ignore_index=True)
all_coords = {**train_coords, **valid_coords}
print(f"✅ {len(all_coords)} structures (train + valid)")

# %% [markdown]
# ## 🔧 5. Fonctions utilitaires

# %%
def kmer_similarity(seq1, seq2, k=4):
    """Similarité k-mer ultra-rapide."""
    if len(seq1) < k or len(seq2) < k:
        return sum(a == b for a, b in zip(seq1, seq2)) / max(len(seq1), len(seq2), 1)
    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
    if not kmers1 or not kmers2:
        return 0.0
    return len(kmers1 & kmers2) / len(kmers1 | kmers2)

def align_sequences(seq1, seq2):
    """Alignement BioPython ou fallback NW."""
    if BIOPYTHON_OK:
        try:
            alignments = pairwise2.align.globalms(Seq(seq1), seq2, 2, -1, -10, -0.5, one_alignment_only=True)
            if alignments:
                a = alignments[0]
                score = a.score / (2 * min(len(seq1), len(seq2)))
                return max(0, score), str(a.seqA), str(a.seqB)
        except:
            pass

    # Fallback NW
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
# ## 🔍 6. Recherche de templates (ULTRA-RAPIDE)

# %%
def find_templates(query_seq, seqs_df, coords_dict, temporal_cutoff=None, top_n=10):
    """Recherche avec pré-filtrage k-mer (top 15 candidats seulement)."""
    if temporal_cutoff is not None and 'temporal_cutoff' in seqs_df.columns:
        filtered = seqs_df[seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered = seqs_df

    # Pré-filtrage k-mer RAPIDE
    candidates = []
    for _, row in filtered.iterrows():
        tid = row['target_id']
        tseq = row['sequence']

        if tid not in coords_dict:
            continue

        len_ratio = min(len(query_seq), len(tseq)) / max(len(query_seq), len(tseq))
        if len_ratio < 0.4:
            continue

        kmer_sim = kmer_similarity(query_seq, tseq, k=4)
        candidates.append((tid, tseq, kmer_sim))

    # Top 15 par k-mer
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:15]

    # Alignement sur top 15 seulement
    results = []
    for tid, tseq, _ in candidates:
        sim, aq, at = align_sequences(query_seq, tseq)
        results.append({
            'target_id': tid, 'sequence': tseq, 'similarity': sim,
            'coords': coords_dict[tid], 'aligned_query': aq, 'aligned_template': at
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

print("✅ Recherche templates (k-mer + top 15)")

# %% [markdown]
# ## 🎯 7. Adaptation et contraintes

# %%
def adapt_template(query_seq, template):
    """Adapte template avec interpolation pandas."""
    aq, at, tc = template['aligned_query'], template['aligned_template'], template['coords']
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
            ti += 1; qi += 1
        elif aq[i] != '-':
            qi += 1
        else:
            ti += 1

    df = pd.DataFrame(coords)
    df = df.interpolate(method='linear', limit_direction='both')
    coords = df.values
    return np.nan_to_num(coords)

def apply_constraints(coords, sequence, confidence=1.0):
    """Contraintes géométriques."""
    refined = coords.copy()
    n = len(sequence)
    if n <= 1:
        return refined

    strength = 0.6 * (1.0 - min(confidence, 0.85))

    for _ in range(2):
        for i in range(n - 1):
            vec = refined[i+1] - refined[i]
            dist = np.linalg.norm(vec)
            if dist < 0.1:
                vec = np.random.normal(0, 1, 3)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            if dist < 5.5 or dist > 6.5:
                adj = (6.0 - dist) * strength
                refined[i+1] = refined[i] + vec * (dist + adj)

    return refined

def generate_denovo(sequence, seed=None):
    """Structure de novo."""
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))
    coords = np.zeros((n, 3))
    for i in range(n):
        theta = i * 0.6
        coords[i] = [10 * np.cos(theta), 10 * np.sin(theta), i * 2.8]
    return coords + np.random.normal(0, 0.5, coords.shape)

print("✅ Adaptation et contraintes")

# %% [markdown]
# ## 🎯 8. Pipeline de prédiction

# %%
def predict_structures(sequence, target_id, seqs_df, coords_dict, n_preds=5, temporal_cutoff=None):
    """Prédit 5 structures."""
    n = len(sequence)
    if n == 0:
        return [np.zeros((0, 3)) for _ in range(n_preds)]

    predictions = []
    templates = find_templates(sequence, seqs_df, coords_dict, temporal_cutoff, top_n=8)

    for t in templates:
        if len(predictions) >= n_preds:
            break
        adapted = adapt_template(sequence, t)
        if len(adapted) != n:
            continue
        refined = apply_constraints(adapted, sequence, t['similarity'])
        noise = max(0.03, 0.6 - t['similarity'])
        predictions.append(refined + np.random.normal(0, noise, refined.shape))

    seed_base = hash(target_id) % 100000
    while len(predictions) < n_preds:
        denovo = generate_denovo(sequence, seed_base + len(predictions) * 777)
        predictions.append(apply_constraints(denovo, sequence, 0.3))

    return predictions[:n_preds]

print("✅ Pipeline de prédiction")

# %% [markdown]
# ## 📝 9. Génération soumission

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
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / speed if speed > 0 else 0
            print(f"   [{idx+1}/{total}] {tid[:20]:20s} | {len(seq):4d} nt | {elapsed:5.1f}s | ETA: {eta:5.0f}s", flush=True)

        preds = predict_structures(seq, tid, seqs_df, coords_dict, 5, cutoff)

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
# ## 🚀 10. Exécution

# %%
print("=" * 60)
print("     🧬 RNA 3D FOLDING - PIPELINE v5")
print("=" * 60)
print(f"""
📋 DONNÉES:
   • Templates: {len(all_coords)} (train + valid)
   • Test: {len(test_seqs)} séquences

🔧 OPTIMISATIONS:
   • BioPython: {'✅' if BIOPYTHON_OK else '⚠️ Fallback'}
   • K-mer pré-filtrage (top 15)
   • Alignement sur 15 candidats max
""")
print("=" * 60)

# %%
submission = generate_submission(test_seqs, all_seqs, all_coords, 'submission.csv')
print(submission.head())
