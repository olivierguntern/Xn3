# %% [markdown]
# # 🧬 RNA 3D Folding - Pipeline v3 (Sans dépendances externes)
#
# Version optimisée fonctionnant SANS internet:
# - Alignement Needleman-Wunsch custom (pas de BioPython)
# - Pré-filtrage k-mer rapide
# - Gestion robuste des edge cases

# %% [markdown]
# ## 📦 1. Imports (packages Kaggle standard uniquement)

# %%
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import random
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
print("✅ Imports OK (aucune dépendance externe)")

# %% [markdown]
# ## 🔧 2. Alignement Needleman-Wunsch (custom, sans BioPython)

# %%
def needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-2):
    """
    Alignement global Needleman-Wunsch implémenté from scratch.
    Retourne: score, seq1_aligned, seq2_aligned
    """
    n, m = len(seq1), len(seq2)

    # Edge cases
    if n == 0:
        return m * gap, '-' * m, seq2
    if m == 0:
        return n * gap, seq1, '-' * n

    # Matrice de scores
    score_matrix = np.zeros((n + 1, m + 1))

    # Initialisation
    for i in range(n + 1):
        score_matrix[i, 0] = i * gap
    for j in range(m + 1):
        score_matrix[0, j] = j * gap

    # Remplissage
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                diag = score_matrix[i-1, j-1] + match
            else:
                diag = score_matrix[i-1, j-1] + mismatch

            up = score_matrix[i-1, j] + gap
            left = score_matrix[i, j-1] + gap

            score_matrix[i, j] = max(diag, up, left)

    # Traceback
    aligned1, aligned2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                current_match = match
            else:
                current_match = mismatch

            if score_matrix[i, j] == score_matrix[i-1, j-1] + current_match:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap:
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

    return score_matrix[n, m], ''.join(reversed(aligned1)), ''.join(reversed(aligned2))

def quick_similarity(seq1, seq2, k=3):
    """Similarité k-mer rapide (O(n+m) au lieu de O(n*m))."""
    if len(seq1) < k or len(seq2) < k:
        # Fallback pour séquences courtes
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / max(len(seq1), len(seq2), 1)

    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))

    if not kmers1 or not kmers2:
        return 0.0

    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)

    return intersection / union if union > 0 else 0.0

print("✅ Alignement Needleman-Wunsch custom défini")

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
# ## 🔧 4. Traitement des labels

# %%
def process_labels(labels_df):
    """Crée un dict target_id -> coordonnées 3D."""
    coords_dict = {}

    for id_prefix, group in labels_df.groupby(lambda x: labels_df['ID'][x].rsplit('_', 1)[0]):
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
# ## 🔍 5. Recherche de templates (optimisée)

# %%
def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, temporal_cutoff=None, top_n=5):
    """
    Recherche RAPIDE de templates similaires.
    1. Pré-filtrage par longueur
    2. Pré-filtrage par k-mers (rapide)
    3. Alignement NW sur top candidats seulement
    """
    query_len = len(query_seq)

    # Edge case: séquence vide
    if query_len == 0:
        return []

    # Filtrer par temporal_cutoff
    if temporal_cutoff is not None and 'temporal_cutoff' in train_seqs_df.columns:
        filtered_df = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered_df = train_seqs_df

    candidates = []

    # ÉTAPE 1: Pré-filtrage rapide
    for _, row in filtered_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        if target_id not in train_coords_dict:
            continue

        train_len = len(train_seq)

        # Skip séquences vides
        if train_len == 0:
            continue

        # Skip si longueurs trop différentes (>40%)
        len_ratio = min(query_len, train_len) / max(query_len, train_len)
        if len_ratio < 0.6:
            continue

        # Similarité k-mer RAPIDE
        kmer_sim = quick_similarity(query_seq, train_seq, k=3)

        candidates.append({
            'target_id': target_id,
            'sequence': train_seq,
            'kmer_sim': kmer_sim,
            'coords': train_coords_dict[target_id]
        })

    # ÉTAPE 2: Garder top 15 candidats par k-mer
    candidates.sort(key=lambda x: x['kmer_sim'], reverse=True)
    candidates = candidates[:15]

    # ÉTAPE 3: Alignement NW sur les meilleurs seulement
    results = []

    for cand in candidates:
        score, aligned_query, aligned_template = needleman_wunsch(
            query_seq, cand['sequence']
        )

        # Normaliser le score
        max_score = 2 * min(query_len, len(cand['sequence']))
        similarity = score / max_score if max_score > 0 else 0
        similarity = max(0, min(1, similarity))  # Clamp [0, 1]

        results.append({
            'target_id': cand['target_id'],
            'sequence': cand['sequence'],
            'similarity': similarity,
            'coords': cand['coords'],
            'aligned_query': aligned_query,
            'aligned_template': aligned_template
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

print("✅ Recherche de templates optimisée")

# %% [markdown]
# ## 🎯 6. Adaptation template → query

# %%
def adapt_template_to_query(query_seq, template_data):
    """
    Adapte les coordonnées du template à la séquence query.
    Gestion robuste des gaps.
    """
    aligned_query = template_data['aligned_query']
    aligned_template = template_data['aligned_template']
    template_coords = template_data['coords']

    query_len = len(query_seq)

    # Edge case
    if query_len == 0:
        return np.zeros((0, 3))

    # Initialiser avec NaN
    query_coords = np.full((query_len, 3), np.nan)

    query_idx = 0
    template_idx = 0

    for i in range(len(aligned_query)):
        if query_idx >= query_len:
            break

        q_char = aligned_query[i]
        t_char = aligned_template[i]

        if q_char != '-' and t_char != '-':
            # Match ou mismatch - copier les coordonnées
            if template_idx < len(template_coords):
                query_coords[query_idx] = template_coords[template_idx]
            template_idx += 1
            query_idx += 1
        elif q_char != '-' and t_char == '-':
            # Gap dans le template - position à interpoler
            query_idx += 1
        elif q_char == '-' and t_char != '-':
            # Gap dans la query - skip la position template
            template_idx += 1

    # Interpoler les positions manquantes
    query_coords = interpolate_missing(query_coords)

    return query_coords

def interpolate_missing(coords):
    """Interpole les positions manquantes (NaN)."""
    n = len(coords)
    if n == 0:
        return coords

    typical_step = 5.9  # Distance C1'-C1' typique

    for i in range(n):
        if np.isnan(coords[i, 0]):
            # Trouver les voisins valides
            prev_valid = -1
            for j in range(i-1, -1, -1):
                if not np.isnan(coords[j, 0]):
                    prev_valid = j
                    break

            next_valid = -1
            for j in range(i+1, n):
                if not np.isnan(coords[j, 0]):
                    next_valid = j
                    break

            if prev_valid >= 0 and next_valid >= 0:
                # Interpolation linéaire
                t = (i - prev_valid) / (next_valid - prev_valid)
                coords[i] = (1 - t) * coords[prev_valid] + t * coords[next_valid]
            elif prev_valid >= 0:
                # Étendre depuis le précédent
                if prev_valid > 0 and not np.isnan(coords[prev_valid-1, 0]):
                    direction = coords[prev_valid] - coords[prev_valid-1]
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm * typical_step
                    else:
                        direction = np.array([typical_step, 0, 0])
                else:
                    direction = np.array([typical_step, 0, 0])
                coords[i] = coords[prev_valid] + direction
            elif next_valid >= 0:
                # Étendre depuis le suivant
                if next_valid < n-1 and not np.isnan(coords[next_valid+1, 0]):
                    direction = coords[next_valid] - coords[next_valid+1]
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm * typical_step
                    else:
                        direction = np.array([-typical_step, 0, 0])
                else:
                    direction = np.array([-typical_step, 0, 0])
                coords[i] = coords[next_valid] + direction
            else:
                # Aucun voisin valide - position par défaut
                coords[i] = np.array([i * typical_step, 0, 0])

    # Remplacer tout NaN restant
    coords = np.nan_to_num(coords, nan=0.0)

    return coords

print("✅ Adaptation template définie")

# %% [markdown]
# ## 🔧 7. Contraintes géométriques

# %%
def apply_constraints(coords, sequence, confidence=1.0):
    """
    Applique des contraintes géométriques RNA.
    """
    n = len(coords)
    if n <= 1:
        return coords

    refined = coords.copy()

    # Force des contraintes (inverse de confiance)
    strength = 0.5 * (1.0 - min(confidence, 0.9))

    # 1. Contraintes de distance séquentielle (5.0-7.0Å)
    for _ in range(3):  # Quelques itérations
        for i in range(n - 1):
            dist = np.linalg.norm(refined[i+1] - refined[i])

            if dist < 0.1:
                # Éviter division par zéro
                refined[i+1] = refined[i] + np.array([5.9, 0, 0])
                dist = 5.9

            if dist < 5.0 or dist > 7.0:
                target = 5.9
                direction = (refined[i+1] - refined[i]) / dist
                adjustment = (target - dist) * strength
                refined[i+1] = refined[i] + direction * (dist + adjustment)

    # 2. Évitement des clashes (min 3.5Å pour non-séquentiels)
    for i in range(n):
        for j in range(i + 2, n):
            diff = refined[j] - refined[i]
            dist = np.linalg.norm(diff)

            if dist < 3.5 and dist > 0.1:
                direction = diff / dist
                push = (3.5 - dist) * 0.5
                refined[i] -= direction * push
                refined[j] += direction * push

    return refined

print("✅ Contraintes géométriques définies")

# %% [markdown]
# ## 🧬 8. Génération de novo

# %%
def generate_helix(sequence, seed=None):
    """Génère une structure hélicale simple."""
    if seed is not None:
        np.random.seed(seed)

    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))

    coords = np.zeros((n, 3))
    radius = 10.0
    rise = 2.8
    angle_per_residue = 0.6

    for i in range(n):
        theta = i * angle_per_residue
        coords[i] = [
            radius * np.cos(theta),
            radius * np.sin(theta),
            i * rise
        ]

    # Ajouter un peu de bruit
    coords += np.random.normal(0, 0.5, coords.shape)

    return coords

def generate_with_basepairing(sequence, seed=None):
    """Génère une structure avec base-pairing."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))

    coords = np.zeros((n, 3))

    # Initialisation
    for i in range(min(3, n)):
        theta = i * 0.6
        coords[i] = [10.0 * np.cos(theta), 10.0 * np.sin(theta), i * 2.8]

    if n <= 3:
        return coords

    direction = np.array([0.0, 0.0, 1.0])
    complement = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}

    for i in range(3, n):
        base = sequence[i]

        # Chercher partenaire
        partner_idx = -1
        for j in range(max(0, i-12), i):
            if sequence[j] == complement.get(base, 'X'):
                partner_idx = j
                break

        if partner_idx >= 0 and random.random() < 0.6:
            # Position près du partenaire
            partner_pos = coords[partner_idx]
            bp_dist = 10.0 + random.uniform(-1.5, 1.5)

            # Direction vers le centre
            center = np.mean(coords[:i], axis=0)
            to_center = center - partner_pos
            norm = np.linalg.norm(to_center)
            if norm > 0:
                to_center = to_center / norm
            else:
                to_center = np.array([1, 0, 0])

            coords[i] = partner_pos + to_center * bp_dist
            coords[i] += np.random.normal(0, 1.5, 3)
        else:
            # Continuer la chaîne
            if random.random() < 0.25:
                # Changement de direction
                angle = random.uniform(0.2, 0.5)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rot = R.from_rotvec(angle * axis)
                direction = rot.apply(direction)

            step = random.uniform(4.0, 6.0)
            coords[i] = coords[i-1] + step * direction
            direction += np.random.normal(0, 0.1, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

    return coords

print("✅ Génération de novo définie")

# %% [markdown]
# ## 🎯 9. Pipeline de prédiction

# %%
def predict_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_preds=5, temporal_cutoff=None):
    """
    Prédit 5 structures diversifiées.
    """
    predictions = []
    seq_len = len(sequence)

    # Edge case: séquence vide
    if seq_len == 0:
        return [np.zeros((0, 3)) for _ in range(n_preds)]

    # Chercher des templates
    try:
        similar = find_similar_sequences(
            sequence, train_seqs_df, train_coords_dict,
            temporal_cutoff, top_n=n_preds
        )
    except Exception as e:
        print(f"   ⚠️ Erreur recherche templates: {e}")
        similar = []

    # Utiliser les templates trouvés
    for template in similar:
        try:
            adapted = adapt_template_to_query(sequence, template)

            if adapted is not None and len(adapted) == seq_len:
                confidence = template['similarity']
                refined = apply_constraints(adapted, sequence, confidence)

                # Bruit selon confiance
                noise_scale = max(0.1, 0.8 - confidence)
                noisy = refined + np.random.normal(0, noise_scale, refined.shape)

                predictions.append(noisy)
        except Exception as e:
            print(f"   ⚠️ Erreur adaptation: {e}")
            continue

        if len(predictions) >= n_preds:
            break

    # Compléter avec de novo
    base_seed = hash(target_id) % 100000

    while len(predictions) < n_preds:
        seed = base_seed + len(predictions) * 1234

        if len(predictions) % 2 == 0:
            de_novo = generate_helix(sequence, seed)
        else:
            de_novo = generate_with_basepairing(sequence, seed)

        refined = apply_constraints(de_novo, sequence, confidence=0.3)
        predictions.append(refined)

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

        # Affichage progression
        if idx % 10 == 0:
            elapsed = time.time() - start
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / speed if speed > 0 else 0
            print(f"   [{idx+1}/{total}] {target_id[:20]:20s} ({len(sequence):4d} nt) | {elapsed:6.1f}s | ETA: {eta:.0f}s", flush=True)

        # Prédire 5 structures
        try:
            preds = predict_structures(
                sequence, target_id, train_seqs_df, train_coords_dict,
                n_preds=5, temporal_cutoff=temporal
            )
        except Exception as e:
            print(f"   ❌ Erreur {target_id}: {e}")
            # Fallback: hélice simple
            preds = [generate_helix(sequence, seed=i) for i in range(5)]

        # Vérifier dimensions
        for p_idx in range(5):
            if len(preds[p_idx]) != len(sequence):
                preds[p_idx] = generate_helix(sequence, seed=p_idx)

        # Créer les lignes
        for j in range(len(sequence)):
            row_data = {
                'ID': f"{target_id}_{j+1}",
                'resname': sequence[j],
                'resid': j + 1
            }

            for p_idx in range(5):
                x, y, z = preds[p_idx][j]
                row_data[f'x_{p_idx+1}'] = round(float(x), 3)
                row_data[f'y_{p_idx+1}'] = round(float(y), 3)
                row_data[f'z_{p_idx+1}'] = round(float(z), 3)

            all_rows.append(row_data)

    # Créer DataFrame
    df = pd.DataFrame(all_rows)
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1, 6) for c in ['x', 'y', 'z']]
    df = df[cols]

    df.to_csv(output, index=False)

    total_time = time.time() - start
    print("=" * 50)
    print(f"✅ Soumission: {output}")
    print(f"   {total} séquences en {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 50)

    return df

# %% [markdown]
# ## 🚀 11. Exécution

# %%
print("=" * 60)
print("     🧬 RNA 3D FOLDING - PIPELINE v3")
print("=" * 60)
print(f"""
📋 CONFIGURATION:
   • Train: {len(train_seqs)} séquences
   • Templates: {len(train_coords)} structures
   • Test: {len(test_seqs)} séquences

🔧 CARACTÉRISTIQUES:
   • Alignement Needleman-Wunsch custom
   • Pré-filtrage k-mer rapide
   • AUCUNE dépendance externe (pas de BioPython)
   • Gestion robuste des edge cases

🚀 Génération en cours...
""")
print("=" * 60)

# %%
# Génération automatique
submission = generate_submission(test_seqs, train_seqs, train_coords, 'submission.csv')
