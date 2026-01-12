# %% [markdown]
# # 🧬 RNA 3D Folding - Pipeline Amélioré v2
#
# Approche Template-Based optimisée avec:
# - Alignement BioPython (pairwise2)
# - Adaptation template→query avec gestion des gaps
# - Contraintes adaptatives selon la confiance
# - Structure de novo avec base-pairing

# %% [markdown]
# ## 📦 1. Imports

# %%
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import random
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# BioPython pour alignement
from Bio import pairwise2
from Bio.Seq import Seq

DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
print("✅ Imports OK")

# %% [markdown]
# ## 📊 2. Chargement des données

# %%
print("📂 Chargement des données...")
train_seqs = pd.read_csv(DATA_PATH / 'train_sequences.csv')
valid_seqs = pd.read_csv(DATA_PATH / 'validation_sequences.csv')
test_seqs = pd.read_csv(DATA_PATH / 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')
valid_labels = pd.read_csv(DATA_PATH / 'validation_labels.csv')

print(f"✅ Train: {len(train_seqs)}, Valid: {len(valid_seqs)}, Test: {len(test_seqs)}")

# %% [markdown]
# ## 🔧 3. Traitement des labels

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
# ## 🔍 4. Recherche de templates similaires

# %%
def find_similar_sequences(query_seq, train_seqs_df, train_coords_dict, temporal_cutoff=None, top_n=5):
    """
    Trouve les séquences similaires avec BioPython pairwise2.
    """
    similar_seqs = []
    query_seq_obj = Seq(query_seq)

    # Filtrer par temporal_cutoff si fourni
    if temporal_cutoff and 'temporal_cutoff' in train_seqs_df.columns:
        filtered_df = train_seqs_df[train_seqs_df['temporal_cutoff'] < temporal_cutoff]
    else:
        filtered_df = train_seqs_df

    for _, row in filtered_df.iterrows():
        target_id = row['target_id']
        train_seq = row['sequence']

        if target_id not in train_coords_dict:
            continue

        # Skip si longueurs trop différentes (>50%)
        len_diff = abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq))
        if len_diff > 0.5:
            continue

        # Alignement BioPython
        alignments = pairwise2.align.globalms(
            query_seq_obj, train_seq,
            2, -1, -10, -0.5,  # match, mismatch, gap_open, gap_extend
            one_alignment_only=True
        )

        if alignments:
            alignment = alignments[0]
            similarity = alignment.score / (2 * min(len(query_seq), len(train_seq)))
            similar_seqs.append((target_id, train_seq, similarity, train_coords_dict[target_id], alignment))

    similar_seqs.sort(key=lambda x: x[2], reverse=True)
    return similar_seqs[:top_n]

print("✅ Fonction find_similar_sequences définie")

# %% [markdown]
# ## 🎯 5. Adaptation template → query

# %%
def adapt_template_to_query(query_seq, template_seq, template_coords, alignment=None):
    """
    Adapte les coordonnées du template à la séquence query.
    Gestion robuste des gaps et valeurs NaN.
    """
    if alignment is None:
        alignments = pairwise2.align.globalms(
            Seq(query_seq), template_seq,
            2, -1, -10, -0.5,
            one_alignment_only=True
        )
        if not alignments:
            return generate_helix_structure(query_seq)
        alignment = alignments[0]

    aligned_query = alignment.seqA
    aligned_template = alignment.seqB

    # Initialiser avec NaN
    query_coords = np.full((len(query_seq), 3), np.nan)

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

    # Interpoler les NaN
    query_coords = interpolate_missing(query_coords)

    return query_coords

def interpolate_missing(coords):
    """Interpole les positions manquantes (NaN)."""
    n = len(coords)
    typical_step = 5.9

    for i in range(n):
        if np.isnan(coords[i, 0]):
            # Trouver les voisins valides
            prev_valid = next((j for j in range(i-1, -1, -1) if not np.isnan(coords[j, 0])), -1)
            next_valid = next((j for j in range(i+1, n) if not np.isnan(coords[j, 0])), -1)

            if prev_valid >= 0 and next_valid >= 0:
                # Interpolation linéaire
                t = (i - prev_valid) / (next_valid - prev_valid)
                coords[i] = (1 - t) * coords[prev_valid] + t * coords[next_valid]
            elif prev_valid >= 0:
                # Étendre depuis le précédent
                if prev_valid > 0 and not np.isnan(coords[prev_valid-1, 0]):
                    direction = coords[prev_valid] - coords[prev_valid-1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10) * typical_step
                else:
                    direction = np.array([typical_step, 0, 0])
                coords[i] = coords[prev_valid] + direction
            elif next_valid >= 0:
                # Étendre depuis le suivant
                if next_valid < n-1 and not np.isnan(coords[next_valid+1, 0]):
                    direction = coords[next_valid] - coords[next_valid+1]
                    direction = direction / (np.linalg.norm(direction) + 1e-10) * typical_step
                else:
                    direction = np.array([-typical_step, 0, 0])
                coords[i] = coords[next_valid] + direction
            else:
                # Aucun voisin valide - générer position
                coords[i] = np.array([i * typical_step, 0, 0])

    return np.nan_to_num(coords)

def generate_helix_structure(sequence):
    """Génère une structure hélicale simple."""
    n = len(sequence)
    coords = np.zeros((n, 3))
    radius, rise, angle = 10.0, 2.5, 0.6

    for i in range(n):
        a = i * angle
        coords[i] = [radius * np.cos(a), radius * np.sin(a), i * rise]

    return coords

print("✅ Fonctions d'adaptation définies")

# %% [markdown]
# ## 🔧 6. Contraintes adaptatives

# %%
def adaptive_constraints(coords, sequence, confidence=1.0):
    """
    Applique des contraintes géométriques RNA adaptatives.
    Plus la confiance est haute, moins on contraint.
    """
    refined = coords.copy()
    n = len(sequence)

    # Force des contraintes (inverse de confiance)
    strength = 0.8 * (1.0 - min(confidence, 0.8))

    # 1. Contraintes de distance séquentielle (5.5-6.5Å)
    seq_min, seq_max = 5.5, 6.5

    for i in range(n - 1):
        dist = np.linalg.norm(refined[i+1] - refined[i])

        if dist < seq_min or dist > seq_max:
            target = (seq_min + seq_max) / 2
            direction = refined[i+1] - refined[i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            adjustment = (target - dist) * strength
            refined[i+1] = refined[i] + direction * (dist + adjustment)

    # 2. Évitement des clashes stériques (min 3.8Å)
    min_dist = 3.8
    dist_mat = distance_matrix(refined, refined)

    clashes = np.where((dist_mat < min_dist) & (dist_mat > 0))

    for idx in range(len(clashes[0])):
        i, j = clashes[0][idx], clashes[1][idx]

        if abs(i - j) <= 1 or i >= j:
            continue

        current_dist = dist_mat[i, j]
        direction = refined[j] - refined[i]
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        adjustment = (min_dist - current_dist) * strength
        refined[i] -= direction * (adjustment / 2)
        refined[j] += direction * (adjustment / 2)

    # 3. Contraintes de base-pairing (si confiance faible)
    if strength > 0.3:
        pairs = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

        for i in range(n):
            complement = pairs.get(sequence[i])
            if not complement:
                continue

            for j in range(i + 3, min(i + 20, n)):
                if sequence[j] == complement:
                    dist = np.linalg.norm(refined[i] - refined[j])

                    if 8.0 < dist < 14.0:
                        target = 10.5  # Distance C1'-C1' typique
                        adjustment = (target - dist) * (strength * 0.3)

                        direction = refined[j] - refined[i]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)

                        refined[i] -= direction * (adjustment / 2)
                        refined[j] += direction * (adjustment / 2)
                        break

    return refined

print("✅ Contraintes adaptatives définies")

# %% [markdown]
# ## 🧬 7. Génération de novo

# %%
def generate_rna_structure(sequence, seed=None):
    """
    Génère une structure RNA réaliste avec base-pairing.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = len(sequence)
    coords = np.zeros((n, 3))

    # Initialiser les premiers résidus
    for i in range(min(3, n)):
        angle = i * 0.6
        coords[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]

    direction = np.array([0.0, 0.0, 1.0])
    complementary = {'G': 'C', 'C': 'G', 'A': 'U', 'U': 'A'}

    for i in range(3, n):
        base = sequence[i]
        has_pair = False
        pair_idx = -1

        # Chercher un partenaire potentiel
        window = min(i, 15)
        for j in range(i - window, i):
            if j >= 0 and sequence[j] == complementary.get(base, 'X'):
                has_pair = True
                pair_idx = j
                break

        if has_pair and i - pair_idx <= 10 and random.random() < 0.7:
            # Positionner près du partenaire
            pair_pos = coords[pair_idx]
            center = np.mean(coords[:i], axis=0)
            dir_to_center = center - pair_pos
            dir_to_center = dir_to_center / (np.linalg.norm(dir_to_center) + 1e-10)

            bp_dist = 10.0 + random.uniform(-1.0, 1.0)
            offset = np.random.normal(0, 1, 3) * 2.0
            coords[i] = pair_pos + dir_to_center * bp_dist + offset

            direction = np.random.normal(0, 0.3, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
        else:
            # Continuer la chaîne
            if random.random() < 0.3:
                angle = random.uniform(0.2, 0.6)
                axis = np.random.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                rot = R.from_rotvec(angle * axis)
                direction = rot.apply(direction)
            else:
                direction += np.random.normal(0, 0.15, 3)
                direction = direction / (np.linalg.norm(direction) + 1e-10)

            step = random.uniform(3.5, 4.5)
            coords[i] = coords[i-1] + step * direction

    return coords

print("✅ Génération de novo définie")

# %% [markdown]
# ## 🎯 8. Pipeline de prédiction

# %%
def predict_structures(sequence, target_id, train_seqs_df, train_coords_dict, n_preds=5, temporal_cutoff=None):
    """
    Prédit 5 structures diversifiées.
    """
    predictions = []

    # Chercher des templates
    similar = find_similar_sequences(sequence, train_seqs_df, train_coords_dict, temporal_cutoff, top_n=n_preds)

    # Utiliser les templates trouvés
    if similar:
        for template_id, template_seq, similarity, template_coords, alignment in similar:
            adapted = adapt_template_to_query(sequence, template_seq, template_coords, alignment)

            if adapted is not None:
                # Contraintes adaptatives selon la similarité
                refined = adaptive_constraints(adapted, sequence, confidence=similarity)

                # Ajouter du bruit (moins pour les bons templates)
                noise_scale = max(0.05, 0.8 - similarity)
                noisy = refined + np.random.normal(0, noise_scale, refined.shape)

                predictions.append(noisy)

                if len(predictions) >= n_preds:
                    break

    # Compléter avec de novo si nécessaire
    while len(predictions) < n_preds:
        seed = hash(target_id) % 10000 + len(predictions) * 1000
        de_novo = generate_rna_structure(sequence, seed=seed)
        refined = adaptive_constraints(de_novo, sequence, confidence=0.2)
        predictions.append(refined)

    return predictions[:n_preds]

print("✅ Pipeline de prédiction défini")

# %% [markdown]
# ## 📝 9. Génération de la soumission

# %%
def generate_submission(test_df, train_seqs_df, train_coords_dict, output='submission.csv'):
    """Génère le fichier de soumission."""
    print("📝 Génération de la soumission...")
    start = time.time()

    all_rows = []
    total = len(test_df)

    for idx, row in test_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal = row.get('temporal_cutoff', None)

        if idx % 5 == 0:
            elapsed = time.time() - start
            print(f"   {idx+1}/{total}: {target_id} ({len(sequence)} nt) - {elapsed:.1f}s", flush=True)

        # Prédire 5 structures
        preds = predict_structures(sequence, target_id, train_seqs_df, train_coords_dict,
                                   n_preds=5, temporal_cutoff=temporal)

        # Créer les lignes
        for j in range(len(sequence)):
            row_data = {
                'ID': f"{target_id}_{j+1}",
                'resname': sequence[j],
                'resid': j + 1
            }

            for p_idx in range(5):
                x, y, z = preds[p_idx][j]
                row_data[f'x_{p_idx+1}'] = round(x, 3)
                row_data[f'y_{p_idx+1}'] = round(y, 3)
                row_data[f'z_{p_idx+1}'] = round(z, 3)

            all_rows.append(row_data)

    # Créer DataFrame
    df = pd.DataFrame(all_rows)
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1, 6) for c in ['x', 'y', 'z']]
    df = df[cols]

    df.to_csv(output, index=False)

    total_time = time.time() - start
    print(f"✅ Soumission générée: {output}")
    print(f"   {len(test_df)} séquences en {total_time:.1f}s")

    return df

# %% [markdown]
# ## 🚀 10. Exécution

# %%
print("=" * 60)
print("     🧬 RNA 3D FOLDING - PIPELINE AMÉLIORÉ v2")
print("=" * 60)
print(f"""
📋 CONFIGURATION:
   • Train: {len(train_seqs)} séquences
   • Templates: {len(train_coords)} structures
   • Test: {len(test_seqs)} séquences

🔧 AMÉLIORATIONS:
   • Alignement BioPython pairwise2
   • Adaptation template avec gestion gaps
   • Contraintes adaptatives
   • Génération de novo avec base-pairing

🚀 POUR GÉNÉRER:
   submission = generate_submission(test_seqs, train_seqs, train_coords)
""")
print("=" * 60)

# %%
# DÉCOMMENTER POUR GÉNÉRER
# submission = generate_submission(test_seqs, train_seqs, train_coords, 'submission.csv')
