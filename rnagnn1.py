# %% [markdown]
# # 🧬 RNA 3D Folding - Pipeline Complet TBM + MSA + GNN EBM
#
# **Stanford RNA 3D Folding Part 2 - Kaggle Competition**
#
# Pipeline complet pour prédire la structure 3D de molécules RNA:
# 1. Template-Based Modeling (TBM)
# 2. Multiple Sequence Alignment (MSA)
# 3. Energy-Based Model simple (règles physiques)
# 4. GNN Energy-Based Model (apprentissage)
#
# **Auteur**: Pipeline hybride TBM + GNN

# %% [markdown]
# ## 📦 1. Imports et Configuration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Vérifier PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ PyTorch disponible - Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    print("⚠️ PyTorch non disponible - Mode CPU uniquement")

# Vérifier Plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    print("✅ Plotly disponible")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly non disponible")

# Configuration
DATA_PATH = Path('/kaggle/input/stanford-rna-3d-folding-2')
NUCLEOTIDE_COLORS = {'A': '#FF6B6B', 'U': '#4ECDC4', 'G': '#45B7D1', 'C': '#96CEB4'}

print(f"\n📁 Chemin des données: {DATA_PATH}")

# %% [markdown]
# ## 📊 2. Chargement des Données

# %%
def parse_fasta(fasta_path):
    """Parse un fichier FASTA."""
    sequences = []
    current_header, current_seq = None, []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            sequences.append((current_header, ''.join(current_seq)))
    return sequences

# Charger les données
print("📂 Chargement des données...")
train_sequences = pd.read_csv(DATA_PATH / 'train_sequences.csv')
validation_sequences = pd.read_csv(DATA_PATH / 'validation_sequences.csv')
test_sequences = pd.read_csv(DATA_PATH / 'test_sequences.csv')
train_labels = pd.read_csv(DATA_PATH / 'train_labels.csv')
validation_labels = pd.read_csv(DATA_PATH / 'validation_labels.csv')
sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv')

print(f"✅ Train: {len(train_sequences)} séquences")
print(f"✅ Validation: {len(validation_sequences)} séquences")
print(f"✅ Test: {len(test_sequences)} séquences")

# %% [markdown]
# ## 🔧 3. Parser CIF et Extraction des Templates

# %%
def parse_cif_file(cif_path):
    """Parse un fichier CIF et extrait les données atom_site."""
    atom_data = []
    columns = []
    in_loop = False
    collecting_columns = False

    def parse_cif_line(line):
        tokens, current, in_quote, quote_char = [], '', False, None
        for char in line:
            if char in ('"', "'") and not in_quote:
                in_quote, quote_char = True, char
            elif char == quote_char and in_quote:
                in_quote, quote_char = False, None
            elif char.isspace() and not in_quote:
                if current:
                    tokens.append(current)
                    current = ''
            else:
                current += char
        if current:
            tokens.append(current)
        return tokens

    with open(cif_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line == 'loop_':
                in_loop, collecting_columns, columns = True, True, []
                continue
            if collecting_columns and line.startswith('_atom_site.'):
                columns.append(line.split('.')[1].split()[0])
                continue
            if collecting_columns and not line.startswith('_'):
                collecting_columns = False
                if not columns or not any('atom' in c.lower() or 'Cartn' in c for c in columns):
                    in_loop, columns = False, []
                    continue
            if in_loop and not collecting_columns and line.startswith('_'):
                break
            if in_loop and not collecting_columns and columns:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    tokens = parse_cif_line(line)
                    if len(tokens) >= len(columns):
                        atom_data.append(dict(zip(columns, tokens[:len(columns)])))
    return atom_data

def extract_rna_from_cif(cif_path):
    """Extrait les séquences RNA et coordonnées C1' d'un fichier CIF."""
    try:
        atom_data = parse_cif_file(cif_path)
        if not atom_data:
            return None

        residues, coords = {}, {}
        rna_residues = ['A', 'U', 'G', 'C', 'ADE', 'URA', 'GUA', 'CYT', 'DA', 'DU', 'DG', 'DC', 'RA', 'RU', 'RG', 'RC']
        res_map = {'ADE': 'A', 'URA': 'U', 'GUA': 'G', 'CYT': 'C', 'DA': 'A', 'DU': 'U', 'DG': 'G', 'DC': 'C', 'RA': 'A', 'RU': 'U', 'RG': 'G', 'RC': 'C'}

        for atom in atom_data:
            atom_name = atom.get('label_atom_id', atom.get('auth_atom_id', ''))
            res_name = atom.get('label_comp_id', atom.get('auth_comp_id', ''))
            chain_id = atom.get('label_asym_id', atom.get('auth_asym_id', ''))
            seq_id = atom.get('label_seq_id', atom.get('auth_seq_id', ''))

            if res_name in rna_residues:
                res_short = res_map.get(res_name, res_name)
                key = (chain_id, seq_id)
                if key not in residues:
                    residues[key] = res_short

                atom_clean = atom_name.strip("'\"")
                if atom_clean in ("C1'", "C1*", "C1"):
                    try:
                        coords[key] = (float(atom.get('Cartn_x', 0)), float(atom.get('Cartn_y', 0)), float(atom.get('Cartn_z', 0)))
                    except:
                        pass

        if not residues:
            return None

        chains = {}
        for (chain_id, seq_id), res in sorted(residues.items(), key=lambda x: (x[0][0], int(x[0][1]) if x[0][1].isdigit() else 0)):
            if chain_id not in chains:
                chains[chain_id] = {'sequence': '', 'coords': []}
            chains[chain_id]['sequence'] += res
            chains[chain_id]['coords'].append(coords.get((chain_id, seq_id), (0.0, 0.0, 0.0)))

        return {'pdb_id': Path(cif_path).stem.upper(), 'chains': chains}
    except:
        return None

def build_template_database(pdb_dir, max_files=None):
    """Construit la base de données de templates."""
    templates = []
    cif_files = list(pdb_dir.glob('*.cif'))[:max_files] if max_files else list(pdb_dir.glob('*.cif'))

    print(f"📦 Construction base de templates ({len(cif_files)} fichiers)...")

    for i, cif_file in enumerate(cif_files):
        if i % 500 == 0:
            print(f"   Progression: {i}/{len(cif_files)}")

        result = extract_rna_from_cif(cif_file)
        if result:
            for chain_id, data in result['chains'].items():
                n_valid = sum(1 for c in data['coords'] if c != (0.0, 0.0, 0.0))
                if len(data['sequence']) >= 10 and n_valid >= len(data['sequence']) * 0.5:
                    templates.append({
                        'pdb_id': result['pdb_id'],
                        'chain_id': chain_id,
                        'sequence': data['sequence'],
                        'coords': data['coords'],
                        'length': len(data['sequence'])
                    })

    print(f"✅ {len(templates)} templates extraits")
    return templates

# Construire la base de templates
pdb_dir = DATA_PATH / 'PDB_RNA'
templates_db = build_template_database(pdb_dir, max_files=1000)

if templates_db:
    lengths = [t['length'] for t in templates_db]
    print(f"📊 Longueur: min={min(lengths)}, max={max(lengths)}, moy={np.mean(lengths):.1f}")

# %% [markdown]
# ## 🧬 4. Exploitation des MSA

# %%
def load_msa(msa_path):
    """Charge un fichier MSA."""
    sequences = parse_fasta(msa_path)
    if not sequences:
        return None, []
    return sequences[0], sequences[1:]

def calculate_conservation(msa_sequences):
    """Calcule le score de conservation par position."""
    if not msa_sequences:
        return []
    seq_length = len(msa_sequences[0][1])
    conservation = []
    for pos in range(seq_length):
        nucs = [seq[pos] for _, seq in msa_sequences if pos < len(seq) and seq[pos] not in '-.' ]
        if nucs:
            counts = Counter(nucs)
            conservation.append(max(counts.values()) / len(nucs))
        else:
            conservation.append(0.0)
    return conservation

def find_msa_templates(target_id, msa_dir, templates_db):
    """Trouve des templates via MSA."""
    import re
    msa_file = msa_dir / f"{target_id}.fasta"
    if not msa_file.exists():
        return [], []

    query, homologs = load_msa(msa_file)
    if not query:
        return [], []

    # Extraire PDB IDs des homologues
    template_index = {(t['pdb_id'], t['chain_id']): t for t in templates_db}
    msa_templates = []
    patterns = [r'([0-9][A-Za-z0-9]{3})_([A-Za-z])', r'pdb\|([0-9][A-Za-z0-9]{3})\|([A-Za-z])']

    for header, seq in homologs:
        for pattern in patterns:
            match = re.search(pattern, header)
            if match:
                key = (match.group(1).upper(), match.group(2).upper())
                if key in template_index:
                    msa_templates.append({'template': template_index[key], 'msa_sequence': seq})
                break

    conservation = calculate_conservation([query] + homologs)
    return msa_templates, conservation

print("✅ Fonctions MSA définies")

# %% [markdown]
# ## 🔍 5. Alignement et Recherche de Templates

# %%
def sequence_alignment(seq1, seq2, conservation=None):
    """Alignement de séquences pondéré par conservation."""
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 0.0, []

    if conservation is None:
        conservation = [1.0] * len1

    best_score, best_offset = 0, 0
    for offset in range(-len2 + 1, len1):
        score = 0
        for i in range(max(len1, len2)):
            pos1, pos2 = i, i - offset
            if 0 <= pos1 < len1 and 0 <= pos2 < len2 and seq1[pos1] == seq2[pos2]:
                score += conservation[pos1] if pos1 < len(conservation) else 1.0
        if score > best_score:
            best_score, best_offset = score, offset

    aligned = [(i, i - best_offset) for i in range(max(len1, len2)) if 0 <= i < len1 and 0 <= i - best_offset < len2]
    total_weight = sum(conservation) if conservation else len1
    return best_score / total_weight if total_weight > 0 else 0, aligned

def find_best_templates(query_seq, templates_db, msa_dir=None, target_id=None, top_k=5):
    """Trouve les meilleurs templates."""
    results = []
    msa_templates, conservation = [], None

    if msa_dir and target_id:
        msa_templates, conservation = find_msa_templates(target_id, msa_dir, templates_db)

    # Templates MSA (bonus x1.5)
    for msa_t in msa_templates:
        template = msa_t['template']
        score, aligned = sequence_alignment(query_seq, template['sequence'], conservation)
        results.append({'template': template, 'score': score * 1.5, 'aligned_positions': aligned, 'source': 'msa'})

    # Templates classiques
    seen = {(r['template']['pdb_id'], r['template']['chain_id']) for r in results}
    for template in templates_db:
        if (template['pdb_id'], template['chain_id']) in seen:
            continue
        score, aligned = sequence_alignment(query_seq, template['sequence'], conservation)
        len_ratio = min(len(query_seq), template['length']) / max(len(query_seq), template['length'])
        results.append({'template': template, 'score': score * 0.7 + len_ratio * 0.3, 'aligned_positions': aligned, 'source': 'align'})

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

print("✅ Fonctions d'alignement définies")

# %% [markdown]
# ## 🔋 6. EBM Simple (Règles Physiques)

# %%
def calc_dist(c1, c2):
    """Distance euclidienne."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def energy_bonds(coords, ideal=5.9):
    """Énergie des distances consécutives."""
    e = 0.0
    for i in range(len(coords) - 1):
        if coords[i] != (0, 0, 0) and coords[i + 1] != (0, 0, 0):
            e += (calc_dist(coords[i], coords[i + 1]) - ideal) ** 2
    return e

def energy_clash(coords, min_dist=3.5):
    """Énergie des clashes."""
    e = 0.0
    for i in range(len(coords)):
        if coords[i] == (0, 0, 0):
            continue
        for j in range(i + 3, len(coords)):
            if coords[j] != (0, 0, 0):
                d = calc_dist(coords[i], coords[j])
                if d < min_dist:
                    e += 10 * (min_dist - d) ** 2
    return e

def energy_rg(coords, seq_len):
    """Énergie du rayon de giration."""
    valid = [c for c in coords if c != (0, 0, 0)]
    if len(valid) < 2:
        return 0.0
    cx, cy, cz = np.mean(valid, axis=0)
    rg = math.sqrt(sum((c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2 for c in valid) / len(valid))
    expected_rg = 2.0 * (seq_len ** 0.4)
    return 0.1 * (rg - expected_rg) ** 2

def total_energy(coords, seq_len):
    """Énergie totale."""
    return energy_bonds(coords) + energy_clash(coords) + energy_rg(coords, seq_len)

def score_structure(coords, seq_len):
    """Score une structure (0-1)."""
    e = total_energy(coords, seq_len)
    return 1.0 / (1.0 + e / seq_len)

def refine_structure(coords, n_iter=20):
    """Raffine une structure."""
    refined = [list(c) for c in coords]
    ideal = 5.9

    for _ in range(n_iter):
        for i in range(len(refined) - 1):
            if refined[i] == [0, 0, 0] or refined[i + 1] == [0, 0, 0]:
                continue
            d = calc_dist(refined[i], refined[i + 1])
            if d > 0.1 and abs(d - ideal) > 1.0:
                direction = [(refined[i + 1][k] - refined[i][k]) / d for k in range(3)]
                correction = (d - ideal) * 0.3
                for k in range(3):
                    refined[i + 1][k] -= direction[k] * correction

    return [tuple(c) for c in refined]

print("✅ EBM simple défini")

# %% [markdown]
# ## 🧠 7. GNN Energy-Based Model

# %%
if TORCH_AVAILABLE:
    # Encodage
    NUC_TO_IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}

    def seq_to_onehot(seq):
        oh = np.zeros((len(seq), 5), dtype=np.float32)
        for i, n in enumerate(seq):
            oh[i, NUC_TO_IDX.get(n, 4)] = 1.0
        return oh

    def coords_to_features(coords):
        coords = np.array(coords, dtype=np.float32)
        valid = ~np.all(coords == 0, axis=1)
        centroid = np.mean(coords[valid], axis=0) if np.any(valid) else np.zeros(3)

        features = []
        for i in range(len(coords)):
            if not valid[i]:
                features.append([0.0] * 9)
                continue
            rel = coords[i] - centroid
            dist = np.linalg.norm(rel)
            rel_norm = rel / (dist + 1e-8)
            direction = np.zeros(3)
            if i < len(coords) - 1 and valid[i + 1]:
                d = coords[i + 1] - coords[i]
                direction = d / (np.linalg.norm(d) + 1e-8)
            features.append(list(rel_norm) + [dist / 50] + list(direction) + [i / len(coords), (len(coords) - i) / len(coords)])
        return np.array(features, dtype=np.float32)

    def build_graph(coords, k=5, max_dist=15):
        n = len(coords)
        coords = np.array(coords, dtype=np.float32)
        valid = ~np.all(coords == 0, axis=1)
        edges, edge_feat = [], []

        for i in range(n - 1):
            d = np.linalg.norm(coords[i + 1] - coords[i])
            edges.extend([[i, i + 1], [i + 1, i]])
            edge_feat.extend([[d / 10, 1, 0], [d / 10, 1, 0]])

        for i in range(n):
            if not valid[i]:
                continue
            dists = [np.linalg.norm(coords[i] - coords[j]) if valid[j] and abs(i - j) > 1 else float('inf') for j in range(n)]
            for j in np.argsort(dists)[:k]:
                if dists[j] < max_dist:
                    edges.append([i, j])
                    edge_feat.append([dists[j] / 10, 0, 1 if dists[j] < 8 else 0])

        if not edges:
            edges, edge_feat = [[0, 0]], [[0, 0, 0]]
        return np.array(edges, dtype=np.int64), np.array(edge_feat, dtype=np.float32)

    class GraphConv(nn.Module):
        def __init__(self, in_dim, out_dim, edge_dim=3):
            super().__init__()
            self.w_self = nn.Linear(in_dim, out_dim)
            self.w_neigh = nn.Linear(in_dim, out_dim)
            self.w_edge = nn.Linear(edge_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, x, edges, edge_attr):
            h = self.w_self(x)
            h_n = torch.zeros_like(h)
            for idx, (s, t) in enumerate(edges):
                h_n[int(t)] += self.w_neigh(x[int(s)]) * torch.sigmoid(self.w_edge(edge_attr[idx]))
            return F.relu(self.norm(h + h_n))

    class RNAGNN(nn.Module):
        def __init__(self, node_dim=14, hidden=64, layers=3):
            super().__init__()
            self.proj = nn.Linear(node_dim, hidden)
            self.convs = nn.ModuleList([GraphConv(hidden, hidden) for _ in range(layers)])
            self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, edges, edge_attr):
            h = self.proj(x)
            for conv in self.convs:
                h = h + conv(h, edges, edge_attr)
            return self.out(torch.cat([h.mean(0), h.max(0)[0]]))

    def train_gnn(templates, epochs=30, hidden=32, layers=2):
        print(f"🏋️ Entraînement GNN sur {len(templates)} templates...")
        model = RNAGNN(hidden=hidden, layers=layers).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for ep in range(epochs):
            total_loss, n = 0, 0
            for t in templates:
                seq, coords = t['sequence'], t['coords']
                node_f = torch.tensor(np.concatenate([seq_to_onehot(seq), coords_to_features(coords)], 1), dtype=torch.float32).to(DEVICE)
                edges, edge_f = build_graph(coords)
                edges_t, edge_f_t = torch.tensor(edges).to(DEVICE), torch.tensor(edge_f, dtype=torch.float32).to(DEVICE)

                # Réel
                e_real = model(node_f, edges_t, edge_f_t)
                loss_r = F.binary_cross_entropy(torch.sigmoid(e_real), torch.tensor([[0.0]]).to(DEVICE))

                # Perturbé
                pert = [(c[0] + np.random.normal(0, 5), c[1] + np.random.normal(0, 5), c[2] + np.random.normal(0, 5)) if c != (0, 0, 0) else c for c in coords]
                node_p = torch.tensor(np.concatenate([seq_to_onehot(seq), coords_to_features(pert)], 1), dtype=torch.float32).to(DEVICE)
                edges_p, edge_p = build_graph(pert)
                e_pert = model(node_p, torch.tensor(edges_p).to(DEVICE), torch.tensor(edge_p, dtype=torch.float32).to(DEVICE))
                loss_p = F.binary_cross_entropy(torch.sigmoid(e_pert), torch.tensor([[1.0]]).to(DEVICE))

                loss = loss_r + loss_p
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n += 1

            if (ep + 1) % 10 == 0:
                print(f"   Epoch {ep + 1}/{epochs} - Loss: {total_loss / n:.4f}")

        print("✅ GNN entraîné")
        return model

    def score_gnn(model, seq, coords):
        model.eval()
        with torch.no_grad():
            node_f = torch.tensor(np.concatenate([seq_to_onehot(seq), coords_to_features(coords)], 1), dtype=torch.float32).to(DEVICE)
            edges, edge_f = build_graph(coords)
            e = model(node_f, torch.tensor(edges).to(DEVICE), torch.tensor(edge_f, dtype=torch.float32).to(DEVICE))
            return 1.0 - torch.sigmoid(e).item()

    print("✅ GNN défini")
else:
    print("⚠️ GNN non disponible (PyTorch requis)")

# %% [markdown]
# ## 🎯 8. Prédiction de Structure

# %%
def predict_from_template(query_seq, template_match, noise=0.0):
    """Prédit les coordonnées depuis un template."""
    template = template_match['template']
    aligned = template_match['aligned_positions']
    t_coords = template['coords']

    pred = [(0.0, 0.0, 0.0)] * len(query_seq)
    for q_pos, t_pos in aligned:
        if t_pos < len(t_coords) and t_coords[t_pos] != (0, 0, 0):
            x, y, z = t_coords[t_pos]
            if noise > 0:
                x += np.random.normal(0, noise)
                y += np.random.normal(0, noise)
                z += np.random.normal(0, noise)
            pred[q_pos] = (x, y, z)

    # Interpoler
    valid = [(i, pred[i]) for i in range(len(pred)) if pred[i] != (0, 0, 0)]
    if len(valid) >= 2:
        for i in range(len(pred)):
            if pred[i] == (0, 0, 0):
                before = [(j, c) for j, c in valid if j < i]
                after = [(j, c) for j, c in valid if j > i]
                if before and after:
                    j1, c1 = before[-1]
                    j2, c2 = after[0]
                    t = (i - j1) / (j2 - j1)
                    pred[i] = tuple(c1[k] + t * (c2[k] - c1[k]) for k in range(3))
                elif before:
                    pred[i] = before[-1][1]
                elif after:
                    pred[i] = after[0][1]

    return pred

def generate_predictions(query_seq, templates_db, target_id=None, msa_dir=None, gnn_model=None, n_preds=5):
    """Génère 5 prédictions diversifiées."""
    best = find_best_templates(query_seq, templates_db, msa_dir, target_id, top_k=10)

    if not best:
        return [[(i * 5.9, np.random.normal(0, 1), np.random.normal(0, 1)) for i in range(len(query_seq))] for _ in range(n_preds)]

    preds = []

    # Top 3 templates
    for i in range(min(3, len(best))):
        preds.append(predict_from_template(query_seq, best[i], noise=i * 0.2))

    # Moyenne des 2 meilleurs
    if len(best) >= 2:
        c1 = predict_from_template(query_seq, best[0])
        c2 = predict_from_template(query_seq, best[1])
        avg = [((c1[i][0] + c2[i][0]) / 2, (c1[i][1] + c2[i][1]) / 2, (c1[i][2] + c2[i][2]) / 2) for i in range(len(query_seq))]
        preds.append(avg)
    else:
        preds.append(predict_from_template(query_seq, best[0], noise=0.5))

    # Perturbation
    preds.append(predict_from_template(query_seq, best[0], noise=0.8))

    while len(preds) < n_preds:
        preds.append(predict_from_template(query_seq, best[0], noise=np.random.uniform(0.3, 1.0)))

    preds = preds[:n_preds]

    # Raffiner et scorer
    seq_len = len(query_seq)
    scored = []
    for pred in preds:
        refined = refine_structure(pred)
        if TORCH_AVAILABLE and gnn_model:
            s = score_gnn(gnn_model, query_seq, refined)
        else:
            s = score_structure(refined, seq_len)
        scored.append((s, refined))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]

print("✅ Fonctions de prédiction définies")

# %% [markdown]
# ## 🏋️ 9. Entraînement du GNN

# %%
gnn_model = None

if TORCH_AVAILABLE and templates_db:
    train_templates = templates_db[:min(200, len(templates_db))]
    if train_templates:
        gnn_model = train_gnn(train_templates, epochs=30, hidden=32, layers=2)

        print("\n📊 Test GNN:")
        for t in train_templates[:3]:
            s = score_gnn(gnn_model, t['sequence'], t['coords'])
            print(f"   {t['pdb_id']}_{t['chain_id']}: {s:.4f}")

# %% [markdown]
# ## 📝 10. Génération de la Soumission

# %%
def generate_submission(test_df, templates_db, gnn_model=None, output='submission.csv', use_msa=True):
    """Génère le fichier de soumission."""
    print("📝 Génération de la soumission...")
    msa_dir = DATA_PATH / 'MSA' if use_msa else None

    rows = []
    for idx, row in test_df.iterrows():
        target_id = row['target_id']
        seq = row['sequence']

        preds = generate_predictions(seq, templates_db, target_id, msa_dir, gnn_model)

        for resid, nuc in enumerate(seq, 1):
            r = {'ID': f"{target_id}_{resid}", 'resname': nuc, 'resid': resid}
            for p_idx, coords in enumerate(preds, 1):
                x, y, z = coords[resid - 1] if resid - 1 < len(coords) else (0, 0, 0)
                r[f'x_{p_idx}'] = round(np.clip(x, -999.999, 9999.999), 3)
                r[f'y_{p_idx}'] = round(np.clip(y, -999.999, 9999.999), 3)
                r[f'z_{p_idx}'] = round(np.clip(z, -999.999, 9999.999), 3)
            rows.append(r)

        if idx % 10 == 0:
            print(f"   {idx + 1}/{len(test_df)}")

    df = pd.DataFrame(rows)
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1, 6) for c in ['x', 'y', 'z']]
    df = df[cols]
    df.to_csv(output, index=False)

    print(f"✅ Soumission: {output} ({df.shape})")
    return df

# %% [markdown]
# ## 🚀 11. Exécution

# %%
print("=" * 70)
print("           🧬 RNA 3D FOLDING - PIPELINE COMPLET")
print("=" * 70)
print(f"""
📋 CONFIGURATION:
   • Templates: {len(templates_db)}
   • PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}
   • GNN: {'✅ Entraîné' if gnn_model else '❌ Non disponible'}
   • Device: {DEVICE}

🚀 POUR GÉNÉRER LA SOUMISSION:
   submission = generate_submission(test_sequences, templates_db, gnn_model)
""")
print("=" * 70)

# %%
# DÉCOMMENTER POUR GÉNÉRER LA SOUMISSION
# submission = generate_submission(test_sequences, templates_db, gnn_model, 'submission.csv')
