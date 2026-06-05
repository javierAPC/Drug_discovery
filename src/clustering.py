import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.spatial.distance import pdist
from IPython.display import display
from IPython.display import Image,  SVG

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, DataStructs, rdFingerprintGenerator, Draw
    from rdkit.ML.Cluster import Butina
except ImportError:
    pass
try:
    from sklearn.metrics.pairwise import pairwise_distances
except ImportError:
    pass


def get_butina_clusters(mols, sim_cutoff=0.6):
    """
    Cluster molecules by Tanimoto similarity using the Butina algorithm.
    Returns a list of tuples: each tuple contains the indices of one cluster.
    """
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpg.GetFingerprint(mol) for mol in mols]
    dists = pdist(fps, metric='jaccard')        # 1 - Tanimoto
    clusters = Butina.ClusterData(
        dists, len(mols), 1 - sim_cutoff, isDistData=True
    )
    return clusters


def assign_cluster_ids(mols, sim_cutoff=0.6):
    """Return an integer cluster ID for each molecule (same length as mols)."""
    clusters = get_butina_clusters(mols, sim_cutoff)
    cluster_ids = np.zeros(len(mols), dtype=int)
    for cid, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            cluster_ids[idx] = cid
    return cluster_ids


def intra_cluster_sims_by_size(cluster_ids, fps):
    unique_cids = np.unique(cluster_ids)
    size_bins = {'2-5': [], '6-20': [], '>20': []}
    for cid in unique_cids:
        idx = np.where(cluster_ids == cid)[0]
        sz = len(idx)
        if sz < 2:
            continue
        cluster_fps = fps[idx]
        dist_matrix = pairwise_distances(cluster_fps, metric='jaccard')
        sim_matrix = 1 - dist_matrix
        triu = np.triu_indices_from(sim_matrix, k=1)
        sims = sim_matrix[triu].tolist()
        if sz <= 5:
            size_bins['2-5'].extend(sims)
        elif sz <= 20:
            size_bins['6-20'].extend(sims)
        else:
            size_bins['>20'].extend(sims)
    return size_bins


def visualize_cluster(cluster_id, ids, X_fp, y_vals, mols, mol_id_list, top_k=5, save=True, save_root='../figs'):

    idx = np.where(ids == cluster_id)[0]
    if len(idx) == 0:
        print(f'Cluster {cluster_id} empty or dosent exist.')
        return None, None

    n_members = len(idx)
    actual_k = min(top_k, n_members)
    print(f'\n{"="*70}')
    print(f'  Cluster {cluster_id}  ---  {n_members} member(s)')
    print(f'{"="*70}')

    # ---- 1. Most REPRESENTATIVE (highest mean intra-cluster similarity) ----
    if n_members >= 2:
        cluster_fps = X_fp[idx]
        sim_matrix = 1.0 - pairwise_distances(cluster_fps, metric='jaccard')
        mean_sim = sim_matrix.mean(axis=1)
        rep_order = np.argsort(mean_sim)[::-1][:actual_k]
    else:
        mean_sim = np.array([1.0])
        rep_order = np.arange(actual_k)

    rep_idx = idx[rep_order]
    rep_mols = [mols[i] for i in rep_idx]
    rep_legends = [
        f'{mol_id_list[i]}\npIC50={y_vals[i]:.2f}\nSim={mean_sim[rep_order[j]]:.3f}'
        for j, i in enumerate(rep_idx)
    ]

    # Generate SVG for representative
    img_rep = Draw.MolsToGridImage(
        rep_mols,
        legends=rep_legends,
        molsPerRow=min(5, actual_k),
        subImgSize=(300, 250),
        useSVG=True
    )
    # Patch SVG: transparent background and larger font
    svg_rep = img_rep.data.replace(
        "opacity:1.0", "opacity:0.0").replace("12px", "18px")
    display(SVG(svg_rep))

    # ---- 2. Most ACTIVE (highest pchembl_value) ----
    act_vals = y_vals[idx]
    act_order = np.argsort(act_vals)[::-1][:actual_k]
    act_idx = idx[act_order]
    act_mols = [mols[i] for i in act_idx]
    act_legends = [
        f'{mol_id_list[i]}\npIC50={y_vals[i]:.2f}' for i in act_idx]

    img_act = Draw.MolsToGridImage(
        act_mols,
        legends=act_legends,
        molsPerRow=min(5, actual_k),
        subImgSize=(300, 250),
        useSVG=True
    )
    svg_act = img_act.data.replace(
        "opacity:1.0", "opacity:0.0").replace("12px", "18px")
    display(SVG(svg_act))

    # ---- Save if requested ----
    if save:
        cluster_dir = os.path.join(
            save_root, f'../figs/cv/cluster{cluster_id}_centroids')
        os.makedirs(cluster_dir, exist_ok=True)

        # Save representative SVG
        rep_path = os.path.join(
            cluster_dir, f'{actual_k}_most_REPRESENTATIVE.svg')
        with open(rep_path, 'w') as f:
            f.write(svg_rep)

        # Save active SVG
        act_path = os.path.join(cluster_dir, f'{actual_k}_most_ACTIVE.svg')
        with open(act_path, 'w') as f:
            f.write(svg_act)

        print(f'\nSVG images saved in: {cluster_dir}')
        print(f'  - {os.path.basename(rep_path)}')
        print(f'  - {os.path.basename(act_path)}')

    return svg_rep, svg_act
