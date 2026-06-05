import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
try:
    from scipy import stats
    from scipy.stats import mannwhitneyu
except ImportError:
    pass
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, DataStructs, Lipinski
except ImportError:
    pass
try:
    from sklearn.model_selection import KFold
except ImportError:
    pass
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch import nn
    import pytorch_lightning as pl
except ImportError:
    pass


def identify_duplicates(df, smiles_column):
    """
    Standardises SMILES, computes InChI keys, and flags duplicate molecules.
    Returns an annotated copy of df with columns:
        canonical_smiles, inchi_key, is_valid, is_duplicate,
        duplicate_group, occurrence_count
    """

    def standardize_smiles(smi):
        """Re-canonicalise a SMILES string using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except Exception:
            return None

    def calculate_inchi_key(smi):
        """Convert SMILES to an InChI key (molecule fingerprint, writing-invariant)."""
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToInchiKey(mol) if mol else None
        except Exception:
            return None

    out = df.copy()
    out['canonical_smiles'] = out[smiles_column].apply(standardize_smiles)
    out['inchi_key'] = out['canonical_smiles'].apply(calculate_inchi_key)
    out['is_valid'] = out['canonical_smiles'].notna()
    out['is_duplicate'] = out['inchi_key'].duplicated(keep='first')
    out['duplicate_group'] = out.groupby('inchi_key').ngroup()
    counts = out['inchi_key'].value_counts()
    out['occurrence_count'] = out['inchi_key'].map(counts)
    return out


def get_duplicate_summary(df_analysis):
    """Print a summary of the duplicate analysis."""
    summary = {
        'total_records': len(df_analysis),
        'invalid_smiles': (~df_analysis['is_valid']).sum(),
        'unique_molecules': df_analysis['inchi_key'].nunique(),
        'duplicate_records_removed': df_analysis['is_duplicate'].sum(),
        'molecules_with_duplicates': (df_analysis['occurrence_count'] > 1).sum(),
        'max_duplicates_one_mol': int(df_analysis['occurrence_count'].max()),
    }
    return summary


def process_duplicates(df, inchi_key_col, pchembl_col):
    """
    Collapses duplicate molecules to a single row per InChI key.
    - zero_sd  (all IC50 values agree) : keep first row as-is
    - nonzero_sd (IC50 values differ)  : keep first row, replace pChEMBL with group mean
    - single_entry                     : no change
    Returns (final_df, summary_df).
    """
    df_work = df.copy()

    # Group statistics per unique molecule
    stats = (
        df_work.groupby(inchi_key_col)[pchembl_col]
        .agg(['count', 'mean', 'std'])
        .round(4)
        .reset_index()
    )
    stats.columns = [inchi_key_col, 'count', 'mean', 'std']

    processed = []

    for _, row in stats.iterrows():
        group = df_work[df_work[inchi_key_col] == row[inchi_key_col]].copy()
        rep = group.iloc[[0]].copy()   # representative row

        if row['count'] == 1:
            rep['group_type'] = 'single_entry'
        elif row['std'] < 0.01:          # identical measurements
            rep['group_type'] = 'zero_sd'
        else:                            # variable measurements → use mean
            rep[pchembl_col] = row['mean']
            rep['group_type'] = 'nonzero_sd'

        processed.append(rep)

    final = pd.concat(processed).sort_values(
        inchi_key_col).reset_index(drop=True)

    summary = pd.DataFrame([
        ('initial_records',          len(df_work)),
        ('unique_molecules',         len(stats)),
        ('single_entry',             (stats['count'] == 1).sum()),
        ('duplicate_groups',         (stats['count'] > 1).sum()),
        ('  → zero_sd (identical)',
         (stats.query('count > 1')['std'] < 0.01).sum()),
        ('  → nonzero_sd (averaged)',
         (stats.query('count > 1')['std'] >= 0.01).sum()),
        ('final_records',            len(final)),
    ], columns=['step', 'count']).set_index('step')

    return final, summary


def assign_class(pchembl):
    if pchembl >= 6.0:
        return 'active'
    elif pchembl <= 5.0:
        return 'inactive'
    else:
        return 'intermediate'


# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation
def lipinski(smiles, verbose=False):

    mol_data = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        mol_data.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in mol_data:

        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
        desc_TPSA = Descriptors.TPSA(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors,
                        desc_TPSA,
                        desc_NumRotatableBonds])

        if (i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i+1

    columnNames = ["MW", "LogP", "NumHDonors",
                   "NumHAcceptors", "TPSA", "NumRotatableBonds"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors


def qqplot_with_bands(data, title, ax):
    probplot = sm.ProbPlot(data, dist=stats.norm, fit=True)
    probplot.qqplot(line='45', ax=ax, marker='o', markerfacecolor='blue',
                    markeredgecolor='black', markersize=6)

    n = len(data)
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha/2)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
    se = 1 / (np.sqrt(n) * stats.norm.pdf(theoretical_quantiles))
    lower = theoretical_quantiles - z * se
    upper = theoretical_quantiles + z * se

    ax.fill_between(theoretical_quantiles, lower, upper,
                    color='gray', alpha=0.2, label='95% CI')
    ax.plot(theoretical_quantiles, lower, '--', color='gray', linewidth=0.8)
    ax.plot(theoretical_quantiles, upper, '--', color='gray', linewidth=0.8)
    ax.set_xlim(-4, 4)  # Limitar el eje X a [-4, 4]
    ax.set_title(title)
    ax.legend()


def interpret_effect_size(r):
    r_abs = abs(r)
    if r_abs < 0.1:
        return "negligible"
    elif r_abs < 0.3:
        return "small"
    elif r_abs < 0.5:
        return "moderate"
    else:
        return "large"


def statistical_comparison(descriptor, df, group_col="bioactivity_class",
                           group1="active", group2="inactive",
                           n_bootstrap=1000, random_state=42,
                           verbose=True):

    rng = np.random.default_rng(random_state)

    g1 = df[df[group_col] == group1][descriptor].dropna().values
    g2 = df[df[group_col] == group2][descriptor].dropna().values

    n1, n2 = len(g1), len(g2)
    n_total = n1 + n2

    # Mann–Whitney U
    u_stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")

    # Rank-biserial effect size. r > 0 → g1 > g2
    r_rb = (2 * u_stat) / (n1 * n2) - 1

    # z-score
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n_total + 1) / 12)
    z = (u_stat - mean_u) / std_u

    # Median difference
    med_diff = np.median(g1) - np.median(g2)

    # Bootstrap CI
    boot_diffs = []
    for _ in range(n_bootstrap):
        samp1 = rng.choice(g1, size=n1, replace=True)
        samp2 = rng.choice(g2, size=n2, replace=True)
        boot_diffs.append(np.median(samp1) - np.median(samp2))

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # Overlap
    q1, q3 = np.percentile(g1, [25, 75])
    overlap = np.mean((g2 >= q1) & (g2 <= q3)) * 100

    # -------- INTERPRETATION --------
    effect_label = interpret_effect_size(r_rb)

    if ci_low > 0:
        ci_interpretation = "difference is consistently positive (group1 > group2)"
    elif ci_high < 0:
        ci_interpretation = "difference is consistently negative (group1 < group2)"
    else:
        ci_interpretation = "difference is uncertain (CI includes 0)"

    if med_diff > 0:
        direction = f"{group1} tends to have higher values"
    elif med_diff < 0:
        direction = f"{group2} tends to have higher values"
    else:
        direction = "no clear directional difference"

    if overlap < 25:
        overlap_label = "low overlap (good separation)"
    elif overlap < 50:
        overlap_label = "moderate overlap"
    else:
        overlap_label = "high overlap (poor separation)"

    report = (
        f"{descriptor}: {group1} vs {group2} | "
        f"Effect size r = {r_rb:.3f} ({effect_label}). "
        f"Median difference = {med_diff:.3f} "
        f"(95% CI [{ci_low:.3f}, {ci_high:.3f}] → {ci_interpretation}). "
        f"{direction}. "
        f"Overlap = {overlap:.1f}% ({overlap_label})."
    )

    if verbose:
        print("\n" + report)

    return {
        "descriptor": descriptor,
        "group1": group1,
        "group2": group2,
        "n_group1": n1,
        "n_group2": n2,
        "u_stat": u_stat,
        "p_value": p_value,
        "z_score": z,
        "effect_size_r_rb": r_rb,
        "effect_magnitude": effect_label,
        "median_diff": med_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "ci_interpretation": ci_interpretation,
        "direction": direction,
        "overlap_pct": overlap,
        "overlap_interpretation": overlap_label,
        "report": report
    }


def plot_boxplot_by_class(descriptor, df, output_dir="/workspaces/Drug_discovery/figs/eda/", show=True):
    """
    Generate a boxplot, shows it in the notebook (show=True) and saves it.
    """
    plt.figure(figsize=(5.5, 5.5))

    sns.boxplot(
        data=df,
        x='bioactivity_class',
        y=descriptor,
        hue='bioactivity_class',
        palette='colorblind',
        legend=False
    )

    plt.xlabel("Bioactivity class", fontsize=14, fontweight="bold")
    plt.ylabel(descriptor, fontsize=14, fontweight="bold")
    plt.tight_layout()

    # save
    filename = f"{output_dir}plot_{descriptor}.pdf"
    plt.savefig(filename)

    # show
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Boxplot save in: {filename}")
