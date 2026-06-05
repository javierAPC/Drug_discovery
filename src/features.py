import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, DataStructs
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

def smiles_to_mol(smi):
    """Parse SMILES → RDKit Mol. Returns None for invalid SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol  # already None if invalid
    except Exception:
        return None

def compute_ecfp6(mol_series, n_bits=2048):
    """
    Compute ECFP6 (Morgan radius=3) fingerprints.
    Returns a DataFrame with integer bit columns.
    """
    rows = []
    for mol in mol_series:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=3, nBits=n_bits
        )
        rows.append(list(fp))

    cols = [f'ecfp6_{i}' for i in range(n_bits)]
    return pd.DataFrame(rows, columns=cols, dtype=int)

def compute_maccs(mol_series):
    """
    Compute MACCS Keys (167 bits, each = a defined SMARTS substructure).
    Returns a DataFrame with integer bit columns.
    """
    rows = []
    for mol in mol_series:
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows.append(list(fp))

    # MACCS keys are 1-indexed (bit 0 is unused), keep all 167
    cols = [f'maccs_{i}' for i in range(167)]
    return pd.DataFrame(rows, columns=cols, dtype=int)

def compute_rdkit2d(mol_series):
    """
    Compute all 2D RDKit descriptors (continuous values).
    Returns a DataFrame. Columns correspond to descriptor names.
    """
    # Get all descriptor names and their functions once
    descriptor_fns = Descriptors.descList  # list of (name, function) tuples

    rows = []
    for mol in mol_series:
        values = []
        for name, fn in descriptor_fns:
            try:
                values.append(fn(mol))
            except Exception:
                values.append(np.nan)
        rows.append(values)

    cols = [name for name, _ in descriptor_fns]
    return pd.DataFrame(rows, columns=cols, dtype=float)

def add_id(X, ids):
    return pd.concat([ids, X.reset_index(drop=True)], axis=1)
