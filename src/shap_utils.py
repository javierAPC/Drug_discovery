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

# MACCS key definitions (selected relevant subset)
# Full list: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
MACCS_DESCRIPTIONS = {
    1: 'isotope',
    2: 'unsaturated heterocycle',
    3: 'O in ring',
    5: 'N-N bond',
    8: 'quaternary nitrogen',
    11: 'epoxide',
    14: 'fused aromatic ring',
    15: 'N-OH',
    16: 'ArN',
    22: 'S in ring',
    24: 'SH group',
    25: 'N in non-aromatic ring',
    29: 'SS',
    32: 'S=O',
    34: 'CH2=A',
    35: 'heteroatom in ring',
    38: 'NC(C)N',
    39: 'N-C=O amide',
    42: 'C=S',
    44: 'Cl',
    52: 'NN',
    53: ' Two heteroatoms with H, separated by 4 atoms',
    62: 'ring assembly',
    65: 'multiple bonds in ring',
    67: 'nitrile C#N',
    70: 'Heteroatom–N–heteroatom',
    71: 'nitrogen heterocycle',
    74: 'O adjacent to N (N-O)',
    75: 'N with exocyclic bond',
    77: 'sulfonamide SO2N',
    82: 'N in aromatic ring',
    83: 'F',
    84: 'NH2',
    86: 'ether C-O-C',
    92: 'Br',
    93: 'ketone C=O',
    95: 'N–A–A–O fragment',
    98: '6‑membered heterocycles',
    99: 'aliphatic OH',
    100: 'O-O peroxide',
    101: 'aromatic amine ArNH2',
    103: 'CL',
    104: 'aliphatic amine',
    106: 'ester C(=O)O',
    107: 'NH2',
    110: 'NH',
    111: 'H-bond donor (OH or NH)',
    114: 'carbonyl C=O',
    115: 'ring',
    116: 'aromatic ring',
    117: 'H-bond acceptor (N or O)',
    118: 'ring system',
    119: 'ring size > 4',
    120: 'ring size > 5',
    121: 'ring size > 6',
    122: 'ring size > 7',
    123: 'ring size > 8',
    124: 'ring size > 9',
    125: 'ring size > 10',
    126: 'ring size > 11',
    127: 'ring size > 12',
    128: 'ring size > 13',
    130: 'aromatic N',
    131: 'H‑bond donors (OH/NH) count >1',
    132: 'aromatic O',
    133: 'tertiary amine',
    135: 'CH2 aliphatic',
    138: 'Heteroatom–CH₂– group count >1',
    139: 'C=C',
    141: 'H-bond acceptor count > 1',
    142: 'H-bond donor count > 1',
    145: 'quaternary C',
    148: 'two aromatic rings',
    149: 'three aromatic rings',
    150: 'C with 4 bonds to C',
    153: 'C=C-C=O conjugated',
    154: 'acrylamide C=C-C=O-N (EGFR covalent warhead!)',
    155: 'aromatic N-heterocycle',
    156: 'N adjacent to ring',
    160: 'N in 6-membered ring',
    161: 'N in ring, not 5 or 6',
    162: 'O in ring',
    163: 'any ring',
    164: 'C in ring',
    165: 'multiple rings',
    166: 'atom count > 11',
}


def maccs_col_to_bit(col_name):
    """Convert column name like 'maccs_163' to integer bit 163."""
    return int(col_name.split('_')[1])


def get_maccs_label(col_name):
    bit = maccs_col_to_bit(col_name)
    desc = MACCS_DESCRIPTIONS.get(bit, f'bit_{bit}')
    return f'MACCS {bit}: {desc}'


def get_morgan_weights(mol, model, radius=3, n_bits=2048):
    """
    Compute per-atom Morgan fingerprint contribution weights
    using the model's feature importances.
    Returns a dict {atom_idx: weight}.
    """
    # Get bit info: which atoms contributed to which bits
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )

    # Feature importances from the RF/XGB model (trained on ECFP6)
    # We use the Part 5 model (ECFP6) for this plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # SVR Pipeline — use uniform weights
        importances = np.ones(n_bits) / n_bits

    atom_weights = {}
    for bit, environments in bit_info.items():
        if bit < len(importances):
            weight = float(importances[bit])
            for (center_atom, _radius) in environments:
                atom_weights[center_atom] = atom_weights.get(
                    center_atom, 0) + weight

    return atom_weights
