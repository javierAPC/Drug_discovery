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


def leverage(Z_row, ZtZ_inv):
    return float(Z_row @ ZtZ_inv @ Z_row)
