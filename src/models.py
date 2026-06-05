import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, DataStructs
except ImportError:
    pass
try:
    from sklearn.model_selection import KFold, BaseCrossValidator, cross_validate
    # Sklearn — metrics
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        roc_auc_score, balanced_accuracy_score,
        matthews_corrcoef, f1_score
    )
except ImportError:
    pass
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch import nn
    import pytorch_lightning as pl
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    import torchmetrics
except ImportError:
    pass


def regression_metrics(y_true, y_pred):
    """Compute all regression metrics at once."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Pearson_r': r}


def classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all classification metrics at once.
    y_prob: predicted probability for class=1 (needed for AUC).
    """
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return {'AUC': auc, 'BalancedAcc': ba, 'MCC': mcc, 'MacroF1': f1}


def run_butina_cv_regression(model, X, y, cluster_ids, n_splits=5):
    """
    Run Butina CV for a regression model.
    Returns a DataFrame with per-fold metrics + mean ± std row.
    """
    cv = ButinaCrossValidator(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=cluster_ids), start=1
    ):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        m = regression_metrics(y[test_idx], y_pred)
        m['Fold'] = fold
        fold_results.append(m)

    df_folds = pd.DataFrame(fold_results).set_index('Fold')

    # Summary row
    mean_row = df_folds.mean().rename('mean')
    std_row = df_folds.std().rename('std')
    df_summary = pd.concat(
        [df_folds, mean_row.to_frame().T, std_row.to_frame().T])
    return df_summary


def run_butina_cv_classification(model, X, y, cluster_ids, n_splits=5):
    """
    Run Butina CV for a classification model.
    Returns a DataFrame with per-fold metrics + mean ± std row.
    """
    cv = ButinaCrossValidator(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=cluster_ids), start=1
    ):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        m = classification_metrics(y[test_idx], y_pred, y_prob)
        m['Fold'] = fold
        fold_results.append(m)

    df_folds = pd.DataFrame(fold_results).set_index('Fold')
    mean_row = df_folds.mean().rename('mean')
    std_row = df_folds.std().rename('std')
    df_summary = pd.concat(
        [df_folds, mean_row.to_frame().T, std_row.to_frame().T])
    return df_summary


def regression_metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    return {'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'Pearson_r': r}


def classification_metrics(y_true, y_pred, y_prob=None):
    return {'AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
            'BalancedAcc': balanced_accuracy_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'MacroF1': f1_score(y_true, y_pred, average='macro')}


class ButinaCrossValidator(BaseCrossValidator):
    """
    K-fold cross-validator that splits by Butina chemical cluster.

    Pass pre-computed cluster_ids as `groups` to .split():
        cv.split(X, y, groups=cluster_ids)

    Each fold's test set contains complete clusters not seen in training.
    Clusters are distributed round-robin across folds by size (largest first),
    so folds are approximately balanced by compound count.
    """

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_masks(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError(
                'groups (cluster_ids) must be provided to ButinaCrossValidator.split()')

        n_samples = len(groups)
        unique_cids = np.unique(groups)

        # Sort clusters largest → smallest so round-robin distributes evenly
        cluster_sizes = {cid: np.sum(groups == cid) for cid in unique_cids}
        sorted_cids = sorted(unique_cids, key=lambda c: -cluster_sizes[c])

        # Assign each cluster to a fold (round-robin)
        fold_assignment = {cid: i % self.n_splits
                           for i, cid in enumerate(sorted_cids)}

        # Yield one boolean test mask per fold
        for fold in range(self.n_splits):
            test_mask = np.array(
                [fold_assignment[cid] == fold for cid in groups],
                dtype=bool
            )
            yield test_mask


def run_cv_regression(model, X, y, cluster_ids, n_splits=5):
    cv, rows = ButinaCrossValidator(n_splits), []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=cluster_ids), 1):
        model.fit(X[tr], y[tr])
        m = regression_metrics(y[te], model.predict(X[te]))
        m['Fold'] = fold
        rows.append(m)
    df = pd.DataFrame(rows).set_index('Fold')
    return pd.concat([df,
                      df.mean().rename('mean').to_frame().T,
                      df.std().rename('std').to_frame().T])


def run_cv_classification(model, X, y, cluster_ids, n_splits=5):
    cv, rows = ButinaCrossValidator(n_splits), []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=cluster_ids), 1):
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        m = classification_metrics(y[te], model.predict(X[te]), prob)
        m['Fold'] = fold
        rows.append(m)
    df = pd.DataFrame(rows).set_index('Fold')
    return pd.concat([df,
                      df.mean().rename('mean').to_frame().T,
                      df.std().rename('std').to_frame().T])


class MoleculeDataset(Dataset):
    """PyTorch Dataset wrapping numpy feature/target arrays."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class EGFRDataModule(L.LightningDataModule):
    """LightningDataModule wrapping pre-split numpy arrays."""

    def __init__(self, X_train, y_train, X_val, y_val, batch_size=64):
        super().__init__()
        self.train_ds = MoleculeDataset(X_train, y_train)
        self.val_ds = MoleculeDataset(X_val, y_val)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=0)


class MLPRegressor(L.LightningModule):
    """3-layer MLP for pChEMBL regression."""

    def __init__(self, input_dim, hidden_dims=(512, 256, 128), dropout=0.3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.BatchNorm1d(in_dim), nn.Linear(in_dim, h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.val_r2 = torchmetrics.R2Score()

    def forward(self, x): return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        y = y.view(-1, 1)
        loss = nn.functional.mse_loss(self(x), y)
        self.log('train_loss', loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self(x)

        loss = nn.functional.mse_loss(y_hat, y)

        self.log('val_loss', loss, prog_bar=True,
                 on_epoch=True, on_step=False)

        self.log('val_r2',
                 self.val_r2(y_hat.view(-1), y.view(-1)),
                 prog_bar=True,
                 on_epoch=True,
                 on_step=False)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}
