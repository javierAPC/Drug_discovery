# EGFR Inhibitor Bioactivity Prediction
### End-to-end ML pipeline for EGFR kinase inhibitor potency prediction

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white)](https://lightning.ai)
[![XGBoost](https://img.shields.io/badge/XGBoost-tuned-FF6600)](https://xgboost.ai)
[![RDKit](https://img.shields.io/badge/RDKit-cheminformatics-009688)](https://rdkit.org)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Results Page](https://img.shields.io/badge/Results-Page-00e5cc?style=flat&logo=github)](https://tu-usuario.github.io/egfr_drug_discovery/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Demo-HF%20Spaces-FFD21E)](https://huggingface.co/spaces/tu-usuario/egfr-predictor)

---

## Overview

Epidermal Growth Factor Receptor (EGFR) is a receptor tyrosine kinase overexpressed in ~30% of all human cancers, most critically non-small cell lung cancer (NSCLC). Five FDA-approved tyrosine kinase inhibitors (TKIs) target EGFR, but resistance mutations — particularly T790M — drive the need for new inhibitor candidates.

This project builds a complete structure–activity relationship (SAR) modelling pipeline trained on **9,876 EGFR bioactivity records** (IC50, CHEMBL203) from ChEMBL, using rigorous Butina cluster-based cross-validation to ensure predictions generalise to structurally novel compounds.

**Headline result:** XGBoost (Optuna-tuned) achieves **R² = 0.727** on 5-fold Butina CV
## Live Demo

A prediction app is available on Hugging Face Spaces:
**[🤗 EGFR Inhibitor Predictor](https://huggingface.co/spaces/JavierPC/EGFR_Drug_discovery)**

Input any SMILES string to get:
- Predicted pChEMBL (pIC50) with estimated IC50 equivalent
- Activity class (active / intermediate / inactive)
- Lipinski Ro5 profile
- Tanimoto applicability domain check
- Structural similarity to 4 FDA-approved EGFR TKIs

A full results page with figures, benchmark tables and SHAP interpretation
is available at:
**[📊 Results Page](https://github.com/javierAPC/Drug_discovery)**


## Biological Context

EGFR inhibitors bind in the ATP-binding pocket through four key interactions:

1. **H-bond to hinge region (Met793)** — requires H-bond donor/acceptor
2. **Hydrophobic contacts** (Leu718, Leu792, Leu844) — requires aromatic rings
3. **Covalent bond to Cys797** (2nd/3rd gen) — requires acrylamide electrophilic warhead
4. **Hydrophobic back pocket** — requires lipophilic substituents

*Sources: Amelia et al., Molecules 2022 ([DOI: 10.3390/molecules27030819](https://doi.org/10.3390/molecules27030819)); Zhao et al., J Hematol Oncol 2022 ([DOI: 10.1186/s13045-022-01311-6](https://doi.org/10.1186/s13045-022-01311-6))*

SHAP analysis confirms the model has learned these pharmacophoric features — MACCS key 98 (heteroatom in aromatic 6-membered ring, encoding the quinazoline/pyrimidine scaffold) is the strongest positive predictor of EGFR inhibition (see Part 6).

---

## Project Structure

```
egfr_drug_discovery/
├── notebooks/
│   ├── part1_data_collection.ipynb      ← ChEMBL query, cleaning, InChI dedup
│   ├── part2_eda_lipinski.ipynb         ← EDA, Lipinski Ro5, Mann-Whitney tests
│   ├── part3_featurization.ipynb        ← ECFP6, MACCS keys, RDKit 2D descriptors
│   ├── part4_baseline_models.ipynb      ← Butina CV, RF baseline, ablation study
│   ├── part5_model_comparison.ipynb     ← Optuna tuning, XGBoost, SVR, MLP (Lightning)
│   ├── part6_shap.ipynb                 ← SHAP explainability, MACCS interpretation
│   └── part7_applicability_domain.ipynb ← Williams plot, UMAP, virtual screen
├── src/
│   ├── clustering.py                    ← ButinaCrossValidator
│   ├── models.py                        ← CV runners, metrics, Lightning modules
│   ├── features.py                      ← Fingerprint computation
│   ├── data_utils.py                    ← Cleaning, pChEMBL conversion
│   ├── shap_utils.py                    ← MACCS labelling, atom weights
│   └── ad_utils.py                      ← Leverage, Tanimoto AD
├── app/
│   ├── app.py                           ← Streamlit prediction app
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── raw/                             ← Original ChEMBL query output
│   ├── processed/                       ← Cleaned datasets, feature matrices
│   ├── clusters/                        ← Butina cluster IDs (.npy)
│   └── results/                         ← CV metrics, SHAP rankings, screen hits
├── models/                              ← Serialised best model (.pkl)
└── figures/
    ├── eda/                             ← Parts 1–3
    ├── cv/                              ← Parts 4–5
    ├── shap/                            ← Part 6
    └── ad/                              ← Part 7
```

---

## Pipeline

### Data Collection & Preprocessing (Part 1)
- Query CHEMBL203 via `chembl_webresource_client` — IC50, binding assays only
- Quality filters: `pchembl_value` not null · `potential_duplicate == 0` · `standard_relation == '='`
- InChI key-based deduplication: compounds with variable IC50 across labs → mean pChEMBL; identical measurements → keep first
- Bioactivity classes: active (pChEMBL ≥ 6.0), intermediate (5.0–6.0), inactive (≤ 5.0)

| | Raw | After filters | After dedup |
|---|---|---|---|
| Records | ~15,000 | ~9,876 | 9,876 |


### Featurization (Part 3)

| Feature set | Raw bits | After filter | Filter type |
|---|---|---|---|
| ECFP6 (Morgan r=3) | 2,048 | 339 | VarianceThreshold(0.05) |
| MACCS keys | 167 | 115 | VarianceThreshold(0.05) |
| RDKit 2D descriptors | 208 | 164 | Correlation > 0.95 removed |

> **Why ECFP6 over PubChem fingerprints:** ECFP6 is the field standard for QSAR benchmarking (Riniker & Landrum, J Cheminform 2013; [DOI: 10.1186/1758-2946-5-26](https://doi.org/10.1186/1758-2946-5-26)) and enables direct comparison with Vignaux et al. 2023.

### Cross-Validation Strategy

**Butina cluster-based CV** (5 folds, sim_cutoff = 0.6) rather than random or Murcko scaffold split.

Butina groups molecules by Tanimoto similarity (1 − Tanimoto as distance). Each fold's test set contains complete clusters of structurally similar compounds never seen during training — this directly tests generalisation across chemotype families (quinazolines, pyrimidines, acrylamides), not interpolation within them.

> Murcko scaffold split was considered but not used: it fragments SAR series into multiple artificial scaffolds due to differing peripheral substituents, overstating structural diversity (Landrum, 2024).

### Model Comparison (Parts 4–5)

All models trained on ECFP6 features, evaluated under identical Butina 5-fold CV.

| Model | Features | Split | R² | RMSE | MAE | Pearson r |
|-------|----------|-------|----|------|-----|-----------|
| Random Forest (default) | ECFP6 | Butina 5-fold | 0.707 ±0.014 | 0.697 | 0.511 | 0.841 |
| RF (Optuna-tuned) | ECFP6 | Butina 5-fold | 0.707 ± 0.014 | 0.696 | 0.511 | 0.842 |
| XGBoost (Optuna-tuned) [Best] | ECFP6 | Butina 5-fold | 0.724 ± 0.008 | 0.676 | 0.495 | 0.851 |
| SVR (Optuna-tuned) | ECFP6 | Butina 5-fold | 0.724 ± 0.011 | 0.676 | 0.495 | 0.851 |
| MLP (Lightning, 3-layer) | ECFP6 | Butina 5-fold | 0.703 ± 0.013 | 0.701 | 0.515 | 0.841 |
| SVR [Literature]<br>Vignaux et al. 2023 · AChE | ECFP6 | 5-fold random | 0.810 | 0.730 | 0.550 | 0.760 |
| GNN [Literature]<br>DeepEGFR · Malik et al. 2025 | Graph + FP | 80/20 random | — | — | — | macro-F1 ≈ 0.94 |

*\* Different target (AChE) and random CV split — not directly comparable, included as methodological reference.*

SVR tied with XGBoost; the latter was chosen for its lower deviation in R², as well as easier SHAP value calculation and training computation.

**Hyperparameter optimisation:** Optuna TPE sampler, 50 trials per model, inner Butina 3-fold on fold-1 training data only (no test leakage).

### Explainability — SHAP on MACCS Keys (Part 6)

MACCS keys (not ECFP6) are used for SHAP because each of the 167 bits maps to a defined SMARTS substructure — directly interpretable without lookup tables.

Top predictive features:

| MACCS bit | Definition | SHAP direction | Biological interpretation |
|---|---|---|---|
| **98** | Heteroatom in aromatic 6-ring | ↑ positive | Quinazoline/pyrimidine scaffold of all approved TKIs |
| **101** | 8-membered ring or larger | ↑ positive | Macrocyclic 3rd-gen inhibitor scaffolds |
| **131** | Multiple H-bond donors (QH > 1) | ↓ negative | Excess H-bond donors reduce membrane permeability |

> **ECFP6 for performance, MACCS for interpretation:** this separation reflects the documented performance–interpretability tradeoff in fingerprint benchmarks (Riniker & Landrum, 2013). MACCS typically underperforms ECFP on virtual screening tasks but its fixed bit definitions enable unambiguous SHAP interpretation.

In the SHAP section of the **[Results Page](https://github.com/javierAPC/Drug_discovery/#shap)** theres a more thruorogh examination of the most influencial bits.


### Applicability Domain (Part 7)

Two complementary AD methods:

**Leverage (Williams plot):** hat matrix diagonal computed on PCA-50 projection. Threshold h\* = 3(k+1)/n. Compounds with h > h\* are flagged as outside the AD. 

*Important limitation:*
Although the Williams plot is a standard approach for defining the AD,
it relies on Euclidean geometry and continuous feature assumptions. In this work,
we use high-dimensional (2048-bit) sparse binary ECFP6 fingerprints, where these
assumptions do not hold.


**Tanimoto distance:** max Tanimoto similarity to any training compound. Threshold = 0.4. More chemically intuitive — directly encodes "has the model seen a similar structure?"

| | Inside AD | Outside AD |
|---|---|---|
| Test compounds | 97.4 % | 2.6 % |
| R² | 0.741 | 0.063 |

---

## Benchmarking

| Paper | Target | Method | R² / F1 | Split |
|---|---|---|---|---|
| **This work** | **EGFR (CHEMBL203)** | **XGBoost, ECFP6** | **R² = 0.727** | **Butina 5-fold** |
| Vignaux et al., 2023 | AChE | SVR, ECFP6 | R² = 0.810 | 5-fold CV (random) |
| Malik et al., 2025 (DeepEGFR) | EGFR | GNN | macro-F1 ≈ 0.94 | 80/20 random |
| Wu et al., 2018 (MoleculeNet) | Multiple | RF | R² 0.44–0.70 | Scaffold |

The gap vs Vignaux (0.727 vs 0.810) is expected: Butina CV tests structural generalisation; random CV allows near-duplicate compounds in both train and test, inflating R² by an estimated 15–30% (Wu et al., 2018).

---

## Key Figures

| Figure | Location | Shows |
|---|---|---|
| pChEMBL distribution | `figures/eda/fig01_pchembl_distribution.png` | Dataset activity distribution with class cutoffs |
| Lipinski chemical space | `figures/eda/fig02_chemical_space_lipinski.png` | MW vs LogP, FDA drugs marked |
| Butina cluster sizes | `figures/cv/fig05_butina_clusters.png` | Cluster distribution, fold balance |
| R² per fold | `figures/cv/fig07_r2_per_fold.png` | CV stability across all models |
| Model comparison | `figures/cv/fig09_r2_comparison.png` | Bar chart with error bars + literature reference |
| SHAP beeswarm | `figures/shap/fig11_shap_beeswarm.png` | Top 20 MACCS bits, per-compound contribution |
| Similarity maps | `figures/shap/fig13_similarity_maps.png` | Atom-level importance on 4 FDA drugs |
| UMAP chemical space | `figures/ad/fig16_umap_chemical_space.png` | Full dataset in 2D, train/test/FDA drugs |
| Williams plot | `figures/ad/fig15_williams_plot.png` | Leverage vs standardised residuals |
| Virtual screen | `figures/ad/fig18_virtual_screen.png` | Top predicted actives with AD flags |

---

## Limitations

- **Dataset scope:** trained on IC50 binding assay data only. Predictions for cell-based or functional assays may not transfer.
- **Single target:** model is specific to wild-type EGFR. T790M and L858R resistance mutations are not modelled (separate ChEMBL targets: CHEMBL3736).
- **No 3D features:** ECFP6 and MACCS encode 2D topology only. 3D pharmacophore and docking-based features could improve performance, particularly for distinguishing stereoisomers.
- **AD boundary is approximate:** both leverage and Tanimoto AD are proxies; the optimal threshold is dataset-dependent.
- **No experimental validation:** virtual screen hits are predictions only — wet-lab confirmation required before drawing biological conclusions.

---

## Reproducing the Results

```bash
# Local
git clone https://github.com/tu-usuario/egfr_drug_discovery.git
pip install -r requirements.txt

# Prediction app (local)
cd app && streamlit run app.py

# Prediction app (Docker)
docker build -t egfr-predictor app/
docker run -p 8501:8501 egfr-predictor
# Open http://localhost:8501
```

> A live version is deployed on
> [Hugging Face Spaces](https://huggingface.co/spaces/tu-usuario/egfr-predictor)
> — no installation required.

---
## Acknowledgements

This project originated from the drug discovery tutorial series by
**Chanin Nantasenamat (Data Professor)**
([github.com/dataprofessor/drugdiscovery](https://github.com/dataprofessor/drugdiscovery)),
which provided the initial framework for querying ChEMBL, computing
Lipinski descriptors, traning on PubChem
fingerprints for acetylcholinesterase, and reuses some of its funtions.

From that foundation, this project diverges substantially: the target
was changed to EGFR (CHEMBL203), the feature pipeline was rebuilt using
ECFP6 and MACCS keys computed natively with RDKit, the evaluation
protocol was replaced with Butina cluster-based CV, and the scope was
extended to include SHAP explainability, applicability domain analysis,
virtual screening.

---
## References

1. **Malik et al. (2025)** — DeepEGFR: a GNN for EGFR bioactivity classification. *Scientific Reports* 15, 38236. [DOI: 10.1038/s41598-025-22126-8](https://doi.org/10.1038/s41598-025-22126-8)
2. **Vignaux et al. (2023)** — Validation of AChE inhibition ML models. *Chem. Res. Toxicol.* 36(2):188–201. [DOI: 10.1021/acs.chemrestox.2c00283](https://doi.org/10.1021/acs.chemrestox.2c00283)
3. **Wu et al. (2018)** — MoleculeNet: a benchmark for molecular ML. *Chem. Sci.* 9:513–530. [DOI: 10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)
4. **Riniker & Landrum (2013)** — Open-source platform to benchmark fingerprints. *J Cheminform* 5:26. [DOI: 10.1186/1758-2946-5-26](https://doi.org/10.1186/1758-2946-5-26)
5. **Amelia et al. (2022)** — Structural insight and development of EGFR TKIs. *Molecules* 27(3):819. [DOI: 10.3390/molecules27030819](https://doi.org/10.3390/molecules27030819)
6. **Zhao et al. (2022)** — Strategies to overcome resistance to 3rd-gen EGFR inhibitors. *J Hematol Oncol* 15:73. [DOI: 10.1186/s13045-022-01311-6](https://doi.org/10.1186/s13045-022-01311-6)

7. **Landrum, G. (2023)** — Variability of x-fold cross validation results.
   *RDKit Blog.*
   [https://greglandrum.github.io/rdkit-blog/posts/2023-08-13-xval-variability1.html](https://greglandrum.github.io/rdkit-blog/posts/2023-08-13-xval-variability1.html)

8. **Landrum, G. (2024)** — The problem(s) with scaffold splits, part 1.
    *RDKit Blog.*
    [https://greglandrum.github.io/rdkit-blog/posts/2024-05-31-scaffold-splits-and-murcko-scaffolds1.html](https://greglandrum.github.io/rdkit-blog/posts/2024-05-31-scaffold-splits-and-murcko-scaffolds1.html)

9. **Nantasenamat, C. (Data Professor)** — Drug Discovery with Machine Learning.
    GitHub repository — original tutorial series this project builds upon.
    [https://github.com/dataprofessor/drugdiscovery](https://github.com/dataprofessor/drugdiscovery)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
