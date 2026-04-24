"""
EGFR Inhibitor Bioactivity Predictor
=====================================
Streamlit app — Part 8 of the EGFR Drug Discovery ML pipeline.

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# ── Imports with graceful error messages ─────────────────────────────────────
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import (
        AllChem, Draw, Descriptors, Lipinski,
        rdFingerprintGenerator, MACCSkeys
    )
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EGFR Inhibitor Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark molecular aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #06080f;
    color: #c8d8e8;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Grid background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,229,204,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,204,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Hero header ── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #e8f4ff;
    line-height: 1.15;
    margin: 0 0 0.4rem 0;
}
.hero-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: #00e5cc;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-desc {
    font-size: 0.9rem;
    color: #7a9ab5;
    max-width: 640px;
    line-height: 1.6;
}

/* ── Metric cards ── */
.metric-card {
    background: #0c1420;
    border: 1px solid #1a2d45;
    border-radius: 8px;
    padding: 18px 22px;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.72rem;
    color: #7a9ab5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Activity badge ── */
.badge-active       { background: rgba(34,211,164,0.15); color: #22d3a4; border: 1px solid rgba(34,211,164,0.35); border-radius: 4px; padding: 4px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; }
.badge-inactive     { background: rgba(245,101,101,0.15); color: #f56565; border: 1px solid rgba(245,101,101,0.35); border-radius: 4px; padding: 4px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; }
.badge-intermediate { background: rgba(245,166,35,0.15); color: #f5a623; border: 1px solid rgba(245,166,35,0.35); border-radius: 4px; padding: 4px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; }

/* ── Section headers ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00e5cc;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1a2d45, transparent);
}

/* ── Info box ── */
.info-box {
    background: rgba(0,229,204,0.06);
    border: 1px solid rgba(0,229,204,0.2);
    border-left: 3px solid #00e5cc;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #c8d8e8;
    line-height: 1.6;
    margin: 12px 0;
}
.warn-box {
    background: rgba(245,166,35,0.07);
    border: 1px solid rgba(245,166,35,0.25);
    border-left: 3px solid #f5a623;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #c8d8e8;
    line-height: 1.6;
    margin: 12px 0;
}

/* ── Lipinski table ── */
.ro5-pass { color: #22d3a4; font-weight: 600; }
.ro5-fail { color: #f56565; font-weight: 600; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1018 !important;
    border-right: 1px solid #1a2d45;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: #0c1420 !important;
    border: 1px solid #1a2d45 !important;
    color: #c8d8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid #1a2d45; margin: 1.5rem 0; }

/* ── Plotly background match ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FDA_DRUGS = {
    "Erlotinib (1st gen)":   "C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Gefitinib (1st gen)":   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Afatinib (2nd gen)":    "C=CC(=O)N1CCC[C@@H]1c1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2c(OCC)c1",
    "Osimertinib (3rd gen)": "C=CC(=O)Nc1cc2c(Nc3ccc(N(C)CCN(C)C)c(OC)c3)ncnc2cn1C",
    "Dacomitinib (2nd gen)": "C=CC(=O)N1CCC[C@@H]1c1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
}

TANIMOTO_THRESHOLD = 0.4

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open("best_model_part5.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def load_training_data():
    try:
        df = pd.read_csv("egfr_bioactivity_cleaned.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner=False)
def load_training_fps():
    try:
        X_df = pd.read_csv("X_ecfp6.csv")
        return X_df.drop(columns="molecule_chembl_id").values.astype(np.float32)
    except FileNotFoundError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# CHEMISTRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def smiles_to_mol(smiles: str):
    if not RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol

def mol_to_ecfp6(mol, n_bits=2048) -> np.ndarray:
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
    fp  = fpg.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32).reshape(1, -1)

def mol_to_image_b64(mol, size=(350, 300)) -> str:
    """Render molecule as base64 PNG."""
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().bondLineWidth = 1.8
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return base64.b64encode(png).decode()

def lipinski_check(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot  = Lipinski.NumRotatableBonds(mol)
    return {
        "Molecular Weight": (mw, "≤ 500 Da", mw <= 500),
        "LogP (AlogP)":     (round(logp, 2), "≤ 5", logp <= 5),
        "H-bond Donors":    (hbd, "≤ 5", hbd <= 5),
        "H-bond Acceptors": (hba, "≤ 10", hba <= 10),
        "TPSA (Å²)":        (round(tpsa, 1), "≤ 140", tpsa <= 140),
        "Rotatable Bonds":  (rot, "≤ 10", rot <= 10),
    }

def tanimoto_ad_check(mol, X_train, threshold=TANIMOTO_THRESHOLD):
    """Return max Tanimoto similarity to training set."""
    if X_train is None:
        return None, False
    fpg      = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fp_query = fpg.GetFingerprint(mol)
    # Convert training numpy array back to RDKit fps for bulk similarity
    # (approximate: use numpy dot product for speed)
    query_arr  = np.array(fp_query, dtype=np.float32)
    train_norm = X_train.sum(axis=1)
    query_norm = float(query_arr.sum())
    dot        = X_train @ query_arr
    union      = train_norm + query_norm - dot
    sims       = np.where(union > 0, dot / union, 0.0)
    max_sim    = float(sims.max())
    return round(max_sim, 3), max_sim >= threshold

def assign_class(pchembl):
    if pchembl >= 6.0:  return "active"
    if pchembl <= 5.0:  return "inactive"
    return "intermediate"

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,20,32,0.6)",
    font=dict(family="IBM Plex Mono, monospace", color="#c8d8e8", size=11),
    xaxis=dict(gridcolor="#1a2d45", zerolinecolor="#1a2d45"),
    yaxis=dict(gridcolor="#1a2d45", zerolinecolor="#1a2d45"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def gauge_chart(value, title, min_val=4, max_val=11):
    """Plotly gauge for pChEMBL value."""
    # Colour zones: inactive < 5, intermediate 5-6, active > 6
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"family": "DM Serif Display", "size": 16, "color": "#e8f4ff"}},
        number={"font": {"family": "IBM Plex Mono", "size": 36, "color": "#00e5cc"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#7a9ab5",
                     "tickfont": {"family": "IBM Plex Mono", "size": 9}},
            "bar":  {"color": "#00e5cc", "thickness": 0.25},
            "bgcolor": "#0c1420",
            "bordercolor": "#1a2d45",
            "steps": [
                {"range": [min_val, 5],  "color": "rgba(245,101,101,0.15)"},
                {"range": [5, 6],        "color": "rgba(245,166,35,0.15)"},
                {"range": [6, max_val],  "color": "rgba(34,211,164,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#22d3a4", "width": 2},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220,
                      margin=dict(l=20, r=20, t=60, b=10))
    return fig

def radar_chart(lipinski_data):
    """Plotly radar for Lipinski properties."""
    labels  = list(lipinski_data.keys())
    values  = [v[0] for v in lipinski_data.values()]
    # Normalise each property to 0–1 scale for the radar
    limits  = [500, 5, 5, 10, 140, 10]
    norm    = [min(v / lim, 1.2) for v, lim in zip(values, limits)]
    norm   += [norm[0]]  # close the polygon
    labels_c = labels + [labels[0]]

    fig = go.Figure()
    # Threshold zone (all properties at limit = 1.0)
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(labels) + 1), theta=labels_c,
        fill="toself", fillcolor="rgba(0,229,204,0.05)",
        line=dict(color="rgba(0,229,204,0.3)", dash="dash", width=1),
        name="Ro5 limit", showlegend=False
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm, theta=labels_c,
        fill="toself", fillcolor="rgba(0,229,204,0.12)",
        line=dict(color="#00e5cc", width=2),
        name="Compound", showlegend=False
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(12,20,32,0.6)",
            angularaxis=dict(gridcolor="#1a2d45", linecolor="#1a2d45",
                             tickfont=dict(family="IBM Plex Mono", size=9, color="#7a9ab5")),
            radialaxis=dict(visible=False, range=[0, 1.3])
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=30, r=30, t=20, b=20),
    )
    return fig

def similarity_bar_chart(max_sim, threshold=TANIMOTO_THRESHOLD):
    """Horizontal bar showing Tanimoto similarity."""
    color = "#22d3a4" if max_sim >= threshold else "#f56565"
    fig = go.Figure(go.Bar(
        x=[max_sim], y=["Tanimoto"],
        orientation="h",
        marker_color=color,
        text=[f"{max_sim:.3f}"],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", color="#c8d8e8", size=13),
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#f5a623",
                  annotation_text=f"AD threshold ({threshold})",
                  annotation_font=dict(family="IBM Plex Mono", size=9, color="#f5a623"))
    fig.update_layout(
        **{**DARK_LAYOUT, "height": 100,
           "xaxis": dict(range=[0, 1.1], gridcolor="#1a2d45"),
           "yaxis": dict(showticklabels=False),
           "margin": dict(l=10, r=80, t=10, b=10)}
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px 0;'>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                    letter-spacing: 2px; color: #00e5cc; text-transform: uppercase;
                    margin-bottom: 6px;'>Target</div>
        <div style='font-family: DM Serif Display, serif; font-size: 20px;
                    color: #e8f4ff; line-height: 1.2;'>EGFR Inhibitor<br>Bioactivity</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                    color: #7a9ab5; margin-top: 6px;'>CHEMBL203 · ChEMBL dataset</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>Input mode</div>", unsafe_allow_html=True)

    input_mode = st.radio(
        "", ["Enter SMILES", "Select FDA drug", "Batch screen CSV"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div class='section-label'>Example drugs</div>", unsafe_allow_html=True)

    for drug_name, smi in FDA_DRUGS.items():
        gen = drug_name.split("(")[1].replace(")", "").strip()
        gen_color = {"1st gen": "#7a9ab5", "2nd gen": "#f5a623", "3rd gen": "#00e5cc"}.get(gen, "#7a9ab5")
        st.markdown(f"""
        <div style='margin-bottom: 6px; padding: 8px 10px;
                    background: #0c1420; border: 1px solid #1a2d45;
                    border-radius: 4px; cursor: default;'>
            <span style='font-family: IBM Plex Mono, monospace; font-size: 10px;
                         font-weight: 600; color: {gen_color};'>{drug_name.split("(")[0].strip()}</span>
            <span style='font-size: 9px; color: #7a9ab5; margin-left: 6px;'>({gen})</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                color: #4a6a80; line-height: 1.7;'>
        Model: Random Forest / XGBoost<br>
        Features: ECFP6 (2048 bits)<br>
        CV: Butina 5-fold<br>
        Dataset: ~2400 EGFR compounds<br>
        <br>
        <a href='https://doi.org/10.1038/s41598-025-22126-8'
           style='color: #00e5cc; text-decoration: none;'>DeepEGFR 2025 ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <div class='hero-subtitle'>🧬 EGFR · Drug Discovery · ML Pipeline</div>
    <div class='hero-title'>Inhibitor Bioactivity<br><em style='color:#00e5cc;'>Predictor</em></div>
    <div class='hero-desc'>
        Predict pChEMBL (pIC50) and activity class for EGFR inhibitor candidates.
        Applicability domain check included — predictions outside the training
        chemical space are flagged as extrapolations.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
model     = load_model()
df_train  = load_training_data()
X_train   = load_training_fps()

if not RDKIT_OK:
    st.error("RDKit is not installed. Run: `pip install rdkit-pypi`")
    st.stop()

if model is None:
    st.warning("""
    **Model file not found.** Place `best_model_part5.pkl` in the same directory as `app.py`.
    The app will still demonstrate the interface with mock predictions.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)

smiles_input = None
compound_name = "Query compound"

if input_mode == "Enter SMILES":
    col_in, col_ex = st.columns([3, 1])
    with col_in:
        smiles_input = st.text_input(
            "SMILES string",
            placeholder="e.g.  C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
            label_visibility="collapsed",
        )
    with col_ex:
        if st.button("Try Erlotinib", use_container_width=True):
            smiles_input = FDA_DRUGS["Erlotinib (1st gen)"]
            compound_name = "Erlotinib"

elif input_mode == "Select FDA drug":
    selected = st.selectbox("Select an FDA-approved EGFR inhibitor",
                             list(FDA_DRUGS.keys()),
                             label_visibility="collapsed")
    smiles_input  = FDA_DRUGS[selected]
    compound_name = selected.split("(")[0].strip()

elif input_mode == "Batch screen CSV":
    st.markdown("""
    <div class='info-box'>
        Upload a CSV with a <code>smiles</code> column (and optionally a <code>name</code> column).
        Each row will be predicted independently.
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if "smiles" not in batch_df.columns:
            st.error("CSV must have a column named `smiles`")
        else:
            smiles_list = batch_df["smiles"].tolist()
            names_list  = batch_df.get("name", pd.Series(
                [f"Compound_{i+1}" for i in range(len(smiles_list))]
            )).tolist()
            # Process batch
            batch_results = []
            progress = st.progress(0)
            for i, (smi, nm) in enumerate(zip(smiles_list, names_list)):
                mol = smiles_to_mol(smi)
                if mol is None:
                    batch_results.append({"Name": nm, "SMILES": smi,
                                          "Predicted pChEMBL": "Invalid SMILES",
                                          "Class": "—", "Inside AD": "—",
                                          "Max Tanimoto": "—"})
                    continue
                fp = mol_to_ecfp6(mol)
                pred = float(model.predict(fp)[0]) if model else 6.0
                max_sim, inside = tanimoto_ad_check(mol, X_train)
                batch_results.append({
                    "Name": nm, "SMILES": smi,
                    "Predicted pChEMBL": round(pred, 3),
                    "Class": assign_class(pred),
                    "Inside AD": "✓" if inside else "✗",
                    "Max Tanimoto": max_sim,
                })
                progress.progress((i + 1) / len(smiles_list))
            progress.empty()
            result_df = pd.DataFrame(batch_results)
            st.markdown("<div class='section-label'>Batch Results</div>", unsafe_allow_html=True)

            # Colour the class column
            def colour_class(val):
                c = {"active": "color: #22d3a4", "inactive": "color: #f56565",
                     "intermediate": "color: #f5a623"}.get(val, "")
                return c
            st.dataframe(
                result_df.style.applymap(colour_class, subset=["Class"]),
                use_container_width=True, height=350
            )
            csv = result_df.to_csv(index=False).encode()
            st.download_button("⬇ Download results CSV", csv,
                               "egfr_screen_results.csv", "text/csv",
                               use_container_width=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-COMPOUND PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if not smiles_input or not smiles_input.strip():
    st.markdown("""
    <div class='info-box' style='margin-top: 2rem;'>
        Enter a SMILES string or select a reference drug above to run a prediction.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

mol = smiles_to_mol(smiles_input)

if mol is None:
    st.markdown("""
    <div class='warn-box'>
        ⚠️ <strong>Invalid SMILES.</strong> Could not parse the input string.
        Please check the syntax and try again.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Compute features ──────────────────────────────────────────────────────────
fp_arr   = mol_to_ecfp6(mol)
pred_val = float(model.predict(fp_arr)[0]) if model else 6.5  # fallback demo
cls      = assign_class(pred_val)
lipo     = lipinski_check(mol)
max_sim, inside_ad = tanimoto_ad_check(mol, X_train)
img_b64  = mol_to_image_b64(mol)
n_atoms  = mol.GetNumAtoms()
n_bonds  = mol.GetNumBonds()

# ── Badge HTML ────────────────────────────────────────────────────────────────
badge_html = f"<span class='badge-{cls}'>{cls.upper()}</span>"

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS ROW 1: Molecule + Gauge + Lipinski radar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Prediction results</div>", unsafe_allow_html=True)

col_mol, col_gauge, col_radar = st.columns([1.1, 1.2, 1.2])

with col_mol:
    st.markdown(f"""
    <div style='background:#0c1420; border:1px solid #1a2d45; border-radius:8px;
                padding:16px; text-align:center;'>
        <div style='font-family:IBM Plex Mono,monospace; font-size:9px;
                    letter-spacing:2px; color:#7a9ab5; text-transform:uppercase;
                    margin-bottom:8px;'>{compound_name}</div>
        <img src='data:image/png;base64,{img_b64}'
             style='max-width:100%; border-radius:4px; background:white;'/>
        <div style='margin-top:10px; display:flex; justify-content:center;
                    gap:16px;'>
            <span style='font-family:IBM Plex Mono,monospace; font-size:10px;
                         color:#7a9ab5;'>{n_atoms} atoms</span>
            <span style='font-family:IBM Plex Mono,monospace; font-size:10px;
                         color:#7a9ab5;'>{n_bonds} bonds</span>
        </div>
        <div style='margin-top:10px;'>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    st.plotly_chart(
        gauge_chart(round(pred_val, 3), "Predicted pChEMBL"),
        use_container_width=True, config={"displayModeBar": False}
    )
    # IC50 equivalent
    ic50_nM = 10 ** (9 - pred_val)
    ic50_str = f"{ic50_nM:.1f} nM" if ic50_nM < 1000 else f"{ic50_nM/1000:.2f} μM"
    st.markdown(f"""
    <div style='text-align:center; font-family:IBM Plex Mono,monospace;
                font-size:11px; color:#7a9ab5; margin-top:-10px;'>
        Estimated IC50 ≈ <span style='color:#c8d8e8;'>{ic50_str}</span>
    </div>
    """, unsafe_allow_html=True)

with col_radar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace; font-size:9px;
                letter-spacing:2px; color:#7a9ab5; text-transform:uppercase;
                margin-bottom:4px;'>Lipinski Ro5 Profile</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(radar_chart(lipo), use_container_width=True,
                    config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS ROW 2: Metrics + Lipinski table + AD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

violations = sum(1 for _, _, ok in lipo.values() if not ok)
ad_label   = "Inside AD" if inside_ad else "Outside AD"
ad_color   = "#22d3a4" if inside_ad else "#f56565"
cls_color  = {"active": "#22d3a4", "inactive": "#f56565",
              "intermediate": "#f5a623"}.get(cls, "#c8d8e8")

for col, (val, label, color) in zip(
    [col_m1, col_m2, col_m3, col_m4],
    [
        (f"{pred_val:.3f}", "Predicted pChEMBL", "#00e5cc"),
        (cls.upper(), "Activity Class", cls_color),
        (f"{max_sim:.3f}" if max_sim is not None else "—", "Max Tanimoto Sim.", ad_color),
        (ad_label, "Applicability Domain", ad_color),
    ]
):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:{color};'>{val}</div>
        <div class='metric-label'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS ROW 3: Lipinski table + AD similarity bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_ro5, col_ad = st.columns([1, 1])

with col_ro5:
    st.markdown("<div class='section-label'>Lipinski Rule of Five</div>",
                unsafe_allow_html=True)
    rows_html = ""
    for prop, (val, rule, ok) in lipo.items():
        status_class = "ro5-pass" if ok else "ro5-fail"
        status_icon  = "✓" if ok else "✗"
        rows_html += f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:8px 12px; border-bottom:1px solid rgba(26,45,69,0.5);
                    font-size:13px;'>
            <span style='color:#c8d8e8;'>{prop}</span>
            <span style='font-family:IBM Plex Mono,monospace; color:#e8f4ff;
                         font-weight:500;'>{val}</span>
            <span style='font-family:IBM Plex Mono,monospace; font-size:11px;
                         color:#7a9ab5;'>{rule}</span>
            <span class='{status_class}'>{status_icon}</span>
        </div>
        """
    viol_str = f"{violations} violation{'s' if violations != 1 else ''}"
    viol_color = "#22d3a4" if violations == 0 else ("#f5a623" if violations <= 1 else "#f56565")
    st.markdown(f"""
    <div style='background:#0c1420; border:1px solid #1a2d45; border-radius:6px;
                overflow:hidden;'>
        {rows_html}
        <div style='padding:10px 12px; display:flex; justify-content:space-between;
                    background:rgba(0,229,204,0.04);'>
            <span style='font-family:IBM Plex Mono,monospace; font-size:10px;
                         color:#7a9ab5; text-transform:uppercase;'>Ro5 assessment</span>
            <span style='font-family:IBM Plex Mono,monospace; font-size:11px;
                         color:{viol_color}; font-weight:600;'>{viol_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_ad:
    st.markdown("<div class='section-label'>Applicability Domain</div>",
                unsafe_allow_html=True)
    if max_sim is not None:
        st.plotly_chart(similarity_bar_chart(max_sim), use_container_width=True,
                        config={"displayModeBar": False})
    ad_box_class = "info-box" if inside_ad else "warn-box"
    ad_icon      = "✓" if inside_ad else "⚠️"
    ad_msg = (
        f"{ad_icon} <strong>Inside the applicability domain</strong> (Tanimoto ≥ {TANIMOTO_THRESHOLD}). "
        "The model has seen structurally similar compounds during training — "
        "this prediction is within the reliable region."
    ) if inside_ad else (
        f"{ad_icon} <strong>Outside the applicability domain</strong> (Tanimoto < {TANIMOTO_THRESHOLD}). "
        "This compound is structurally novel relative to the training set. "
        "Treat this prediction as an extrapolation — experimental validation is required."
    )
    st.markdown(f"<div class='{ad_box_class}'>{ad_msg}</div>", unsafe_allow_html=True)

    # Reference comparison
    if df_train is not None:
        st.markdown("<div class='section-label' style='margin-top:1rem;'>Reference compounds</div>",
                    unsafe_allow_html=True)
        ref_data = []
        for drug_name, drug_smi in list(FDA_DRUGS.items())[:3]:
            drug_mol = Chem.MolFromSmiles(drug_smi)
            if drug_mol:
                fpg      = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
                fp_drug  = fpg.GetFingerprint(drug_mol)
                fp_query = fpg.GetFingerprint(mol)
                sim      = DataStructs.TanimotoSimilarity(fp_query, fp_drug)
                ref_data.append({"Drug": drug_name.split("(")[0].strip(),
                                  "Similarity": round(sim, 3)})
        if ref_data:
            ref_df = pd.DataFrame(ref_data)
            fig_ref = go.Figure(go.Bar(
                x=ref_df["Similarity"], y=ref_df["Drug"],
                orientation="h",
                marker_color=["#00e5cc" if s >= 0.4 else "#7a9ab5"
                              for s in ref_df["Similarity"]],
                text=[f"{s:.3f}" for s in ref_df["Similarity"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10, color="#c8d8e8"),
            ))
            fig_ref.update_layout(
                **{**DARK_LAYOUT, "height": 130,
                   "xaxis": dict(range=[0, 1.1], gridcolor="#1a2d45", title=""),
                   "yaxis": dict(gridcolor="#1a2d45"),
                   "margin": dict(l=10, r=60, t=10, b=10)}
            )
            st.plotly_chart(fig_ref, use_container_width=True,
                            config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# SMILES OUTPUT + EXPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋  Canonical SMILES & export"):
    canonical = Chem.MolToSmiles(mol, canonical=True)
    st.code(canonical, language="text")

    export_row = pd.DataFrame([{
        "SMILES": canonical,
        "Predicted pChEMBL": round(pred_val, 3),
        "Estimated IC50 (nM)": round(ic50_nM, 2),
        "Activity class": cls,
        "Max Tanimoto to training": max_sim,
        "Inside AD": inside_ad,
        "Ro5 violations": violations,
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": round(Descriptors.TPSA(mol), 1),
    }])
    st.download_button(
        "⬇ Download prediction report (CSV)",
        export_row.to_csv(index=False).encode(),
        f"egfr_prediction_{compound_name.replace(' ', '_')}.csv",
        "text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family:IBM Plex Mono,monospace; font-size:9px;
            color:#4a6a80; line-height:1.9; padding: 8px 0 20px 0;'>
    EGFR Drug Discovery ML Pipeline · CHEMBL203 · Butina 5-fold CV<br>
    Benchmark: <a href='https://doi.org/10.1038/s41598-025-22126-8'
    style='color:#00e5cc; text-decoration:none;'>DeepEGFR (Malik et al., 2025)</a>
    · <a href='https://doi.org/10.1021/acs.chemrestox.2c00283'
    style='color:#00e5cc; text-decoration:none;'>Vignaux et al., 2023</a><br>
    Predictions are for research purposes only. Not for clinical use.
</div>
""", unsafe_allow_html=True)
