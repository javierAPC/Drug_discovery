"""
EGFR Inhibitor Bioactivity Predictor — Scientific Edition
==========================================================
Run: streamlit run app.py
"""
 
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import io
import base64
import warnings
warnings.filterwarnings("ignore")
 
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdFingerprintGenerator, MACCSkeys
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
 
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EGFR Bioactivity Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────────────────────────────────────
# SCIENTIFIC CSS THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; font-size: 14px; }
.stApp { background: #f7f9fc; color: #1e2d3d; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1100px; }

[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #dde3ec !important; }

.sci-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
             letter-spacing: 0.12em; text-transform: uppercase; color: #6b7280; margin-bottom: 6px; }
.sci-title { font-family: 'Source Serif 4', serif; font-size: 2rem; font-weight: 600;
             color: #0f172a; line-height: 1.2; margin-bottom: 4px; }
.sci-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #1d4ed8;
                letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px; }
.sci-desc { font-size: 13.5px; color: #475569; max-width: 680px; line-height: 1.65; }

.metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-top: 3px solid #1d4ed8;
               border-radius: 0 0 6px 6px; padding: 16px 18px; text-align: center; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 500;
                line-height: 1; margin-bottom: 4px; }
.metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }

.badge-active       { display:inline-block; background:#dcfce7; color:#15803d; border:1px solid #bbf7d0;
                      border-radius:3px; padding:3px 12px; font-family:'JetBrains Mono',monospace;
                      font-size:11px; font-weight:600; letter-spacing:0.08em; }
.badge-inactive     { display:inline-block; background:#fee2e2; color:#dc2626; border:1px solid #fecaca;
                      border-radius:3px; padding:3px 12px; font-family:'JetBrains Mono',monospace;
                      font-size:11px; font-weight:600; letter-spacing:0.08em; }
.badge-intermediate { display:inline-block; background:#fef9c3; color:#a16207; border:1px solid #fef08a;
                      border-radius:3px; padding:3px 12px; font-family:'JetBrains Mono',monospace;
                      font-size:11px; font-weight:600; letter-spacing:0.08em; }

.info-box { background:#eff6ff; border:1px solid #bfdbfe; border-left:3px solid #3b82f6;
            border-radius:0 4px 4px 0; padding:11px 15px; font-size:13px; color:#1e40af;
            line-height:1.6; margin:10px 0; }
.warn-box { background:#fffbeb; border:1px solid #fde68a; border-left:3px solid #f59e0b;
            border-radius:0 4px 4px 0; padding:11px 15px; font-size:13px; color:#92400e;
            line-height:1.6; margin:10px 0; }

.ro5-pass { color:#15803d; font-weight:600; }
.ro5-fail { color:#dc2626; font-weight:600; }

hr { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }

.stTextInput input {
    border: 1px solid #cbd5e1 !important; border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
    background: #ffffff !important; color: #0f172a !important; padding: 10px 14px !important;
}
.stTextInput input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.12) !important; }

.stButton button {
    background: #1d4ed8 !important; color: white !important; border: none !important;
    border-radius: 4px !important; font-weight: 500 !important; font-size: 13px !important;
}
.stButton button:hover { background: #1e40af !important; }

.section-head { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:500;
                letter-spacing:0.15em; text-transform:uppercase; color:#94a3b8;
                border-bottom:1px solid #e2e8f0; padding-bottom:6px; margin-bottom:14px; }

.mol-frame { background:#ffffff; border:1px solid #e2e8f0; border-radius:6px;
             padding:14px; text-align:center; }

/* ===== black text in all the app ===== */
body, .stApp, div, p, span, label, .stMarkdown, [class*="css"] {
    color: #000000 !important;
}

/* white background (opcional, pero evita negro sobre negro) */
.stApp {
    background-color: #f0f2f6 !important;
}

/* Exceptions*/
.stExpander, .stExpander *,
.stCodeBlock, .stCodeBlock *,
.stCodeBlock pre, .stCodeBlock code {
    color: #0f172a !important;  /* texto oscuro, no negro absoluto */
    background-color: #f1f5f9 !important;
}

[data-testid="stSidebar"] * {
    color: #000000 !important;
}
[data-testid="stSidebar"] {
    background-color: #e6f4f1 !important;
}

</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FDA_DRUGS = {
    "Erlotinib (1st gen)":   "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCO",
    "Gefitinib (1st gen)":   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Afatinib (2nd gen)":    "C=CC(=O)N1CCC[C@@H]1c1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2c(OCC)c1",
    "Osimertinib (3rd gen)": "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC",
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
def load_training_fps():
    try:
        df = pd.read_csv("X_ecfp6.csv")
        return df.drop(columns="molecule_chembl_id").values.astype(np.float32)
    except FileNotFoundError:
        return None
 
@st.cache_data(show_spinner=False)
def load_kept_bit_indices():
    """
    The model was trained on ECFP6 bits that survived VarianceThreshold in Part 3.
    X_ecfp6.csv columns encode which bit indices were kept (e.g. 'ecfp6_42' → bit 42).
    """
    try:
        df = pd.read_csv("X_ecfp6.csv")
        cols = df.drop(columns="molecule_chembl_id").columns.tolist()
        return [int(c.split("_")[1]) for c in cols]
    except FileNotFoundError:
        return None
 
# ─────────────────────────────────────────────────────────────────────────────
# CHEMISTRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def smiles_to_mol(smiles: str):
    """Parse SMILES with explicit sanitization fallback."""
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
            if mol:
                Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None
 
def mol_to_ecfp6(mol) -> np.ndarray:
    """
    Compute ECFP6 and apply the same VarianceThreshold feature selection
    that was applied during training (Part 3). Model expects 339 features.
    """
    kept_indices = load_kept_bit_indices()
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fp  = list(fpg.GetFingerprint(mol))
    if kept_indices is not None:
        fp = [fp[i] for i in kept_indices]
    return np.array(fp, dtype=np.float32).reshape(1, -1)
 
def mol_to_image_b64(mol, size=(340, 280)) -> str:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 1.6
    opts.backgroundColour = (1, 1, 1, 1)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText()).decode()
 
def lipinski_check(mol):
    return {
        "Molecular Weight":   (round(Descriptors.MolWt(mol), 1),      "≤ 500 Da",  Descriptors.MolWt(mol) <= 500),
        "LogP (AlogP)":       (round(Descriptors.MolLogP(mol), 2),    "≤ 5",       Descriptors.MolLogP(mol) <= 5),
        "H-bond Donors":      (Lipinski.NumHDonors(mol),               "≤ 5",       Lipinski.NumHDonors(mol) <= 5),
        "H-bond Acceptors":   (Lipinski.NumHAcceptors(mol),            "≤ 10",      Lipinski.NumHAcceptors(mol) <= 10),
        "TPSA (Å²)":          (round(Descriptors.TPSA(mol), 1),       "≤ 140",     Descriptors.TPSA(mol) <= 140),
        "Rotatable Bonds":    (Lipinski.NumRotatableBonds(mol),        "≤ 10",      Lipinski.NumRotatableBonds(mol) <= 10),
    }
 
def tanimoto_ad_check(mol, X_train):
    if X_train is None:
        return None, False
    # Apply the SAME VarianceThreshold selection used during training
    # X_train has 339 columns → query fingerprint must also have 339
    kept_indices = load_kept_bit_indices()
    fpg      = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fp_full  = list(fpg.GetFingerprint(mol))
    q_arr    = np.array(
        [fp_full[i] for i in kept_indices] if kept_indices else fp_full,
        dtype=np.float32
    )
    # Tanimoto via dot product: sim = |A∩B| / |A∪B|
    q_norm     = float(q_arr.sum())
    train_norm = X_train.sum(axis=1)
    dot        = X_train @ q_arr                     # now both are 339-dim ✓
    union      = train_norm + q_norm - dot
    sims       = np.where(union > 0, dot / union, 0.0)
    max_sim    = float(sims.max())
    return round(max_sim, 3), max_sim >= TANIMOTO_THRESHOLD
 
def assign_class(pchembl):
    if pchembl >= 6.0:  return "active"
    if pchembl <= 5.0:  return "inactive"
    return "intermediate"
 
# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY — SCIENTIFIC PALETTE
# ─────────────────────────────────────────────────────────────────────────────
SCI_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="JetBrains Mono, monospace", color="#374151", size=11),
    xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0"),
    margin=dict(l=12, r=12, t=36, b=12),
)
 
def gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"family": "Source Serif 4, serif", "size": 14, "color": "#0f172a"}},
        number={"font": {"family": "JetBrains Mono", "size": 34, "color": "#1d4ed8"}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [4, 11], "tickcolor": "#94a3b8",
                     "tickfont": {"family": "JetBrains Mono", "size": 9}},
            "bar":  {"color": "#1d4ed8", "thickness": 0.22},
            "bgcolor": "#f8fafc",
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [4, 5],  "color": "#fee2e2"},
                {"range": [5, 6],  "color": "#fef9c3"},
                {"range": [6, 11], "color": "#dcfce7"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=210,
                      margin=dict(l=20, r=20, t=55, b=10))
    return fig
 
def radar_chart(lipo):
    labels = list(lipo.keys())
    vals   = [v[0] for v in lipo.values()]
    limits = [500, 5, 5, 10, 140, 10]
    norm   = [min(v / lim, 1.25) for v, lim in zip(vals, limits)]
    norm  += [norm[0]]
    lbls   = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(labels) + 1), theta=lbls, fill="toself",
        fillcolor="rgba(59,130,246,0.06)",
        line=dict(color="rgba(59,130,246,0.3)", dash="dash", width=1),
        showlegend=False
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm, theta=lbls, fill="toself",
        fillcolor="rgba(29,78,216,0.1)",
        line=dict(color="#1d4ed8", width=2),
        showlegend=False
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            angularaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0",
                             tickfont=dict(family="JetBrains Mono", size=9, color="#94a3b8")),
            radialaxis=dict(visible=False, range=[0, 1.35])
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=230,
        margin=dict(l=30, r=30, t=15, b=15),
    )
    return fig
 
def similarity_bar(max_sim):
    color = "#16a34a" if max_sim >= TANIMOTO_THRESHOLD else "#dc2626"
    fig = go.Figure(go.Bar(
        x=[max_sim], y=["Similarity"],
        orientation="h", marker_color=color,
        text=[f"{max_sim:.3f}"], textposition="outside",
        textfont=dict(family="JetBrains Mono", color="#374151", size=12),
    ))
    fig.add_vline(x=TANIMOTO_THRESHOLD, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"AD threshold = {TANIMOTO_THRESHOLD}",
                  annotation_font=dict(family="JetBrains Mono", size=9, color="#92400e"))
    fig.update_layout(
        **{**SCI_LAYOUT, "height": 95,
           "xaxis": dict(range=[0, 1.1], gridcolor="#f1f5f9"),
           "yaxis": dict(showticklabels=False),
           "margin": dict(l=8, r=80, t=8, b=8)}
    )
    return fig
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 16px 0;'>
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:2px;
                    color:#94a3b8;text-transform:uppercase;margin-bottom:6px;'>Target protein</div>
        <div style='font-family:Source Serif 4,serif;font-size:18px;font-weight:600;
                    color:#0f172a;line-height:1.3;'>EGFR · ErbB1<br>
            <span style='font-size:13px;font-weight:400;color:#475569;'>Bioactivity prediction</span>
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;margin-top:8px;'>
            CHEMBL203 · ChEMBL database
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<hr style='border-top:1px solid #e2e8f0;margin:0 0 14px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='sci-label'>Input mode</div>", unsafe_allow_html=True)
 
    input_mode = st.radio(
        "", ["Enter SMILES", "Select reference drug", "Batch screen CSV"],
        label_visibility="collapsed"
    )
 
    st.markdown("<hr style='border-top:1px solid #e2e8f0;margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='sci-label'>Reference compounds</div>", unsafe_allow_html=True)
 
    gen_color = {"1st gen": "#6b7280", "2nd gen": "#d97706", "3rd gen": "#1d4ed8"}
    for name in FDA_DRUGS:
        gen  = name.split("(")[1].replace(")", "").strip()
        drug = name.split("(")[0].strip()
        col  = gen_color.get(gen, "#6b7280")
        st.markdown(f"""
        <div style='padding:7px 10px;border:1px solid #e2e8f0;border-radius:4px;
                    margin-bottom:4px;background:#f8fafc;'>
            <span style='font-size:12.5px;font-weight:500;color:#1e2d3d;'>{drug}</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;
                         color:{col};margin-left:6px;'>({gen})</span>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<hr style='border-top:1px solid #e2e8f0;margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;line-height:1.9;'>
        Model: XGBoost (Optuna-tuned)<br>
        Features: ECFP6 (339 bits, VarianceThreshold)<br>
        Validation: Butina 5-fold CV<br>
        R² = 0.727 · n = 9,876 compounds<br><br>
    </div>
    """, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
model    = load_model()
X_train  = load_training_fps()
 
if not RDKIT_OK:
    st.error("RDKit not found. Install with: `pip install rdkit`")
    st.stop()
 
if model is None:
    st.warning("Model file `best_model_part5.pkl` not found in `/app`. "
               "Place it alongside app.py and rebuild the container.")
 
# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.8rem;'>
    <div class='sci-subtitle'>💊 Cheminformatics · QSAR · Drug Discovery</div>
    <div class='sci-title'>EGFR Inhibitor Bioactivity Predictor</div>
    <div class='sci-desc' style='margin-top:8px;'>
        Predict pChEMBL (−log₁₀ IC50) and activity class for candidate EGFR inhibitors
        from SMILES input. Includes applicability domain assessment via Tanimoto similarity
        to the ChEMBL203 training set (n = 9,876 compounds).
    </div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-head'>Input</div>", unsafe_allow_html=True)
 
smiles_input   = None
compound_name  = "Query compound"
 
if input_mode == "Enter SMILES":
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        smiles_input = st.text_input(
            "SMILES", placeholder="Enter canonical SMILES string…",
            label_visibility="collapsed"
        )
    with col_btn:
        if st.button("Load Erlotinib", use_container_width=True):
            smiles_input  = FDA_DRUGS["Erlotinib (1st gen)"]
            compound_name = "Erlotinib"
 
elif input_mode == "Select reference drug":
    selected = st.radio(
        "Reference drug",
        list(FDA_DRUGS.keys()),
        label_visibility="collapsed",
        horizontal=True,
    )
    smiles_input  = FDA_DRUGS[selected]
    compound_name = selected.split("(")[0].strip()
    st.markdown(f"""
    <div class='info-box' style='margin-top:6px;'>
        <strong>{compound_name}</strong> — {selected.split('(')[1].replace(')','').strip()} EGFR inhibitor
        &nbsp;·&nbsp; <span style='font-family:JetBrains Mono,monospace;font-size:11px;'>{smiles_input}</span>
    </div>
    """, unsafe_allow_html=True)
 
elif input_mode == "Batch screen CSV":
    st.markdown("""
    <div class='info-box'>
        Upload a CSV with a <code>smiles</code> column and optionally a <code>name</code> column.
        All compounds will be predicted and returned as a downloadable CSV.
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if "smiles" not in batch_df.columns:
            st.error("CSV must contain a column named `smiles`")
        else:
            names_col = batch_df.get("name", pd.Series(
                [f"Cpd_{i+1}" for i in range(len(batch_df))]
            ))
            results, bar = [], st.progress(0)
            for i, (smi, nm) in enumerate(zip(batch_df["smiles"], names_col)):
                mol = smiles_to_mol(str(smi))
                if mol is None:
                    results.append({"Name": nm, "SMILES": smi, "Predicted pChEMBL": "Invalid",
                                    "Class": "—", "Inside AD": "—", "Max Tanimoto": "—"})
                else:
                    fp   = mol_to_ecfp6(mol)
                    pred = float(model.predict(fp)[0]) if model else 6.0
                    sim, inside = tanimoto_ad_check(mol, X_train)
                    results.append({"Name": nm, "SMILES": smi,
                                    "Predicted pChEMBL": round(pred, 3),
                                    "Class": assign_class(pred),
                                    "Inside AD": "Yes" if inside else "No",
                                    "Max Tanimoto": sim})
                bar.progress((i + 1) / len(batch_df))
            bar.empty()
            res_df = pd.DataFrame(results)
            st.markdown("<div class='section-head'>Batch results</div>", unsafe_allow_html=True)
            st.dataframe(res_df, use_container_width=True, height=380)
            st.download_button("⬇ Download CSV", res_df.to_csv(index=False).encode(),
                               "egfr_screen_results.csv", "text/csv", use_container_width=True)
    st.stop()
 
# ─────────────────────────────────────────────────────────────────────────────
# GUARD: empty input
# ─────────────────────────────────────────────────────────────────────────────
if not smiles_input or not smiles_input.strip():
    st.markdown("""
    <div class='info-box' style='margin-top:2rem;'>
        Enter a SMILES string above or select a reference compound to run a prediction.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
 
mol = smiles_to_mol(smiles_input)
if mol is None:
    st.markdown("""
    <div class='warn-box'>
        <strong>⚠ Invalid SMILES.</strong>
        Could not parse the input string. Please verify the syntax and try again.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
 
# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
fp_arr       = mol_to_ecfp6(mol)
pred_val     = float(model.predict(fp_arr)[0]) if model else 6.5
cls          = assign_class(pred_val)
lipo         = lipinski_check(mol)
max_sim, inside_ad = tanimoto_ad_check(mol, X_train)
img_b64      = mol_to_image_b64(mol)
ic50_nM      = 10 ** (9 - pred_val)
ic50_str     = f"{ic50_nM:.1f} nM" if ic50_nM < 1000 else f"{ic50_nM/1000:.2f} µM"
violations   = sum(1 for _, _, ok in lipo.values() if not ok)
badge_html   = f"<span class='badge-{cls}'>{cls.upper()}</span>"
ad_color     = "#16a34a" if inside_ad else "#dc2626"
ad_top_color = "#16a34a" if inside_ad else "#dc2626"
cls_color    = {"active": "#16a34a", "inactive": "#dc2626", "intermediate": "#a16207"}.get(cls, "#374151")
 
st.markdown("---")
 
# ─────────────────────────────────────────────────────────────────────────────
# ROW 1: Molecule · Gauge · Radar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-head'>Prediction results</div>", unsafe_allow_html=True)
 
c_mol, c_gauge, c_radar = st.columns([1.1, 1.2, 1.2])
 
with c_mol:
    st.markdown(f"""
    <div class='mol-frame'>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>
            {compound_name}
        </div>
        <img src='data:image/png;base64,{img_b64}'
             style='max-width:100%;border-radius:2px;'/>
        <div style='margin-top:10px;display:flex;justify-content:center;gap:18px;'>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;'>
                {mol.GetNumAtoms()} atoms
            </span>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;'>
                {mol.GetNumBonds()} bonds
            </span>
        </div>
        <div style='margin-top:10px;'>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)
 
with c_gauge:
    st.plotly_chart(gauge_chart(round(pred_val, 3), "Predicted pChEMBL"),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""
    <div style='text-align:center;font-family:JetBrains Mono,monospace;font-size:11px;
                color:#94a3b8;margin-top:-8px;'>
        Estimated IC₅₀ ≈ <span style='color:#374151;font-weight:500;'>{ic50_str}</span>
    </div>
    """, unsafe_allow_html=True)
 
with c_radar:
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;"
                "text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px;'>"
                "Lipinski Ro5 Profile</div>", unsafe_allow_html=True)
    st.plotly_chart(radar_chart(lipo), use_container_width=True,
                    config={"displayModeBar": False})
 
# ─────────────────────────────────────────────────────────────────────────────
# ROW 2: Four metric cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
# Crear columnas vacías a los lados para centrar
left, center, right = st.columns([1, 2, 1])
with center:
    st.markdown(f"""
    <div class='metric-card' style='border-top-color:{ad_top_color}; text-align:center;'>
        <div class='metric-value' style='color:{ad_top_color};'>{'Inside AD' if inside_ad else 'Outside AD'}</div>
        <div class='metric-label'>Domain assessment</div>
    </div>
    """, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# ROW 3: Ro5 table · AD panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
c_ro5, c_ad = st.columns(2)
 
with c_ro5:
    st.markdown("<div class='section-head'>Lipinski Rule of Five</div>",
                unsafe_allow_html=True)

    lipo_rows = []
    for prop, (val, rule, ok) in lipo.items():
        lipo_rows.append({
            "Property": prop,
            "Value": str(val),
            "Limit": rule,
            "Status": "✓" if ok else "✗"
        })

    lipo_df = pd.DataFrame(lipo_rows)

    def colour_status(val):
        return "color: #15803d; font-weight: 600" if val == "✓" \
               else "color: #dc2626; font-weight: 600"

    viol = sum(1 for _, _, ok in lipo.values() if not ok)
    viol_color = "#15803d" if viol == 0 else ("#d97706" if viol == 1 else "#dc2626")
    viol_str   = f"{viol} violation{'s' if viol != 1 else ''}"

    st.dataframe(
        lipo_df.style
            .applymap(colour_status, subset=["Status"])
            .set_properties(**{"background-color": "#ffffff", "color": "#374151",
                               "font-size": "13px"})
            .hide(axis="index"),
        use_container_width=True,
        height=265
    )
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
        f"color:{viol_color};font-weight:600;text-align:right;"
        f"margin-top:4px;'>Ro5 verdict: {viol_str}</div>",
        unsafe_allow_html=True
    )
 
with c_ad:
    st.markdown("<div class='section-head'>Applicability Domain</div>", unsafe_allow_html=True)
    if max_sim is not None:
        st.plotly_chart(similarity_bar(max_sim), use_container_width=True,
                        config={"displayModeBar": False})
    ad_box = "info-box" if inside_ad else "warn-box"
    ad_msg = (
        f"<strong>Inside the applicability domain</strong> (Tanimoto ≥ {TANIMOTO_THRESHOLD}). "
        "The query compound is structurally similar to training compounds. "
        "This prediction is within the reliable chemical space of the model."
    ) if inside_ad else (
        f"<strong>Outside the applicability domain</strong> (Tanimoto &lt; {TANIMOTO_THRESHOLD}). "
        "The query compound is structurally novel relative to the training set. "
        "This prediction represents an extrapolation — experimental validation is required."
    )
    st.markdown(f"<div class='{ad_box}'>{ad_msg}</div>", unsafe_allow_html=True)
 
    # Similarity to reference drugs
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;"
                "text-transform:uppercase;letter-spacing:0.12em;margin:12px 0 8px;'>"
                "Similarity to reference TKIs</div>", unsafe_allow_html=True)
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fp_q = fpg.GetFingerprint(mol)
    ref_rows = ""
    for drug_name, drug_smi in list(FDA_DRUGS.items())[:4]:
        dm = Chem.MolFromSmiles(drug_smi)
        if dm:
            sim = DataStructs.TanimotoSimilarity(fp_q, fpg.GetFingerprint(dm))
            bar_w = int(sim * 100)
            bar_c = "#16a34a" if sim >= 0.4 else "#94a3b8"
            label = drug_name.split("(")[0].strip()
            ref_rows += f"""
            <div style='margin-bottom:6px;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:2px;'>
                    <span style='font-size:11.5px;color:#374151;'>{label}</span>
                    <span style='font-family:JetBrains Mono,monospace;font-size:10px;
                                 color:{bar_c};font-weight:500;'>{sim:.3f}</span>
                </div>
                <div style='background:#f1f5f9;border-radius:2px;height:5px;'>
                    <div style='background:{bar_c};width:{bar_w}%;height:5px;border-radius:2px;'></div>
                </div>
            </div>
            """
    st.markdown(ref_rows, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋  Canonical SMILES & export"):
    canonical = Chem.MolToSmiles(mol, canonical=True)
    st.code(canonical, language="text")
    export_df = pd.DataFrame([{
        "SMILES": canonical, "Compound": compound_name,
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
    st.download_button("⬇ Download prediction report",
                       export_df.to_csv(index=False).encode(),
                       f"egfr_{compound_name.replace(' ','_')}.csv",
                       "text/csv", use_container_width=True)
 

