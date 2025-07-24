import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Single-Branch Extrapolation")

uploaded_file = st.file_uploader(
    "Upload polarization file (.xlsx/.csv)",
    type=['xlsx', 'csv'], accept_multiple_files=False
)

def clean_data(E, I):
    mask = (I != 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def fit_region(E, logI):
    slope, intercept, r2, _, _ = linregress(E, logI)
    return slope, intercept, r2**2

def find_Ecorr(E, I):
    idx = np.argmin(np.abs(I))
    return E[idx], I[idx]

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("#### Data preview")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    guess_E = next((c for c in cols if "potential" in c.lower()), cols[0])
    guess_I = next((c for c in cols if "current" in c.lower()), cols[1])
    potential_col = st.selectbox("Select the Potential column:", cols, index=cols.index(guess_E))
    current_col = st.selectbox("Select the Current column:", cols, index=cols.index(guess_I))

    E_raw = df[potential_col].values.astype(float)
    I_raw = df[current_col].values.astype(float)
    E, I = clean_data(E_raw, np.abs(I_raw))

    if len(E) < 10:
        st.warning("Too few valid data points for analysis.")
        st.stop()

    idx_sort = np.argsort(E)
    E = E[idx_sort]
    I = I[idx_sort]
    logI = np.log10(I)

    # Ecorr (minimum |I|, commonly used extrapolation reference)
    Ecorr, _ = find_Ecorr(E, I)
    st.write(f"**Ecorr (minimum |I|):** `{Ecorr:.5f}` V`")

    # ---- Choose branch ----
    branch = st.radio("Choose which Tafel branch to analyze:", ["Cathodic (E < Ecorr)", "Anodic (E > Ecorr)"])
    if branch.startswith("Cathodic"):
        branch_mask = E < Ecorr
        branch_label = 'cathodic'
        color = 'blue'
        beta_sign = -2.303
    else:
        branch_mask = E > Ecorr
        branch_label = 'anodic'
        color = 'orange'
        beta_sign = 2.303

    indices = np.where(branch_mask)[0]
    if len(indices) < 5:
        st.error("Not enough points on this branch for Tafel fitting.")
        st.stop()

    def subsample(ar):
        step = max(1, len(ar)//40)
        return ar[::step] if len(ar) > 40 else ar

    options = [float(f"{E[i]:.5f}") for i in subsample(indices)]
    st.write(f"**Pick linear {branch_label} region for Tafel fit:**")
    region = st.select_slider(
        f"Tafel fit window ({branch_label})", options,
        value=(options[2], options[-2])
    )

    mask_fit = (E >= region[0]) & (E <= region[1])
    if mask_fit.sum() < 3:
        st.error("Pick a wider region for meaningful fit.")
        st.stop()

    E_fit, logI_fit, I_fit = E[mask_fit], logI[mask_fit], I[mask_fit]

    # Fit
    slope, intercept, r2 = fit_region(E_fit, logI_fit)
    beta = beta_sign/slope

    # Extrapolate fit to Ecorr
    logIcorr = slope * Ecorr + intercept
    Icorr = 10 ** logIcorr
    corrosion_rate = 0.00327 * Icorr  # mm/y, placeholder

    # -------- Plot ---------
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(E, logI, '.', color="lightgray", ms=3, label="All log|I| vs E")
    ax.plot(E_fit, logI_fit, 'o', color=color, label=f"Selected {branch_label} region")
    ax.plot(E_fit, slope*E_fit + intercept, '-', color=color, lw=2, label=f"Fit (R²={r2:.2f})")
    ax.axvline(Ecorr, color='purple', ls='--', lw=1.7, label=f'Ecorr = {Ecorr:.3f} V')
    ax.scatter([Ecorr], [logIcorr], color="red", marker='*', s=120, zorder=10, label="Extrapolated log(Icorr)")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(|Current| / A)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.caption(
        f"Fit the linear ({branch_label}) branch, then extrapolate to Ecorr for Icorr."
    )

    # Results
    st.markdown(f"### **Tafel Fit Results ({branch_label.capitalize()} branch, Single-branch Extrapolation):**")
    st.write(f"**Slope:** `{slope:.3f}` (log10A/V)")
    st.write(f"**Intercept:** `{intercept:.3f}`")
    st.write(f"**Beta ({branch_label}, V/dec):** `{beta:.3e}`")
    st.write(f"**Ecorr (V):** `{Ecorr:.5f}`")
    st.write(f"**Extrapolated Icorr (A):** `{Icorr:.3e}`")
    st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
    st.write(f"**R² fit:** `{r2:.3f}`")

    st.info("Switch branch via the radio button above. Repeat for both cathodic and anodic sides for comparison or as needed.")
