import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Analysis App (Auto-Suggested Activation Region)")

uploaded_file = st.file_uploader(
    "Upload your polarization file (.xlsx/.csv)",
    type=['xlsx', 'csv'], accept_multiple_files=False
)

def clean_data(E, I):
    mask = (I != 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def fit_region(E, logI):
    slope, intercept, r2, _, _ = linregress(E, logI)
    return slope, intercept, r2**2

def interpolate_corr(E, I):
    idx = np.argmin(np.abs(I))
    Ecorr = E[idx]
    Icorr = I[idx]
    return Ecorr, Icorr

if uploaded_file:
    # Read
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("#### Data preview")
    st.dataframe(df.head())

    # Let user pick columns
    cols = df.columns.tolist()
    guess_E = next((c for c in cols if "potential" in c.lower()), cols[0])
    guess_I = next((c for c in cols if "current" in c.lower()), cols[1])
    potential_col = st.selectbox("Select the Potential column:", cols, index=cols.index(guess_E))
    current_col = st.selectbox("Select the Current column:", cols, index=cols.index(guess_I))

    # Data
    E_raw = df[potential_col].values.astype(float)
    I_raw = df[current_col].values.astype(float)
    E, I = clean_data(E_raw, np.abs(I_raw))  # Tafel analysis is log(|I|) vs. E

    if len(E) < 10:
        st.warning("Too few valid data points for analysis.")
        st.stop()

    # Sort by E for plotting and processing
    idx_sort = np.argsort(E)
    E = E[idx_sort]
    I = I[idx_sort]
    logI = np.log10(I)

    # Find Ecorr/Icorr as closest to zero current
    Ecorr, Icorr = interpolate_corr(E, I)

    # ---- AUTO REGION: activation region as Ecorr ± 0.07 V (clipped to data range) ----
    delta_V = 0.07
    E_cath_left = max(np.min(E), Ecorr - delta_V)
    E_cath_right = Ecorr
    E_anod_left = Ecorr
    E_anod_right = min(np.max(E), Ecorr + delta_V)

    # Mask for fitting (auto region!)
    mask_cath = (E >= E_cath_left) & (E < E_cath_right)
    mask_anod = (E > E_anod_left) & (E <= E_anod_right)

    # Show warning if region too small
    if mask_cath.sum() < 3 or mask_anod.sum() < 3:
        st.error("Auto activation region (Ecorr ± 0.07 V) is too small for fitting. Adjust delta_V or check your data.")
        st.stop()

    E_cath, logI_cath = E[mask_cath], logI[mask_cath]
    E_anod, logI_anod = E[mask_anod], logI[mask_anod]
    I_cath = I[mask_cath]
    I_anod = I[mask_anod]

    # ---- Preview plot: log(I) vs E with yellow highlight ----
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(E, logI, '.', color="lightgray", markersize=3, label="All log|I| vs E")
    ax.axvspan(E_cath_left, E_cath_right, color='yellow', alpha=0.3, label="Auto cathodic region")
    ax.axvspan(E_anod_left, E_anod_right, color='gold', alpha=0.3, label="Auto anodic region")
    ax.axvline(Ecorr, color='gray', linestyle='--', lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(Current / A)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.caption("Yellow = Automatically detected activation (Tafel) region used for fit.")

    # ---- Raw i vs. E plot with auto region highlighted (for user context) ----
    fig0, ax0 = plt.subplots(figsize=(8,4))
    ax0.plot(E, I, '.', color='lightgray', markersize=3, label="All Data")
    ax0.plot(E_cath, I_cath, 'x', color='blue', label="Auto cathodic region")
    ax0.plot(E_anod, I_anod, 'o', color='orange', label="Auto anodic region")
    ax0.axvspan(E_cath_left, E_cath_right, color='yellow', alpha=0.3)
    ax0.axvspan(E_anod_left, E_anod_right, color='gold', alpha=0.3)
    ax0.axvline(Ecorr, color='gray', linestyle='--', lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax0.set_xlabel("Potential (V)")
    ax0.set_ylabel("Current (A)")
    ax0.legend()
    ax0.grid(True)
    st.pyplot(fig0)
    st.caption("Yellow/Gold = Activation (fit) window (auto-detected). Used for parameter extraction.")

    # ---- Tafel fit on activation regions only ----
    slope_c, int_c, r2_c = fit_region(E_cath, logI_cath)
    slope_a, int_a, r2_a = fit_region(E_anod, logI_anod)
    beta_c = -2.303/slope_c
    beta_a = 2.303/slope_a
    corrosion_rate = 0.00327 * Icorr  # mm/y, placeholder

    # --- Tafel plot with fit lines on activation region ---
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(E, logI, '.', color='lightgray', markersize=3, label="All log|I| vs E")
    ax2.plot(E_cath, logI_cath, 'x', color='blue', label="Auto cathodic region")
    ax2.plot(E_anod, logI_anod, 'o', color='orange', label="Auto anodic region")
    ax2.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={r2_c:.2f})')
    ax2.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={r2_a:.2f})')
    ax2.axvspan(E_cath_left, E_cath_right, color='yellow', alpha=0.3)
    ax2.axvspan(E_anod_left, E_anod_right, color='gold', alpha=0.3)
    ax2.axvline(Ecorr, color='gray', linestyle='--', lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("log10(Current / A)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    st.caption("Tafel fits on activation regions. Regions selected automatically.")

    # -------- Table of results --------
    st.markdown("### **Tafel Fit Parameters (from auto region):**")
    st.write(f"**Ecorr (V):** `{Ecorr:.5f}`")
    st.write(f"**Icorr (A):** `{Icorr:.3e}`")
    st.write(f"**Beta_a (V/dec):** `{beta_a:.3e}`")
    st.write(f"**Beta_c (V/dec):** `{beta_c:.3e}`")
    st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
    st.write(f"**R² anodic:** `{r2_a:.3f}`")
    st.write(f"**R² cathodic:** `{r2_c:.3f}`")
