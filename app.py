import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Analysis App (log(I) vs E)")

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

    # Let user pick columns (robust for any header name)
    cols = df.columns.tolist()
    # Try to guess likely column (default to those containing 'current' or 'potential')
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

    # Sort for slider logic and plotting
    idx_sort = np.argsort(E)
    E = E[idx_sort]
    I = I[idx_sort]
    logI = np.log10(I)

    # Find Ecorr/Icorr as closest to zero current
    Ecorr, Icorr = interpolate_corr(E, I)

    # ---------- Preview plot ----------
    fig, ax = plt.subplots()
    ax.plot(E, logI, '.', color="lightgray", markersize=3, label="All log|I| vs E")
    ax.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.3f} V')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(Current / A)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.info("Select the linear(flat) cathodic and anodic regions using sliders ⬇️")
    # ---------- Choose window for fit regions ----------
    # Points left/right of Ecorr
    cath_idx = np.where(E < Ecorr)[0]
    anod_idx = np.where(E > Ecorr)[0]

    if len(cath_idx) < 3 or len(anod_idx) < 3:
        st.error("Not enough data points in one or both regions around Ecorr!")
        st.stop()

    # To avoid massive slider lists, only sample up to 40 points for region endpoints
    def subsample(ar):
        step = max(1, len(ar)//40)
        return ar[::step] if len(ar) > 40 else ar

    cath_options = [float(f"{E[i]:.5f}") for i in subsample(cath_idx)]
    anod_options = [float(f"{E[i]:.5f}") for i in subsample(anod_idx)]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cathodic region:**")
        cath_range = st.select_slider(
            "Pick V range (cathodic, left/blue branch)", cath_options,
            value=(cath_options[min(2, len(cath_options)-2)], cath_options[-2])
        )
    with col2:
        st.write("**Anodic region:**")
        anod_range = st.select_slider(
            "Pick V range (anodic, right/orange branch)", anod_options,
            value=(anod_options[1], anod_options[-2])
        )

    # Mask for fitting
    mask_cath = (E >= cath_range[0]) & (E <= cath_range[1])
    mask_anod = (E >= anod_range[0]) & (E <= anod_range[1])

    if mask_cath.sum() < 3:
        st.error("Select a wider cathodic region to allow fitting.")
        st.stop()
    if mask_anod.sum() < 3:
        st.error("Select a wider anodic region to allow fitting.")
        st.stop()
    E_cath, logI_cath = E[mask_cath], logI[mask_cath]
    E_anod, logI_anod = E[mask_anod], logI[mask_anod]

    slope_c, int_c, r2_c = fit_region(E_cath, logI_cath)
    slope_a, int_a, r2_a = fit_region(E_anod, logI_anod)
    beta_c = -2.303/slope_c
    beta_a = 2.303/slope_a
    corrosion_rate = 0.00327 * Icorr  # mm/y, placeholder

    # ---------- Final plot with fits ----------
    fig2, ax2 = plt.subplots()
    ax2.plot(E, logI, '.', color='lightgray', markersize=3, label="All log|I| vs E")
    ax2.plot(E_cath, logI_cath, 'x', color='blue', label="Cathodic")
    ax2.plot(E_anod, logI_anod, 'o', color='orange', label="Anodic")
    ax2.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={r2_c:.2f})')
    ax2.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={r2_a:.2f})')
    ax2.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.3f} V')
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("log10(Current / A)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # -------- Table of results --------
    st.markdown("### **Tafel Fit Parameters:**")
    st.write(f"**Ecorr (V):** `{Ecorr:.5f}`")
    st.write(f"**Icorr (A):** `{Icorr:.3e}`")
    st.write(f"**Beta_a (V/dec):** `{beta_a:.3e}`")
    st.write(f"**Beta_c (V/dec):** `{beta_c:.3e}`")
    st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
    st.write(f"**R² anodic:** `{r2_a:.3f}`")
    st.write(f"**R² cathodic:** `{r2_c:.3f}`")

    st.success("You can adjust the regions above for the best straight-line fit. The plot above always shows BOTH linear fit regions and their fits.")
