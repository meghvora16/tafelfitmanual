import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Analysis App (Auto Linear Region Detection)")

uploaded_file = st.file_uploader(
    "Upload polarization file (.xlsx/.csv)", type=['xlsx', 'csv'], accept_multiple_files=False
)

def clean_data(E, I):
    mask = (I != 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def find_best_linear_region(E, logI, side='cathodic', Ecorr=None, min_pts=5):
    # Find the longest window with best R² for logI vs E, each side of Ecorr
    best_r2 = -np.inf
    best_slice = slice(0, min_pts)
    N = len(E)
    indices = (E < Ecorr) if side == 'cathodic' else (E > Ecorr)
    idxs = np.where(indices)[0]
    for i in range(len(idxs)):
        for j in range(i + min_pts - 1, len(idxs)):
            window = idxs[i:j+1]
            x, y = E[window], logI[window]
            if len(x) < min_pts: continue
            slope, intercept, r_value, _, _ = linregress(x, y)
            if r_value ** 2 > best_r2:
                best_r2 = r_value ** 2
                best_slice = window
    return best_slice, best_r2

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

    # Sort for processing
    idx_sort = np.argsort(E)
    E = E[idx_sort]
    I = I[idx_sort]
    logI = np.log10(I)

    Ecorr, Icorr = interpolate_corr(E, I)

    st.write(f"**Auto-detected Ecorr:** {Ecorr:.3f} V")

    # -------- Auto find best linear region each side of Ecorr --------
    cath_slice, r2_c = find_best_linear_region(E, logI, 'cathodic', Ecorr)
    anod_slice, r2_a = find_best_linear_region(E, logI, 'anodic', Ecorr)

    E_cath, logI_cath, I_cath = E[cath_slice], logI[cath_slice], I[cath_slice]
    E_anod, logI_anod, I_anod = E[anod_slice], logI[anod_slice], I[anod_slice]

    slope_c, int_c, fitr2_c = fit_region(E_cath, logI_cath)
    slope_a, int_a, fitr2_a = fit_region(E_anod, logI_anod)
    beta_c = -2.303/slope_c
    beta_a = 2.303/slope_a
    corrosion_rate = 0.00327 * Icorr  # mm/y, placeholder

    # -------- Raw plot with regions highlighted -------
    fig0, ax0 = plt.subplots(figsize=(8,4))
    ax0.plot(E, I, '.', color='lightgray', ms=3, label='All Data')
    ax0.plot(E_cath, I_cath, 'x', color='blue', label='Best cathodic linear region')
    ax0.plot(E_anod, I_anod, 'o', color='orange', label='Best anodic linear region')
    ax0.axvline(Ecorr, color='gray', ls='--', lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax0.set_xlabel('Potential (V)')
    ax0.set_ylabel('Current (A)')
    ax0.legend()
    ax0.grid(True)
    st.pyplot(fig0)
    st.caption("Best-fit Tafel region on raw LSV curve (detected automatically).")

    # ------- Tafel plot -----------
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(E, logI, '.', color="lightgray", markersize=3, label="All log|I| vs E")
    ax.plot(E_cath, logI_cath, 'x', color='blue', label="Auto cathodic region")
    ax.plot(E_anod, logI_anod, 'o', color='orange', label="Auto anodic region")
    ax.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={fitr2_c:.2f})')
    ax.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={fitr2_a:.2f})')
    ax.axvline(Ecorr, color='gray', linestyle='--', lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(Current / A)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.caption("Auto-detected most linear Tafel regions and linear fits.")

    # -------- Table of results --------
    st.markdown("### **Tafel Fit Parameters (auto linear region):**")
    st.write(f"**Ecorr (V):** `{Ecorr:.5f}`")
    st.write(f"**Icorr (A):** `{Icorr:.3e}`")
    st.write(f"**Beta_a (V/dec):** `{beta_a:.3e}`")
    st.write(f"**Beta_c (V/dec):** `{beta_c:.3e}`")
    st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
    st.write(f"**R² anodic:** `{fitr2_a:.3f}`")
    st.write(f"**R² cathodic:** `{fitr2_c:.3f}`")
