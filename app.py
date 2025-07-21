import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def sci_notation(val, precision=3):
    try:
        return f"{float(val):.{precision}e}"
    except (TypeError, ValueError):
        return val

def fit_tafel_region(E, logI):
    slope, intercept, r_val, _, _ = linregress(E, logI)
    return slope, intercept, r_val**2

def interpolate_corr(E, I):
    idx = np.argmin(np.abs(I))
    Ecorr = E[idx]
    Icorr = I[idx]
    return Ecorr, Icorr

st.set_page_config(layout='wide')
st.title("Tafel Analysis With Slider-Selected Linear Region")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=['xlsx', 'csv'])

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("#### Data preview")
    st.dataframe(df.head())

    # User picks columns
    columns = df.columns.tolist()
    E_col = st.selectbox("Select the Potential column:", columns, index=0)
    I_col = st.selectbox("Select the Current column:", columns, index=1)

    E_all = df[E_col].values.astype(float)
    I_all = df[I_col].values.astype(float)
    mask_valid = (np.isfinite(E_all) & np.isfinite(I_all)) & (I_all != 0)
    E = E_all[mask_valid]
    I = I_all[mask_valid]
    logI = np.log10(np.abs(I))
    Ecorr, Icorr = interpolate_corr(E, I)

    st.markdown("### 1. Choose the region (window) for the Tafel (linear) fit:")
    default_win = 0.25
    left_default = float(max(np.min(E), Ecorr - default_win))
    right_default = float(min(np.max(E), Ecorr + default_win))

    region = st.slider("Select Potential Range For Fitting (activation/linear region)",
                       float(np.min(E)), float(np.max(E)),
                       (left_default, right_default), step=0.001)

    mask_fit = (E >= region[0]) & (E <= region[1])
    E_win = E[mask_fit]
    I_win = I[mask_fit]
    logI_win = logI[mask_fit]

    # --- Plot 1: Raw I vs E with highlighted fit region
    fig_raw, ax_raw = plt.subplots(figsize=(8, 4))
    ax_raw.plot(E, I, '.', label='Raw Data')
    ax_raw.axvspan(region[0], region[1], color='yellow', alpha=0.3, label='Fit region')
    ax_raw.axvline(Ecorr, color="gray", ls="--", lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax_raw.set_xlabel('Potential (V)')
    ax_raw.set_ylabel('Current (A)')
    ax_raw.legend()
    ax_raw.grid(True)
    st.pyplot(fig_raw)
    st.caption("**Activation (fit) region highlighted in yellow.**")

    # --- Plot 2: Tafel (log|I| vs E) with highlighted fit region and linear fit
    slope, intercept, r2 = fit_tafel_region(E_win, logI_win)
    fit_line = slope * E_win + intercept
    beta = 2.303/slope if slope > 0 else -2.303/slope
    corrosion_rate = 0.00327 * np.abs(Icorr)  # mm/y, placeholder

    fig_tafel, ax_tafel = plt.subplots(figsize=(8, 4))
    ax_tafel.plot(E, logI, '.', color='gray', label='All log|I| vs E')
    ax_tafel.plot(E_win, logI_win, 'o', color='C1', label='Selected Linear Region')
    ax_tafel.plot(E_win, fit_line, '-', color='C0', lw=2, label=f'Linear Fit (R²={r2:.3f})')
    ax_tafel.axvspan(region[0], region[1], color='yellow', alpha=0.3)
    ax_tafel.axvline(Ecorr, color="gray", ls="--", lw=1.2, label=f'Ecorr = {Ecorr:.3f} V')
    ax_tafel.set_xlabel('Potential (V)')
    ax_tafel.set_ylabel('log10(|Current| / A)')
    ax_tafel.legend()
    ax_tafel.grid(True)
    st.pyplot(fig_tafel)
    st.caption("**Tafel plot with fit region & linear fit.**")

    # --- Results Table ---
    st.markdown("### Results")
    result_table = {
        "Ecorr (V)": sci_notation(Ecorr),
        "Icorr (A)": sci_notation(Icorr),
        "Beta (V/dec)": sci_notation(beta),
        "Corrosion Rate (mm/y)": sci_notation(corrosion_rate),
        "R² fit": f"{r2:.3f}"
    }
    st.table(result_table)

    if r2 < 0.95:
        st.warning("Low R² for fit: try tightening region to a straight section only.")
