import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Analysis App (log(i) vs E)")

uploaded_files = st.file_uploader(
    "Upload your polarization files (.xlsx)", type="xlsx", accept_multiple_files=True)

def clean_data(E, I):
    # Mask: Only positive currents, and both finite
    mask = (I > 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def fit_region(E, logI):
    slope, intercept, r_val, _, _ = linregress(E, logI)
    return slope, intercept, r_val**2

def interpolate_corr(E, I):
    # Find point where current crosses zero (Ecorr, Icorr)
    idx = np.argmin(np.abs(I))
    Ecorr = E[idx]
    Icorr = I[idx]
    return Ecorr, Icorr

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")

        try:
            df = pd.read_excel(file)
            # Lowercase and strip column names for flexiblity
            df.columns = [c.lower().strip() for c in df.columns]
            potential_col = next((col for col in df.columns if 'potential' in col), None)
            current_col = next((col for col in df.columns if 'current' in col), None)

            if not potential_col or not current_col:
                st.error("Could not find 'potential' and 'current' columns in the file.")
                continue

            E = df[potential_col].values
            I = df[current_col].values
            E, I = clean_data(E, I)

            if len(I) < 10:
                st.warning("Too few valid data points.")
                continue

            # Sort by potential for consistent selection
            idx_sort = np.argsort(E)
            E = E[idx_sort]
            I = I[idx_sort]
            logI = np.log10(I)

            # Get Ecorr/Icorr for sliders and plotting
            Ecorr, Icorr = interpolate_corr(E, I)

            # Show preview
            st.write("#### Data preview (gray: all, vertical Ecorr):")
            fig, ax = plt.subplots()
            ax.plot(E, logI, '.', color="lightgray", ms=2, label="All log|I| vs E")
            ax.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.3f} V')
            ax.set_xlabel("Potential (V)")
            ax.set_ylabel("log10(Current / A)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # -- Sliders for fit region selection --
            cathodic_idx = np.where(E < Ecorr)[0]
            anodic_idx = np.where(E > Ecorr)[0]
            # For usability, subsample if too many points
            subsample = lambda ar: ar[::max(1,len(ar)//30)] if len(ar)>30 else ar

            if len(cathodic_idx) < 3 or len(anodic_idx) < 3:
                st.warning("Not enough cathodic or anodic points for fitting. Please check your data.")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Select cathodic linear region:")
                cath_options = [float(f"{E[i]:.5f}") for i in subsample(cathodic_idx)]
                cath_range = st.select_slider(
                    "Cathodic region (choose flat/linear part, before Ecorr):",
                    options=cath_options,
                    value=(cath_options[2], cath_options[-2])
                )
            with col2:
                st.markdown("#### Select anodic linear region:")
                anod_options = [float(f"{E[i]:.5f}") for i in subsample(anodic_idx)]
                anod_range = st.select_slider(
                    "Anodic region (choose flat/linear part, after Ecorr):",
                    options=anod_options,
                    value=(anod_options[2], anod_options[-2])
                )

            # Build masks, fit, handle edge cases
            mask_cath = (E >= cath_range[0]) & (E <= cath_range[1])
            mask_anod = (E >= anod_range[0]) & (E <= anod_range[1])

            if mask_cath.sum() < 3:
                st.warning("Select a wider cathodic window for proper fitting.")
                continue
            if mask_anod.sum() < 3:
                st.warning("Select a wider anodic window for proper fitting.")
                continue

            E_cath, logI_cath = E[mask_cath], logI[mask_cath]
            E_anod, logI_anod = E[mask_anod], logI[mask_anod]

            # Linear fits
            slope_c, int_c, r2_c = fit_region(E_cath, logI_cath)
            slope_a, int_a, r2_a = fit_region(E_anod, logI_anod)

            beta_c = -2.303/slope_c   # See documentation for sign
            beta_a = 2.303/slope_a
            corrosion_rate = (0.00327 * Icorr * 1.0) / (1.0)  # mm/year as placeholder

            # -- Final plot with fits
            fig2, ax2 = plt.subplots()
            ax2.plot(E, logI, '.', color='lightgray', label="All log|I| vs E", lw=0.5)
            ax2.plot(E_cath, logI_cath, 'x', color='blue', label="Selected Cathodic")
            ax2.plot(E_anod, logI_anod, 'o', color='orange', label="Selected Anodic")

            # Fitted lines
            ax2.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={r2_c:.2f})')
            ax2.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={r2_a:.2f})')
            ax2.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.3f} V')
            ax2.set_xlabel("Potential (V)")
            ax2.set_ylabel("log10(Current / A)")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

            st.write("### Tafel Fit Parameters:")
            st.write(f"**Ecorr (V):** `{Ecorr:.5f}`")
            st.write(f"**Icorr (A):** `{Icorr:.3e}`")
            st.write(f"**Beta_a (V/dec):** `{beta_a:.3e}`")
            st.write(f"**Beta_c (V/dec):** `{beta_c:.3e}`")
            st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
            st.write(f"**R² anodic:** `{r2_a:.3f}`")
            st.write(f"**R² cathodic:** `{r2_c:.3f}`")

        except Exception as e:
            st.error(f"Error processing file: {e}")
