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
    mask = (I > 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def fit_region(E, logI):
    slope, intercept, r_val, _, _ = linregress(E, logI)
    return slope, intercept, r_val**2

def interpolate_corr(E, I):
    idx = np.argmin(np.abs(I))
    Ecorr = E[idx]
    Icorr = I[idx]
    return Ecorr, Icorr

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f'📄 {file.name}')
        try:
            df = pd.read_excel(file)
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

            logI = np.log10(I)
            Ecorr, Icorr = interpolate_corr(E, I)

            # Sort E and corresponding logI, for easier selection and plotting
            sort_idx = np.argsort(E)
            E = E[sort_idx]
            logI = logI[sort_idx]

            st.write("### Choose fit region for Tafel slopes (flat linear):")

            # Show the entire graph so user can preview windows
            fig, ax = plt.subplots()
            ax.plot(E, logI, 'o', color="grey", ms=2, label="All log|I| vs E")
            ax.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.2f} V')
            ax.set_xlabel("Potential (V)")
            ax.set_ylabel("log10(Current / A)")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Interactive: Use select_slider, showing actual values to guide the user
            # Show a limited number of points for the slider so it is usable
            cathodic_indices = np.where(E < Ecorr)[0]
            anodic_indices = np.where(E > Ecorr)[0]
            if len(cathodic_indices) < 3 or len(anodic_indices) < 3:
                st.warning("Not enough points for Tafel fitting on one side of Ecorr.")
                continue

            # For user-friendliness, use every Nth point for window endpoints (so we don't show 2000 points)
            cathodic_sub_idx = cathodic_indices[::max(1,len(cathodic_indices)//30)]
            anodic_sub_idx = anodic_indices[::max(1,len(anodic_indices)//30)]

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Cathodic (left/blue) region")
                cath_range = st.select_slider(
                    "Select cathodic fit region (potential window)", 
                    options=[float(f"{E[i]:.4f}") for i in cathodic_sub_idx],
                    value=(float(f"{E[cathodic_sub_idx[2]]:.4f}"), float(f"{E[cathodic_sub_idx[-2]]:.4f}"))
                )
            with col2:
                st.write("#### Anodic (right/orange) region")
                anod_range = st.select_slider(
                    "Select anodic fit region (potential window)", 
                    options=[float(f"{E[i]:.4f}") for i in anodic_sub_idx],
                    value=(float(f"{E[anodic_sub_idx[2]]:.4f}"), float(f"{E[anodic_sub_idx[-2]]:.4f}"))
                )

            # Mask for fit regions
            mask_cath = (E >= cath_range[0]) & (E <= cath_range[1])
            mask_anod = (E >= anod_range[0]) & (E <= anod_range[1])

            # Check for empty
            if mask_cath.sum() < 3 or mask_anod.sum() < 3:
                st.warning("Choose a wider fit range on both sides for proper fitting.")
                continue

            E_cath, logI_cath = E[mask_cath], logI[mask_cath]
            E_anod, logI_anod = E[mask_anod], logI[mask_anod]

            # ---- Fit -----
            slope_c, int_c, r2_c = fit_region(E_cath, logI_cath)
            slope_a, int_a, r2_a = fit_region(E_anod, logI_anod)
            beta_c = -1/slope_c * 2.303
            beta_a = 1/slope_a * 2.303
            corrosion_rate = (0.00327 * Icorr * 1.0) / (1.0)

            # ---- Second plot, for fit ----
            fig2, ax2 = plt.subplots()
            ax2.plot(E, logI, '.', color='lightgray', label="All log|I| vs E", lw=0.5)
            ax2.plot(E_cath, logI_cath, 'x', color='blue', label="Cathodic Region")
            ax2.plot(E_anod, logI_anod, 'o', color='orange', label="Anodic Region")
            ax2.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={r2_c:.2f})')
            ax2.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={r2_a:.2f})')
            ax2.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.2f} V')
            ax2.set_xlabel("Potential (V)")
            ax2.set_ylabel("log10(Current / A)")
            ax2.legend()
            ax2.grid()
            st.pyplot(fig2)

            st.write("### Tafel Fit Parameters:")
            st.write(f"**Ecorr (V):** {Ecorr:.3e}")
            st.write(f"**Icorr (A):** {Icorr:.3e}")
            st.write(f"**Beta_a (V/dec):** {beta_a:.3e}")
            st.write(f"**Beta_c (V/dec):** {beta_c:.3e}")
            st.write(f"**Corrosion Rate (mm/y):** {corrosion_rate:.3e}")
            st.write(f"**R² anodic:** {r2_a:.3f}")
            st.write(f"**R² cathodic:** {r2_c:.3f}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
