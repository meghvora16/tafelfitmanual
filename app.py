import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

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

            # --------- Sliders for fit regions --------
            st.write("### Pick fitting window for Cathodic (left, blue) and Anodic (right, orange):")
            min_E, max_E = np.min(E), np.max(E)
            col1, col2 = st.columns(2)

            with col1:
                cathodic_range = st.slider(
                    "Cathodic fit region (V)", float(min_E), float(Ecorr), 
                    (float(min_E + 0.1*(Ecorr-min_E)), float(Ecorr - 0.02)),
                    step=0.01
                )
            with col2:
                anodic_range = st.slider(
                    "Anodic fit region (V)", float(Ecorr), float(max_E),
                    (float(Ecorr + 0.02), float(max_E-0.1*(max_E-Ecorr))), 
                    step=0.01
                )

            # Get indices for each region
            idx_cath = (E >= cathodic_range[0]) & (E <= cathodic_range[1])
            idx_anod = (E >= anodic_range[0]) & (E <= anodic_range[1])

            # --------- Fit only selected region ---------
            E_cath = E[idx_cath]
            logI_cath = logI[idx_cath]
            E_anod = E[idx_anod]
            logI_anod = logI[idx_anod]

            slope_c, int_c, r2_c = fit_region(E_cath, logI_cath)
            slope_a, int_a, r2_a = fit_region(E_anod, logI_anod)

            beta_c = -1/slope_c * 2.303
            beta_a = 1/slope_a * 2.303

            corrosion_rate = (0.00327 * Icorr * 1.0) / (1.0) # Placeholder

            # -------- Show plot including only fit region -----
            fig, ax = plt.subplots()

            # All data
            ax.plot(E, logI, '.', color='lightgray', label="All log|I| vs E")
            # Fitted region highlight
            ax.plot(E_cath, logI_cath, 'x', color='blue', label="Cathodic Fit Region")
            ax.plot(E_anod, logI_anod, 'x', color='orange', label="Anodic Fit Region")
            
            # Fit lines
            fit_cath_line = slope_c*E_cath + int_c
            fit_anod_line = slope_a*E_anod + int_a
            ax.plot(E_cath, fit_cath_line, 'b-', lw=2, label=f'Cathodic Fit (R²={r2_c:.2f})')
            ax.plot(E_anod, fit_anod_line, 'r-', lw=2, label=f'Anodic Fit (R²={r2_a:.2f})')
            # Ecorr
            ax.axvline(Ecorr, color='gray', linestyle='--', label=f'Ecorr = {Ecorr:.2f} V')

            ax.set_xlabel("Potential (V)")
            ax.set_ylabel("log₁₀(Current) [A]")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.markdown("### Tafel Fit Parameters:")
            st.write(f"**Ecorr (V):** {Ecorr:.3e}")
            st.write(f"**Icorr (A):** {Icorr:.3e}")
            st.write(f"**Beta_a (V/dec):** {beta_a:.3e}")
            st.write(f"**Beta_c (V/dec):** {beta_c:.3e}")
            st.write(f"**Corrosion Rate (mm/y):** {corrosion_rate:.3e}")
            st.write(f"**R² anodic:** {r2_a:.3f}")
            st.write(f"**R² cathodic:** {r2_c:.3f}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
