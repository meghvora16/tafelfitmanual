import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("Tafel Analysis App (log(i) vs E)")

# Upload files
uploaded_files = st.file_uploader("Upload your polarization files (.xlsx)",
type="xlsx", accept_multiple_files=True)

def clean_data(E, I):
  mask = (I > 0) & np.isfinite(E) & np.isfinite(I)
  return E[mask], I[mask]

def split_regions(E, logI):
  mid_idx = np.argmin(np.abs(E - np.median(E)))
  return slice(0, mid_idx), slice(mid_idx, None)

def fit_region(E, logI):
  slope, intercept, r, _, _ = linregress(E, logI)
  return slope, intercept, r**2

def interpolate_corr(E, I):
  f = interp1d(E, I, bounds_error=False, fill_value="extrapolate")
  idx = np.argmin(np.abs(I))
  Ecorr = E[idx]
  Icorr = I[idx]
  return Ecorr, Icorr

def calculate_parameters(E, I):
  logI = np.log10(I)
  cath, anod = split_regions(E, logI)
  
  slope_c, int_c, r2_c = fit_region(E[cath], logI[cath])
  slope_a, int_a, r2_a = fit_region(E[anod], logI[anod])
  
  Ecorr, Icorr = interpolate_corr(E, I)
  
  beta_c = -1 / slope_c * 2.303
  beta_a = 1 / slope_a * 2.303
  
  # Placeholder corrosion rate formula (customize as needed)
  corrosion_rate = (0.00327 * Icorr * 1.0) / (1.0) # mm/year
  
  return {
  "Ecorr (V)": Ecorr,
  "Icorr (A)": Icorr,
  "Beta_a (V/dec)": beta_a,
  "Beta_c (V/dec)": beta_c,
  "Corrosion Rate (mm/y)": corrosion_rate,
  "R² anodic": r2_a,
  "R² cathodic": r2_c
  }, slope_a, int_a, slope_c, int_c

def plot_fit(E, I, slope_a, int_a, slope_c, int_c):
  logI = np.log10(I)
  fig, ax = plt.subplots()
  ax.plot(E, logI, 'ko', label='Data')
  ax.plot(E, slope_a*E + int_a, 'r--', label='Anodic Fit')
  ax.plot(E, slope_c*E + int_c, 'b--', label='Cathodic Fit')
  ax.set_xlabel("Potential (V)")
  ax.set_ylabel("log(Current) [log A]")
  ax.grid(True)
  ax.legend()
  st.pyplot(fig)

if uploaded_files:
  for file in uploaded_files:
    st.subheader(f"📄 {file.name}")
  try:
    df = pd.read_excel(file)
    df.columns = [c.lower().strip() for c in df.columns]
    potential_col = next((col for col in df.columns if 'potential' in col),None)
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

params, slope_a, int_a, slope_c, int_c = calculate_parameters(E, I)

plot_fit(E, I, slope_a, int_a, slope_c, int_c)

st.markdown("Extracted Tafel Parameters:")
for k, v in params.items():
  if isinstance(v, float):
  st.write(f"**{k}:** {v:.3e}")
  else:
  st.write(f"**{k}:** {v}")

except Exception as e:
  st.error(f"Error processing file: {e}")
