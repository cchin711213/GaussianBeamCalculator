import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gaussian Beam Simulator", layout="wide")

st.title("🔦 Gaussian Beam ($TEM_{00}$) Propagation")
st.markdown("""
This app calculates the **$1/e^2$ beam radius** along the optical axis using the **ABCD Matrix formalism**.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Initial Beam Parameters")
w0_init = st.sidebar.number_input("Initial Waist $w_0$ (mm)", value=0.05, step=0.01, format="%.3f")
wavelength_nm = st.sidebar.number_input("Wavelength $\lambda$ (nm)", value=852.0, step=1.0)
x_max = st.sidebar.number_input("Max Propagation Distance $x$ (mm)", value=500.0, step=50.0)

st.sidebar.header("Lens Configuration")
num_lenses = st.sidebar.slider("Number of Lenses", 0, 5, 3)

lenses = []
for i in range(num_lenses):
    st.sidebar.subheader(f"Lens {i+1}")
    # Default values based on your setup: f=[50, 100, 200] at x=[50, 150, 450]
    default_x = [50.0, 150.0, 450.0][i] if i < 3 else 100.0 * (i+1)
    default_f = [50.0, 100.0, 200.0][i] if i < 3 else 100.0
    
    lp = st.sidebar.number_input(f"Position $x_{i+1}$ (mm)", value=default_x, key=f"pos_{i}")
    lf = st.sidebar.number_input(f"Focal Length $f_{i+1}$ (mm)", value=default_f, key=f"foc_{i}")
    lenses.append((lp, lf))

# --- CALCULATION LOGIC ---
wavelength = wavelength_nm * 1e-6
zr0 = (np.pi * w0_init**2) / wavelength
q0 = 1j * zr0

x_vals = np.linspace(0, x_max, 2000)
waists = []

for x in x_vals:
    A, B, C, D = 1, 0, 0, 1
    current_pos = 0
    for lp, lf in sorted(lenses):
        if x > lp:
            d = lp - current_pos
            A, B, C, D = (A + C*d), (B + D*d), C, D
            A, B, C, D = A, B, (C - A/lf), (D - B/lf)
            current_pos = lp
    
    d_final = x - current_pos
    A, B, C, D = (A + C*d_final), (B + D*d_final), C, D
    q_x = (A * q0 + B) / (C * q0 + D)
    w_x = np.sqrt(-wavelength / (np.pi * np.imag(1.0 / q_x)))
    waists.append(w_x)

waists = np.array(waists)

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(12, 7))

# Font size settings
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 20, 18, 14, 12

ax.plot(x_vals, waists, color='darkblue', linewidth=2.5, label='$1/e^2$ Radius $w(x)$')
ax.plot(x_vals, -waists, color='darkblue', linewidth=2.5)
ax.fill_between(x_vals, -waists, waists, color='royalblue', alpha=0.15)

# Annotate Lenses
lens_results = []
for lp, lf in lenses:
    idx = np.argmin(np.abs(x_vals - lp))
    w_at_l = waists[idx]
    lens_results.append({"Pos": lp, "f": lf, "Waist": w_at_l})
    
    ax.axvline(x=lp, color='red', linestyle='--', alpha=0.5)
    ax.text(lp, max(waists)*0.8, f'f={lf}mm\n$w$={w_at_l:.3f}mm', 
             color='red', ha='center', fontsize=FS_ANNOT, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

ax.set_title(f'Gaussian Beam Propagation ($\lambda$={wavelength_nm}nm)', fontsize=FS_TITLE)
ax.set_xlabel('Distance $x$ (mm)', fontsize=FS_LABEL)
ax.set_ylabel('Beam Radius (mm)', fontsize=FS_LABEL)
ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=FS_LABEL, loc='upper left')
ax.set_ylim(-max(waists)*1.3, max(waists)*1.3)

st.pyplot(fig)

# --- DATA TABLE ---
st.subheader("Beam Size at Lenses")
st.table(lens_results)

# Final Focus Logic
last_lens_x = max([l[0] for l in lenses]) if lenses else 0
after_lens_indices = np.where(x_vals > last_lens_x)[0]
if len(after_lens_indices) > 0:
    min_idx = after_lens_indices[np.argmin(waists[after_lens_indices])]
    st.success(f"**Final Focus Detected:** $w$ = {waists[min_idx]:.5f} mm at $x$ = {x_vals[min_idx]:.2f} mm")
