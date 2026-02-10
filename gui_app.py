# gui_app.py
import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fit_model import fit_first_order  # fits Ka and tau (and optionally y0)

st.set_page_config(page_title="First-Order Fit (Ka, τ)", layout="wide")
st.title("First-Order Step Fit (fit Ka and τ fast)")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1) Upload data")
    uploaded = st.file_uploader(
        "Excel (.xlsx) with time in col A and y in col B",
        type=["xlsx"]
    )

    st.subheader("2) Model inputs")
    t0 = st.number_input("Step time t₀", value=0.0)
    fit_y0 = st.checkbox("Fit baseline offset y₀", value=True)

    st.subheader("Excel read options")
    sheet = st.text_input("Sheet name (blank = first sheet)", value="")
    header = st.checkbox("First row is header", value=True)

    fit_btn = st.button("Fit model", type="primary", use_container_width=True)

with right:
    st.subheader("Plot")

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

# Read Excel
try:
    df = pd.read_excel(
        uploaded,
        sheet_name=sheet.strip() if sheet.strip() else 0,
        header=0 if header else None
    )
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

if df.shape[1] < 2:
    st.error("Need at least 2 columns: time (A) and measurement (B).")
    st.stop()

t = df.iloc[:, 0].to_numpy()
y = df.iloc[:, 1].to_numpy()

# Plot data always
fig, ax = plt.subplots()
ax.plot(t, y, marker="o", linestyle="None", label="Data")
ax.set_xlabel("t")
ax.set_ylabel("y")
ax.grid(True)

result = None
if fit_btn:
    try:
        result = fit_first_order(t, y, t0=float(t0), fit_y0=bool(fit_y0))
        ax.plot(result["t"], result["y_fit"], linestyle="-", label="Fit")
        ax.legend()
    except Exception as e:
        st.error(f"Fit failed: {e}")

st.pyplot(fig, clear_figure=True)

# Show numbers + download
if result is not None:
    st.success("Fit complete.")
    st.write(
        f"**Ka** = {result['Ka']:.6g}   |   **τ** = {result['tau']:.6g}   |   "
        f"**y₀** = {result['y0']:.6g}   |   **SSE** = {result['SSE']:.6g}   |   **R²** = {result['R2']:.6g}"
    )
    st.caption(
        f"Initial guesses: Ka0={result['Ka0']:.4g}, tau0={result['tau0']:.4g}, y0_guess={result['y0_guess']:.4g}"
    )

    out = pd.DataFrame({
        "t": result["t"],
        "y": result["y"],
        "y_fit": result["y_fit"],
        "residual": result["residuals"]
    })

    summary = pd.DataFrame({
        "parameter": ["t0", "Ka", "tau", "y0", "SSE", "R2"],
        "value": [t0, result["Ka"], result["tau"], result["y0"], result["SSE"], result["R2"]]
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="FittedData")
        summary.to_excel(writer, index=False, sheet_name="FitSummary")

    st.download_button(
        "Download fitted Excel",
        data=buffer.getvalue(),
        file_name="first_order_fitted.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )