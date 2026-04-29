import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import get_metric, median_ra_global, sigma_robust   

st.set_page_config(page_title="RMT Dashboard", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def plot_spectrum(results):
    eigvals = results["eigvals"]
    lamda_cong = results["lamda_cong"]
    lamda_tru= results["lamda_tru"]
    q = results["q"]

    fig, ax = plt.subplots(figsize=(9,4))

    ax.hist(eigvals, bins=40, density=True, alpha=0.7)
    
    x = np.linspace(max(lamda_tru,1e-6), lamda_cong, 300)
    mp = np.sqrt(np.maximum((lamda_cong-x)*(x-lamda_tru),0)) / (2*np.pi*q*x)

    ax.plot(x, mp, lw=2)
    ax.axvline(lamda_cong, ls='--')

    ax.set_title("Eigenvalue Spectrum")
    return fig


def plot_sync_heat(rolling_df):
    # Calculate Z-score
    rolling_df["z_score"] = (rolling_df["lamda_ra"] - median_ra_global) / sigma_robust
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot background colors (Heat Zones)
    ax.axhspan(-5, 1.5, color='green', alpha=0.1, label='Stable (<1.5)')
    ax.axhspan(1.5, 3.0, color='yellow', alpha=0.1, label='Caution (1.5-3.0)')
    ax.axhspan(3.0, 15, color='red', alpha=0.1, label='Black Swan (>3.0)')
    
    # Plot Z-score
    ax.plot(rolling_df.index, rolling_df["z_score"], label="Robust Z-Score", color='#1f77b4', lw=2)
    
    # Plot Signal Gap Score
    if "signal_gap_score" in rolling_df.columns:
        ax.plot(rolling_df.index, rolling_df["signal_gap_score"], label="Signal Gap Score", color='#ff7f0e', alpha=0.7, ls='--')
    
    # Threshold Line
    ax.axhline(1.5, color='red', ls=':', alpha=0.6, label='Z=1.5 Threshold')
    
    # Regime Labeling
    regimes = [
        ("COVID-19", "2020-03-01", "2020-05-31", "gray"),
        ("Bull Run", "2021-01-01", "2022-04-30", "blue"),
        ("Downturn", "2022-05-01", "2022-11-30", "orange")
    ]
    
    for label, start, end, color in regimes:
        s_dt = pd.to_datetime(start)
        e_dt = pd.to_datetime(end)
        if s_dt in rolling_df.index and e_dt in rolling_df.index:
            ax.axvspan(s_dt, e_dt, color=color, alpha=0.05)
            # Find a middle point for the label
            mid_date = s_dt + (e_dt - s_dt) / 2
            ax.text(mid_date, ax.get_ylim()[1]*0.9, label, horizontalalignment='center', fontsize=8, color=color, weight='bold')

    ax.set_title("Sync-Heat Monitoring: RMT Robust Z-Score & Signal Gap")
    ax.set_ylabel("MAD Deviations / Gap Score")
    ax.set_ylim(-2, max(10, rolling_df["z_score"].max() + 2))
    ax.legend(loc='upper left', fontsize='small', ncol=2)
    ax.grid(True, alpha=0.2)
    
    return fig


st.title("RMT Market Dashboard")

df = load_data("dataset.csv")

window = st.sidebar.slider("Window", 60, 250, 120)

if "idx" not in st.session_state:
    st.session_state.idx = len(df) - window

available_dates = df.index[:-window]
if st.session_state.idx >= len(available_dates):
    st.session_state.idx = len(available_dates) - 1

start_dt = st.sidebar.slider(
    "Timeline",
    min_value=available_dates[0].to_pydatetime(),
    max_value=available_dates[-1].to_pydatetime(),
    value=available_dates[st.session_state.idx].to_pydatetime(),
    format="MM/YYYY"
)


start = df.index.get_indexer([pd.Timestamp(start_dt)], method='pad')[0]
st.session_state.idx = start

c_start = df.index[start]
c_end = df.index[start+window-1]
st.sidebar.info(f"**Khoảng thời gian:**  \n{c_start.strftime('%d/%m/%Y')} — {c_end.strftime('%d/%m/%Y')}")

sub_df = df.iloc[start:start+window]

results = get_metric(sub_df)
metrics = results["chỉ số"]
lamda_ra = metrics["lamda_ra"]
num_sig = metrics["num_sig"]
z_score = results["z_score_global"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("λ1 Sync", f"{lamda_ra*100:.1f}%")
col2.metric("Signals", int(num_sig))
col3.metric("Robust Z-Score (MAD)", f"{z_score:.2f}")
col4.metric("Avg Correlation", f"{metrics['corr_tb']:.3f}")

st.info(f"**Trạng thái:** {results['insight_state']}")
st.write(f"💡 **Phân tích:** {results.get('insight_detail', '')}")
 
st.subheader("1. Phân tích Phổ Trị Riêng (Snap-shot)")
st.pyplot(plot_spectrum(results))

st.subheader("2. Biểu đồ Giám sát Sync-Heat (Time-series)")
history_df = load_data("rolling_metrics.csv")
st.pyplot(plot_sync_heat(history_df))
