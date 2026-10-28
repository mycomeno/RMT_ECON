import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import get_metric, median_ra_global, sigma_robust

st.set_page_config(page_title="RMT Market Dashboard", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def plot_spectrum(results):
    eigvals    = results["eigvals"]
    lamda_cong = results["lamda_cong"]
    lamda_tru  = results["lamda_tru"]
    q          = results["q"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(eigvals, bins=40, density=True, alpha=0.7, color="#4682b4", edgecolor="black", linewidth=0.5)

    x  = np.linspace(max(lamda_tru, 1e-6), lamda_cong, 300)
    mp = np.sqrt(np.maximum((lamda_cong - x) * (x - lamda_tru), 0)) / (2 * np.pi * q * x)
    ax.plot(x, mp, color="red", lw=2, label="Marchenko-Pastur")
    ax.axvline(lamda_cong, color="black", ls="--", lw=1.5, label="λ+ (Ngưỡng nhiễu)")
    ax.set_title("Eigenvalue Spectrum (Phổ Trị Riêng)")
    ax.set_xlabel("Trị riêng (Eigenvalue)")
    ax.set_ylabel("Mật độ xác suất (Density)")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sync_heat(rolling_df):
    """Sync-Heat chart: Robust Z-score theo thời gian + 4 vùng cảnh báo rủi ro."""
    df = rolling_df.copy()

    if "mu" in df.columns:
        df = df[df["mu"].notna()].copy()

    df["z_score"] = (df["lamda_ra"] - median_ra_global) / sigma_robust

    y_max = max(4.0, float(df["z_score"].max()) + 1.0)
    y_min = min(-2.0, float(df["z_score"].min()) - 0.5)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_ylim(y_min, y_max)

    # 4 vùng trạng thái rủi ro
    ax.axhspan(y_min, 1.0, color="green",  alpha=0.08, lw=0, label="Bình thường (<1.0)")
    ax.axhspan(1.0,   1.5, color="yellow", alpha=0.12, lw=0, label="Cảnh báo (1.0–1.5)")
    ax.axhspan(1.5,   3.0, color="orange", alpha=0.12, lw=0, label="Rủi ro (1.5–3.0)")
    ax.axhspan(3.0, y_max, color="red",    alpha=0.12, lw=0, label="Cực đoan (>3.0)")

    # Đường Z-Score chính
    ax.plot(df.index, df["z_score"], color="#1f77b4", lw=1.8, label="Robust Z-Score")

    # Đường Signal Gap Score (nếu có trong dữ liệu)
    if "signal_gap_score" in df.columns and df["signal_gap_score"].notna().any():
        ax.plot(df.index, df["signal_gap_score"], label="Signal Gap Score", color="#ff7f0e", alpha=0.7, ls=":")

    # Đường Rolling Mean (Z)
    if "mu" in df.columns:
        mu_z = (df["mu"] - median_ra_global) / sigma_robust
        ax.plot(df.index, mu_z, color="purple", lw=1.2, ls="--", alpha=0.7, label="Rolling Mean (Z)")

    # Ngưỡng tham chiếu
    ax.axhline(1.5, color="red",    ls=":", lw=1.2, alpha=0.7, label="Z = 1.5 (Ngưỡng Rủi ro)")
    ax.axhline(1.0, color="orange", ls=":", lw=1.2, alpha=0.6, label="Z = 1.0 (Ngưỡng Cảnh báo)")

    # Đánh dấu các giai đoạn thị trường đặc biệt
    regimes = [
        ("COVID-19", "2020-03-01", "2020-05-31", "gray"),
        ("Bull Run", "2021-01-01", "2022-04-30", "blue"),
        ("Downturn", "2022-05-01", "2022-11-30", "darkorange"),
    ]
    for lbl, start, end, color in regimes:
        s_dt = pd.to_datetime(start)
        e_dt = pd.to_datetime(end)
        mask = (df.index >= s_dt) & (df.index <= e_dt)
        if mask.any():
            ax.axvspan(s_dt, e_dt, color=color, alpha=0.07, lw=0)
            mid = s_dt + (e_dt - s_dt) / 2
            ax.text(mid, y_max * 0.88, lbl,
                    ha="center", fontsize=8, color=color, weight="bold")

    ax.set_title("Sync-Heat: Giám sát RMT Z-Score theo thời gian")
    ax.set_ylabel("Standard Deviations (Robust Z)")
    ax.legend(loc="upper left", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ------------------ GIAO DIỆN CHÍNH ------------------
st.title("RMT Market Dashboard")

df = load_data("dataset.csv")

window = st.sidebar.slider("Cửa sổ phân tích (Window)", 60, 250, 120)

if "idx" not in st.session_state:
    st.session_state.idx = len(df) - window

available_dates = df.index[:-window]
if st.session_state.idx >= len(available_dates):
    st.session_state.idx = len(available_dates) - 1

start_dt = st.sidebar.slider(
    "Mốc thời gian (Timeline)",
    min_value=available_dates[0].to_pydatetime(),
    max_value=available_dates[-1].to_pydatetime(),
    value=available_dates[st.session_state.idx].to_pydatetime(),
    format="MM/YYYY",
)

start = df.index.get_indexer([pd.Timestamp(start_dt)], method="pad")[0]
st.session_state.idx = start

c_start = df.index[start]
c_end   = df.index[start + window - 1]
st.sidebar.info(
    f"**Khoảng thời gian:**  \n{c_start.strftime('%d/%m/%Y')} — {c_end.strftime('%d/%m/%Y')}"
)

# Tính toán chỉ số cho cửa sổ hiện tại
sub_df   = df.iloc[start : start + window]
results  = get_metric(sub_df)
metrics  = results["chỉ số"]
lamda_ra = metrics["lamda_ra"]
num_sig  = metrics["num_sig"]
z_score  = results["z_score_global"]

# Hiển thị 4 thẻ chỉ số
col1, col2, col3, col4 = st.columns(4)
col1.metric("λ1 Sync", f"{lamda_ra * 100:.1f}%")
col2.metric("Signals", int(num_sig))
col3.metric("Robust Z-Score (MAD)", f"{z_score:.2f}")
col4.metric("Avg Correlation", f"{metrics['corr_tb']:.3f}")

# Trạng thái và Phân tích thị trường
st.info(f"**Trạng thái:** {results['insight_state']}")
st.write(f"💡 **Phân tích:** {results.get('insight', results.get('insight_detail', ''))}")

# Biểu đồ 1: Phổ trị riêng
st.subheader("1. Phân tích Phổ Trị Riêng (Eigenvalue Spectrum)")
st.pyplot(plot_spectrum(results))

# Biểu đồ 2: Biểu đồ Sync-Heat chuỗi thời gian
st.subheader("2. Biểu đồ Giám sát Sync-Heat (Time-series)")
rolling_path = "rolling_metrics.csv"
if os.path.exists(rolling_path):
    history_df = load_data(rolling_path)
    st.pyplot(plot_sync_heat(history_df))
else:
    st.warning(
        "⚠️ Chưa có file `rolling_metrics.csv`. "
        "Vui lòng chạy lệnh `python process_metrics.py` để tính rolling metrics trước."
    )
