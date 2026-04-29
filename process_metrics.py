import pandas as pd
import numpy as np
from main import get_metric

def process_metrics(input_file="dataset.csv", output_file="rolling_metrics.csv", window_size=120, baseline_window=120):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    results = []
    print(f"Calculating rolling lamda_ra...")
    # Tính lamda_ra cho từng vị trí
    for i in range(window_size, len(df)):
        sub = df.iloc[i-window_size : i]
        try:
            res = get_metric(sub)
            results.append({
                "date": df.index[i-1],
                "lamda_ra": res["chỉ số"]["lamda_ra"]
            })
        except:
            continue
            
    res_df = pd.DataFrame(results).set_index("date")

    # Tính Trung bình và Độ lệch chuẩn dựa trên cửa sổ lịch sử (Baseline)
    print(f"Calculating Mu and Sigma...")
    res_df["mu"] = res_df["lamda_ra"].rolling(window=baseline_window).mean()
    res_df["sigma"] = res_df["lamda_ra"].rolling(window=baseline_window).std()
    
    res_df.to_csv(output_file)
    print(f"Saved to {output_file}")

    global_median = res_df["lamda_ra"].median()
    global_mad = (res_df["lamda_ra"] - global_median).abs().median()
    
    print("\n" + "="*30)
    print("GLOBAL STATISTICS (All Data)")
    print(f"Median Lamda_ra: {global_median:.6f}")
    print(f"MAD Lamda_ra:    {global_mad:.6f}")
    print("="*30)

if __name__ == "__main__":
    process_metrics()
