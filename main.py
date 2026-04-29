import pandas as pd
import numpy as np

median_ra_global = 0.373226
sigma_robust = 0.122008 

def get_metric(price_df):
   
   returns = (price_df / price_df.shift(1) - 1).dropna()

   returns = returns.loc[:, returns.std() > 0]
   
   if returns.empty:
       raise ValueError("Không có đủ dữ liệu biến động trong giai đoạn này.")

   returns_std = ( returns - returns.mean()) / returns.std()
   C = np.corrcoef(returns_std.T)
   eigvals, _= np.linalg.eigh(C)
   eigvals = np.sort(eigvals)[::-1]

   N = len(eigvals)
   T = len(returns)
   q = N/T
   sigma = 1.0
   lamda_cong = sigma * (1 + np.sqrt(q))**2
   lamda_tru = sigma * ( 1 -np.sqrt(q))**2
    
   lamda_ra = eigvals[0] / np.sum(eigvals)
   num_sig = np.sum(eigvals > lamda_cong)
  
   vals = []
   for i in range (C.shape[0]):
       for j in range (i+1, C.shape[1]):
           vals.append(C[i,j])
   corr_tb = sum(vals) / len(vals)


   z_score_robust = (lamda_ra - median_ra_global) / sigma_robust
 
   noise_mask = eigvals <= lamda_cong
   signal_mask = eigvals > lamda_cong
      
   noise_ra = noise_mask.sum() / len(eigvals)
   signal_ra = signal_mask.sum() / len(eigvals)
 
   signal_gap = eigvals[0] - lamda_cong
   signal_gap_score = signal_gap / (lamda_cong - lamda_tru)
    
 
   state = "BÌNH THƯỜNG"
   insight = "Thị trường vận động tự nhiên"
   if z_score_robust < 1.5 and z_score_robust >= 1:
         state = "CẢNH BÁO"
         insight = "Mức độ tương quan tăng nhanh hơn. Các cổ phiếu bắt đầu di chuyển cùng chiều bất thường"
   elif z_score_robust >= 1.5:
       state  = "RỦI RO"
       insight = "Biến động cực đoan. Thị trường phản ứng cực đoan với một thông tin chung"
   elif  z_score_robust<= -1.5:
       state = "PHÂN HÓA NỘI TẠI"
       insight = "Tính dẫn dắt giảm mạnh"
      
 
   return {
         "eigvals": eigvals,
         "lamda_tru": lamda_tru,
         "lamda_cong": lamda_cong,
         "z_score_global": z_score_robust,
        "chỉ số": {
            "lamda_ra": lamda_ra,
            "num_sig": num_sig,
            "corr_tb": corr_tb,
            "noise_ra": noise_ra,
            "signal_ra": signal_ra,
            "signal_gap": signal_gap,
            "signal_gap_score": signal_gap_score
        },
        "insight_state": state,
        "insight_detail": insight,
        "insight": insight,
        "q": q,
        "N": N,
        "T": T
    }
if __name__ == "__main__":
   df = pd.read_csv("dataset.csv", index_col = 0, parse_dates = True)
   results = get_metric(df)
