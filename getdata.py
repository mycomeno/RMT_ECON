from vnstock import Quote
import pandas as pd
from datetime import datetime
import time
def get_data ( ):
    
    tickers = [
        'HPG', 'VNM', 'VIC', 'VCB', 'FPT', 'GAS', 'MSN', 'MWG', 'TCB', 'VPB',
        'ACB', 'MBB', 'VHM', 'STB', 'HDB', 'VRE', 'TPB', 'VJC', 'SSI', 'PLX',
        'POW', 'SAB', 'GVR', 'BID', 'BVH', 'CTG', 'VIB', 'SHB', 'BCM', 'SSB'

    ]
    print (f"Fetching VN30 components for RMT analysis...")
    
    all_prices = {}
    start = '2015-01-01'
    end = '2024-01-01'
    for i, ticker in enumerate(tickers):
        q = Quote (symbol = ticker, source  = 'VCI')
        df = q.history(start = start, end = end)
        if 'time' in df.columns:
            df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        all_prices[ticker] = df['close']
        time.sleep(1)
    
    if all_prices:
        final_df = pd.DataFrame(all_prices)

        final_df = final_df.ffill().bfill()

        final_df.to_csv("dataset.csv")
        print("-"*20)
        print(f"Thành Công")
    else: 
        print("Tải thất bại")

if __name__ == "__main__":
    get_data()