# modules/data_engine.py
import yfinance as yf
import pandas as pd

def get_data(tickers, start, end):
    """
    Récupère les prix ajustés des tickers donnés sur la période start-end.
    Renvoie un DataFrame avec les deux colonnes 'X2' et 'X1'.
    """
    if not tickers or len(tickers) < 2:
        return pd.DataFrame()
    
    price_map = {}
    for t in tickers[:2]:  # juste les deux premiers
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                continue
            price_map[t] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        except:
            continue

    if len(price_map) < 2:
        return pd.DataFrame()
    
    df_final = pd.concat(price_map.values(), axis=1)
    df_final.columns = ['X2', 'X1']
    return df_final.ffill().dropna()
