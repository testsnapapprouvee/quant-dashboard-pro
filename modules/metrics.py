import streamlit as st
import numpy as np

def render(df):
    def calc_metrics(series):
        total_ret = series.iloc[-1] / series.iloc[0] - 1
        cagr = (series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1
        drawdown = (series / series.cummax() - 1).min()
        return total_ret, cagr, drawdown

    metrics = {c: calc_metrics(df[c]) for c in df.columns}

    st.write("### 📐 Metrics")
    st.table(
        {k: dict(zip(["Total Return", "CAGR", "Max DD"], v))
         for k, v in metrics.items()}
    )
