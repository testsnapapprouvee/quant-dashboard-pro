import streamlit as st
from core_dashboard import render_core
from module_loader import load_module

st.set_page_config(layout="wide")

df = render_core()

if df is not None:
    st.sidebar.markdown("## 🧩 Modules")

    if st.sidebar.checkbox("Metrics"):
        mod = load_module("metrics")
        if mod:
            mod.render(df)

    if st.sidebar.checkbox("Drawdown"):
        mod = load_module("drawdown")
        if mod:
            mod.render(df)

    if st.sidebar.checkbox("Monte Carlo"):
        mod = load_module("montecarlo")
        if mod:
            mod.render(df)
