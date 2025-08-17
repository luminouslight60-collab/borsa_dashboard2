import time
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------
# Yardımcı Fonksiyonlar
# ----------------------
def compute_ma(df, windows):
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    return df

def compute_rsi(df, period=14):
    close = df["Close"].squeeze()  # Tek boyuta indir
    delta = close.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=df.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, short=12, long=26, signal=9):
    close = df["Close"].squeeze()

    exp1 = close.ewm(span=short, adjust=False).mean().squeeze()
    exp2 = close.ewm(span=long, adjust=False).mean().squeeze()
    macd = (exp1 - exp2).squeeze()
    signal_line = macd.ewm(span=signal, adjust=False).mean().squeeze()
    hist = (macd - signal_line).squeeze()

    return pd.DataFrame({
        "MACD": macd,
        "Signal": signal_line,
        "Hist": hist
    }, index=df.index)


def fetch_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df.dropna()

# ----------------------
# Streamlit Uygulaması
# ----------------------
st.set_page_config(page_title="Canlı Borsa Dashboard", layout="wide")
st.title("📈 Canlı Borsa Dashboard")

with st.sidebar:
    st.header("Ayarlar")
    symbols_text = st.text_input("Hisseler (virgülle ayır):", "ASELS.IS, THYAO.IS")
    period = st.selectbox("Dönem", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","max"], index=2)
    interval = st.selectbox("Zaman Aralığı", ["1m","5m","15m","30m","60m","1d","1wk"], index=3)
    refresh_seconds = st.number_input("Yenileme (saniye)", min_value=15, max_value=600, value=60, step=15)
    st.divider()
    st.subheader("İndikatörler")
    ma10 = st.checkbox("MA10", value=True)
    ma20 = st.checkbox("MA20", value=True)
    ma50 = st.checkbox("MA50", value=False)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12,26,9)", value=True)

symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]
tabs = st.tabs(symbols)

def plot_symbol(tab, symbol):
    with tab:
        df = fetch_data(symbol, period, interval)
        if df.empty:
            st.warning(f"{symbol} için veri alınamadı.")
            return
        ma_windows = []
        if ma10: ma_windows.append(10)
        if ma20: ma_windows.append(20)
        if ma50: ma_windows.append(50)
        if ma_windows:
            df = compute_ma(df, ma_windows)
        rsi = compute_rsi(df) if show_rsi else None
        macd_df = compute_macd(df) if show_macd else None

        # Fiyat grafiği
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df.index, df["Close"], label="Close")
        for w in ma_windows:
            ax.plot(df.index, df[f"MA{w}"], label=f"MA{w}")
        ax.legend()
        st.pyplot(fig)

        if show_rsi:
            fig2, ax2 = plt.subplots(figsize=(12,3))
            ax2.plot(rsi.index, rsi, label="RSI")
            ax2.axhline(70, linestyle="--", color="red")
            ax2.axhline(30, linestyle="--", color="green")
            st.pyplot(fig2)

        if show_macd:
            fig3, ax3 = plt.subplots(figsize=(12,3))
            ax3.plot(macd_df.index, macd_df["MACD"], label="MACD")
            ax3.plot(macd_df.index, macd_df["Signal"], label="Signal")
            ax3.bar(macd_df.index, macd_df["Hist"], alpha=0.3)
            st.pyplot(fig3)

        st.dataframe(df.tail(20))

for i, sym in enumerate(symbols):
    plot_symbol(tabs[i], sym)

st.caption(f"Son güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
time.sleep(int(refresh_seconds))
st.rerun()
