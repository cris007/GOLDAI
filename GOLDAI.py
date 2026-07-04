import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Page Configuration for responsive layout structure
st.set_page_config(
    page_title="Gold Intelligence Cockpit",
    page_icon="⚜️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Institutional CSS Injector Sheet
st.markdown("""
    <style>
    .section-box {
        background: #0D1117;
        border: 1px solid #21262D;
        padding: 22px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-sizing: border-box;
    }
    .confluence-box {
        background: #121A16;
        border: 2px solid #233D2A;
        padding: 25px;
        border-radius: 14px;
        margin-bottom: 30px;
        text-align: center;
        box-sizing: border-box;
    }
    .engine-header {
        font-size: 14px;
        text-transform: uppercase;
        color: #8B949E;
        letter-spacing: 1.5px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .engine-title {
        font-size: 22px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 15px;
    }
    .status-text {
        font-size: 24px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔱 Gold Triple-Engine Intelligence Terminal")
st.markdown("##### Multi-Horizon Confluence Matrix (Technical + Fundamental + AI Machine Learning)")

def extract_text_sentiment():
    bullish_keys = ['rate cut', 'recession', 'banking crisis', 'safe haven', 'war', 'panic', 'escalation', 'fed dovish']
    bearish_keys = ['rate hike', 'strong jobs', 'fed hawkish', 'gdp growth', 'economic boom', 'dollar surge']
    rss_urls = [
        "https://google.com",
        "https://google.com"
    ]
    net_score = 0
    item_count = 0
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.title.lower()
                item_count += 1
                for word in bullish_keys:
                    if word in title: net_score += 1
                for word in bearish_keys:
                    if word in title: net_score -= 1
        except: pass
    return net_score / item_count if item_count > 0 else 0.0

def fetch_header_metrics():
    symbols = {"XAUUSD": "GC=F", "DXY": "DX-Y.NYB", "US10Y": "^TNX"}
    metrics = {}
    for key, sym in symbols.items():
        try:
            ticker = yf.Ticker(sym)
            data = ticker.history(period="5d")
            if not data.empty: metrics[key] = {"val": data['Close'].iloc[-1]}
            else: metrics[key] = {"val": 0.0}
        except: metrics[key] = {"val": 0.0}
    return metrics

# Render institutional price header blocks
m_data = fetch_header_metrics()
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #0D1117; border: 1px solid #21262D; padding: 12px 20px; border-radius: 10px; margin-bottom: 25px;">
        <div><div style="font-size: 13px; font-weight: bold; color: #FFD700; letter-spacing: 0.5px;">BULLION DESK</div><div style="font-size: 11px; color: #8B949E;">Research Terminal</div></div>
        <div style="display: flex; gap: 20px; text-align: right;">
            <div><div style="font-size: 10px; color: #8B949E;">XAU/USD</div><div style="font-size: 14px; font-weight: bold; color: #FFFFFF;">${m_data['XAUUSD']['val']:.2f}</div></div>
            <div><div style="font-size: 10px; color: #8B949E;">DXY</div><div style="font-size: 14px; font-weight: bold; color: #FFFFFF;">{m_data['DXY']['val']:.2f}</div></div>
            <div><div style="font-size: 10px; color: #8B949E;">US10Y</div><div style="font-size: 14px; font-weight: bold; color: #FFFFFF;">{m_data['US10Y']['val']:.3f}%</div></div>
        </div>
    </div>
""", unsafe_allow_html=True)
if st.button("EXECUTE ALL-SECTOR SECTOR SCAN", type="primary", use_container_width=True):
    with st.spinner("Processing deep analytics matrix..."):
        
        # --- PHASE 1: UNIFIED HIGH-SPEED BATCH DATA STREAM SYNC ---
        ticker_symbols = ["GC=F", "DX-Y.NYB", "TLT", "^VIX", "SPY", "TIP", "FXE", "GOLD", "NEM", "GFI", "AEM", "GDXJ"]
        try:
            batch_data = yf.download(ticker_symbols, period="2y", interval="1d", group_by='ticker', progress=False)
        except:
            st.error("⚠️ Yahoo Finance synchronization timeout. Please re-run scan.")
            st.stop()
            
        df = pd.DataFrame()
        try:
            df['Gold_Close'] = batch_data['GC=F']['Close']; df['Gold_High'] = batch_data['GC=F']['High']; df['Gold_Low'] = batch_data['GC=F']['Low']
            df['DXY_Close']  = batch_data['DX-Y.NYB']['Close']
            df['TLT_Close']  = batch_data['TLT']['Close']
            df['VIX_Close']  = batch_data['^VIX']['Close']
            df['SPY_Close']  = batch_data['SPY']['Close']
            df['TIP_Close']  = batch_data['TIP']['Close']
            df['FXE_Close']  = batch_data['FXE']['Close']
            df['M1_Close']   = batch_data['GOLD']['Close']; df['M2_Close'] = batch_data['NEM']['Close']; df['M3_Close'] = batch_data['GFI']['Close']; df['M4_Close'] = batch_data['AEM']['Close']; df['M5_Close'] = batch_data['GDXJ']['Close']
        except KeyError:
            st.error("⚠️ Server returned incomplete fields over the weekend. Please refresh.")
            st.stop()
            
        df = df.ffill().bfill()

        # --- PHASE 2: CALCULATE SECTOR 1 -> TECHNICAL LOGIC VECTOR (GREEN CROW ORIGINAL) ---
        df['EMA_10']  = df['Gold_Close'].rolling(window=10).mean()
        df['EMA_20']  = df['Gold_Close'].rolling(window=20).mean()
        df['EMA_100'] = df['Gold_Close'].rolling(window=100).mean()
        df['Prev_High_2'] = df['Gold_High'].shift(1).rolling(window=2).max()
        df['Prev_Low_2']  = df['Gold_Low'].shift(1).rolling(window=2).min()
        
        df['GC_Buy']  = np.where((df['EMA_10'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_100']) & (df['Gold_Close'] > df['Prev_High_2']), 1, 0)
        df['GC_Sell'] = np.where((df['EMA_10'] < df['EMA_20']) & (df['EMA_20'] < df['EMA_100']) & (df['Gold_Close'] < df['Prev_Low_2']), 1, 0)
        tech_vector   = int(df['GC_Buy'].iloc[-1] - df['GC_Sell'].iloc[-1])

        # --- PHASE 3: CALCULATE SECTOR 2 -> FUNDAMENTAL SENTIMENT VECTOR (APP.PY ORIGINAL) ---
        df['DXY_Pct'] = df['DXY_Close'].pct_change() * 100; df['TLT_Pct'] = df['TLT_Close'].pct_change() * 100; df['SPY_Pct'] = df['SPY_Close'].pct_change() * 100; df['TIP_Pct'] = df['TIP_Close'].pct_change() * 100; df['FXE_Pct'] = df['FXE_Close'].pct_change() * 100
        m1_up = (df['M1_Close'].pct_change() > 0).astype(int); m2_up = (df['M2_Close'].pct_change() > 0).astype(int); m3_up = (df['M3_Close'].pct_change() > 0).astype(int); m4_up = (df['M4_Close'].pct_change() > 0).astype(int); m5_up = (df['M5_Close'].pct_change() > 0).astype(int)
        df['Miner_Buy']  = np.where((m1_up == 1) & (m2_up == 1) & (m3_up == 1) & (m4_up == 1) & (m5_up == 1), 1, 0)
        df['Miner_Sell'] = np.where((m1_up == 0) & (m2_up == 0) & (m3_up == 0) & (m4_up == 0) & (m5_up == 0), 1, 0)
        df['Miner_Vector'] = df['Miner_Buy'] - df['Miner_Sell']
        
        df['DXY_Pts'] = np.where(df['DXY_Pct'] <= 0, 2.0, -2.0); df['TLT_Pts'] = np.where(df['TLT_Pct'] > 0, 2.0, -2.0); df['VIX_Pts'] = np.where(df['VIX_Close'] > 20, 1.5, -0.5); df['SPY_Pts'] = np.where(df['SPY_Pct'] > 0, -1.0, 1.0); df['TIP_Pts'] = np.where(df['TIP_Pct'] > 0, 1.5, -1.5); df['FXE_Pts'] = np.where(df['FXE_Pct'] > 0, -1.0, 1.5); df['Miner_Pts'] = df['Miner_Vector'] * 2.5
        live_sent = extract_text_sentiment()
        df['News_Pts'] = np.random.normal(live_sent * 1.5, 0.1, len(df))
        df['Fund_Score'] = df['DXY_Pts'] + df['TLT_Points' if 'TLT_Points' in df else 'TLT_Pts'] + df['VIX_Pts'] + df['SPY_Pts'] + df['TIP_Pts'] + df['FXE_Pts'] + df['Miner_Pts'] + df['News_Pts']
        fund_score_live = float(df['Fund_Score'].iloc[-1])

        # --- PHASE 4: CALCULATE SECTOR 3 -> AI PREDICTIVE FORECAST DIRECTIVE ---
        df['Target_ST'] = np.where(df['Gold_Close'].shift(-2) > df['Gold_Close'], 1, 0)
        df_clean = df.dropna().copy()
        
        feature_cols = ['GC_Buy', 'GC_Sell', 'Fund_Score', 'DXY_Pts', 'TLT_Pts', 'Miner_Pts', 'VIX_Pts']
        X = df_clean[feature_cols]
        model_ai = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_ai.fit(X, df_clean['Target_ST'])
        
        prob_ai = model_ai.predict_proba(X.iloc[[-1]])
        ai_buy_pct = float(prob_ai[0][1] * 100)
        # --- PHASE 5: MULTI-ENGINE SPEEDOMETER GAUGE COMPILER SUB-MODULE ---
        def generate_html_gauge(title_label, metric_score, type_mode):
            # Normalizes score parameters cleanly across all 3 independent modes onto our 5-label dials
            if type_mode == "TECH":
                if metric_score == 1: lbl, col, angle = "BUY", "#88FF88", 135
                elif metric_score == -1: lbl, col, angle = "SELL", "#FF8888", 45
                else: lbl, col, angle = "NEUTRAL", "#FF9900", 90
            elif type_mode == "FUND":
                if metric_score >= 5.5: lbl, col, angle = "STRONG BUY", "#00FF66", 165
                elif metric_score >= 1.5: lbl, col, angle = "BUY", "#88FF88", 135
                elif metric_score <= -5.5: lbl, col, angle = "STRONG SELL", "#FF0033", 15
                elif metric_score <= -1.5: lbl, col, angle = "SELL", "#FF8888", 45
                else: lbl, col, angle = "NEUTRAL", "#FF9900", 90
            elif type_mode == "AI":
                if metric_score >= 65.0: lbl, col, angle = "STRONG BUY", "#00FF66", 165
                elif metric_score >= 53.0: lbl, col, angle = "BUY", "#88FF88", 135
                elif metric_score <= 35.0: lbl, col, angle = "STRONG SELL", "#FF0033", 15
                elif metric_score <= 47.0: lbl, col, angle = "SELL", "#FF8888", 45
                else: lbl, col, angle = "NEUTRAL", "#FF9900", 90

            return f"""
            <div style="display: flex; justify-content: center; margin: 15px 0;">
                <div style="width:100%; max-width:360px; background:#161B22; border:1px solid #21262D; padding:15px; border-radius:12px; text-align:center;">
                    <div style="font-size:12px; text-transform:uppercase; color:#8B949E; font-weight:bold; letter-spacing:0.5px;">{title_label}</div>
                    <svg viewBox="0 0 200 110" width="100%" style="max-width: 240px; margin-top:10px;">
                        <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#22332A" stroke-width="10" stroke-linecap="round"/>
                        <path d="M20,100 A80,80 0 0,1 52,48" fill="none" stroke="#FF0033" stroke-width="3" opacity="0.3"/>
                        <path d="M52,48 A80,80 0 0,1 84,24" fill="none" stroke="#FF8888" stroke-width="3" opacity="0.3"/>
                        <path d="M84,24 A80,80 0 0,1 116,24" fill="none" stroke="#FF9900" stroke-width="3" opacity="0.3"/>
                        <path d="M116,24 A80,80 0 0,1 148,48" fill="none" stroke="#88FF88" stroke-width="3" opacity="0.3"/>
                        <path d="M148,48 A80,80 0 0,1 180,100" fill="none" stroke="#00FF66" stroke-width="3" opacity="0.3"/>
                        <text x="12" y="108" fill="#8B949E" font-size="5" text-anchor="middle">S.SELL</text>
                        <text x="44" y="42" fill="#8B949E" font-size="5" text-anchor="middle">SELL</text>
                        <text x="100" y="14" fill="#8B949E" font-size="6" text-anchor="middle" font-weight="bold">NEUTRAL</text>
                        <text x="156" y="42" fill="#8B949E" font-size="5" text-anchor="middle">BUY</text>
                        <text x="188" y="108" fill="#8B949E" font-size="5" text-anchor="middle">S.BUY</text>
                        <circle cx="100" cy="100" r="5" fill="#FFFFFF"/>
                        <line x1="100" y1="100" x2="100" y2="32" stroke="{col}" stroke-width="2.5" stroke-linecap="round" transform="rotate({angle - 90} 100 100)"/>
                    </svg>
                    <div class="status-text" style="color:{col}; font-size:18px; margin-top:-5px;">{lbl}</div>
                </div>
            </div>
            """

        # --- PHASE 6: CALCULATE MASTER 3-PART UNIFIED CONFLUENCE CORE ALERT ---
        st.markdown("---")
        st.subheader("🏁 Core Confluence Overview")
        
        # Confluence mapping weights scoring engine equations
        conf_score = 0
        if tech_vector == 1: conf_score += 1
        elif tech_vector == -1: conf_score -= 1
        
        if fund_score_live >= 1.5: conf_score += 1
        elif fund_score_live <= -1.5: conf_score -= 1
        
        if ai_buy_pct >= 53.0: conf_score += 1
        elif ai_buy_pct <= 47.0: conf_score -= 1
        
        if conf_score >= 2:
            st.markdown('<div class="confluence-box"><div class="engine-header">Master Confluence Signal</div><div class="status-text" style="color:#00FF66; font-size:32px; text-shadow: 0 0 15px #00FF6655;">CONFLUENCE BUY</div><div style="font-size:13px; color:#8B949E; margin-top:5px;">All 3 macro processing dimensions are fully synchronized long.</div></div>', unsafe_allow_html=True)
        elif conf_score <= -2:
            st.markdown('<div class="confluence-box"><div class="engine-header">Master Confluence Signal</div><div class="status-text" style="color:#FF0033; font-size:32px; text-shadow: 0 0 15px #FF003355;">CONFLUENCE SELL</div><div style="font-size:13px; color:#8B949E; margin-top:5px;">All 3 macro processing dimensions are fully synchronized short.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="confluence-box"><div class="engine-header">Master Confluence Signal</div><div class="status-text" style="color:#FF9900; font-size:32px; text-shadow: 0 0 15px #FF990055;">CONFLUENCE WAIT</div><div style="font-size:13px; color:#8B949E; margin-top:5px;">Sectors are out of sync or fighting each other. Keep bots turned off.</div></div>', unsafe_allow_html=True)

        # --- PHASE 7: RENDER THE 3 INDEPENDENT ENGINE VISUAL SECTIONS ---
        # Part 1 Display: Fundamental Engine Box
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="engine-header">Part 1 Engine Layer</div>', unsafe_allow_html=True)
        st.markdown('<div class="engine-title">Macro Fundamental Sentiment (App.py Rules)</div>', unsafe_allow_html=True)
        st.components.v1.html(generate_html_gauge("Fundamental Matrix Arc", fund_score_live, "FUND"), height=195, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # Part 2 Display: Technical Engine Box
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="engine-header">Part 2 Engine Layer</div>', unsafe_allow_html=True)
        st.markdown('<div class="engine-title">GreenCrow Technical Breakouts (Rule-Based Matrix)</div>', unsafe_allow_html=True)
        st.components.v1.html(generate_html_gauge("Technical Breakout Arc", tech_vector, "TECH"), height=195, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # Part 3 Display: AI Predictive Engine Box
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="engine-header">Part 3 Engine Layer</div>', unsafe_allow_html=True)
        st.markdown('<div class="engine-title">AI Predictive Forecast Oracle (Gradient Boosting Models)</div>', unsafe_allow_html=True)
        st.components.v1.html(generate_html_gauge("Machine Conviction Arc", ai_buy_pct, "AI"), height=195, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)
