import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Page Configuration for absolute optimal mobile layout responsive display
st.set_page_config(
    page_title="Gold Predictive AI Oracle",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Institutional CSS Injector Layer
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
        font-size: 13px;
        text-transform: uppercase;
        color: #8B949E;
        letter-spacing: 1.5px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .engine-title {
        font-size: 20px;
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
# Processing Trigger Button
if st.button("RUN DEEP HYBRID ENSEMBLE CALCULATOR", type="primary", use_container_width=True):
    with st.spinner("Processing deep analytics matrix..."):
        
        # --- PHASE 1: UNIFIED HIGH-SPEED BATCH DATA STREAM SYNC ---
        ticker_symbols = ["GC=F", "DX-Y.NYB", "TLT", "^TNX", "^VIX", "SPY", "TIP", "XAUEUR=X", "GDX", "GDXJ", "GOLD", "NEM", "GFI", "AEM"]
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
            df['TNX_Close']  = batch_data['^TNX']['Close'] # Nominal Yield
            df['VIX_Close']  = batch_data['^VIX']['Close']
            df['SPY_Close']  = batch_data['SPY']['Close']
            df['TIP_Close']  = batch_data['TIP']['Close']  # Real Yield Proxy
            df['XAE_Close']  = batch_data['XAUEUR=X']['Close']
            df['GDX_Close']  = batch_data['GDX']['Close']
            df['GDXJ_Close'] = batch_data['GDXJ']['Close']
            df['M1_Close']   = batch_data['GOLD']['Close']; df['M2_Close'] = batch_data['NEM']['Close']; df['M3_Close'] = batch_data['GFI']['Close']; df['M4_Close'] = batch_data['AEM']['Close']
        except KeyError:
            st.error("⚠️ Server returned incomplete fields over the weekend. Please refresh.")
            st.stop()
            
        # FIXED CRITICAL DATA DROP VALUE ERROR: Gap-fill empty cells FIRST before processing indicators or rolling metrics
        df = df.ffill().bfill()

        # --- PHASE 2: GENERATE THE METRICS SCORECARD TABLE ROWS ---
        table_rows = []
        assets_mapping = {
            "DXY Index (US Dollar)": "DXY_Close",
            "US 10Y Treasury Nominal Yield (^TNX)": "TNX_Close",
            "US 10Y Treasury Real Yield Proxy (TIP)": "TIP_Close",
            "Volatility Fear Gauge (VIX)": "VIX_Close",
            "Stock Market Index (SPY)": "SPY_Close",
            "Cross-Asset Safe Haven (XAU/EUR)": "XAE_Close",
            "VanEck Gold Miners ETF (GDX)": "GDX_Close",
            "Junior Gold Miners ETF (GDXJ)": "GDXJ_Close",
            "Barrick Gold Corp (GOLD)": "M1_Close",
            "Newmont Corp (NEM)": "M2_Close",
            "Gold Fields Ltd (GFI)": "M3_Close",
            "Agnico Eagle Mines (AEM)": "M4_Close"
        }
        
        for display_name, col_key in assets_mapping.items():
            val_latest = df[col_key].iloc[-1]
            val_previous = df[col_key].iloc[-2]
            point_delta = val_latest - val_previous
            percentage_delta = (point_delta / val_previous) * 100
            
            table_rows.append({
                "Asset Component Name": display_name,
                "Current Price/Level": f"{val_latest:.3f}%" if "TNX" in col_key else f"{val_latest:.2f}" if "VIX" in col_key or "DXY" in col_key else f"${val_latest:.2f}",
                "Change (Points)": point_delta,
                "Change (%)": percentage_delta
            })
            
        df_scorecard = pd.DataFrame(table_rows)

        # --- PHASE 3: CALCULATE STRATEGY PARAMETERS AND CRITERIA SCORES ---
        df['EMA_10']  = df['Gold_Close'].rolling(window=10).mean()
        df['EMA_20']  = df['Gold_Close'].rolling(window=20).mean()
        df['EMA_100'] = df['Gold_Close'].rolling(window=100).mean()
        df['Prev_High_2'] = df['Gold_High'].shift(1).rolling(window=2).max()
        df['Prev_Low_2']  = df['Gold_Low'].shift(1).rolling(window=2).min()
        
        df['GC_Buy']  = np.where((df['EMA_10'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_100']) & (df['Gold_Close'] > df['Prev_High_2']), 1, 0)
        df['GC_Sell'] = np.where((df['EMA_10'] < df['EMA_20']) & (df['EMA_20'] < df['EMA_100']) & (df['Gold_Close'] < df['Prev_Low_2']), 1, 0)
        tech_vector   = int(df['GC_Buy'].iloc[-1] - df['GC_Sell'].iloc[-1])

        df['DXY_Pct'] = df['DXY_Close'].pct_change() * 100; df['TLT_Pct'] = df['TLT_Close'].pct_change() * 100; df['SPY_Pct'] = df['SPY_Close'].pct_change() * 100; df['TIP_Pct'] = df['TIP_Close'].pct_change() * 100; df['XAE_Pct'] = df['XAE_Close'].pct_change() * 100
        m1_up = (df['M1_Close'].pct_change() > 0).astype(int); m2_up = (df['M2_Close'].pct_change() > 0).astype(int); m3_up = (df['M3_Close'].pct_change() > 0).astype(int); m4_up = (df['M4_Close'].pct_change() > 0).astype(int)
        df['Miner_Buy']  = np.where((m1_up == 1) & (m2_up == 1) & (m3_up == 1) & (m4_up == 1), 1, 0)
        df['Miner_Sell'] = np.where((m1_up == 0) & (m2_up == 0) & (m3_up == 0) & (m4_up == 0), 1, 0)
        df['Miner_Vector'] = df['Miner_Buy'] - df['Miner_Sell']
        
        df['DXY_Pts'] = np.where(df['DXY_Pct'] <= 0, 2.0, -2.0); df['TLT_Pts'] = np.where(df['TLT_Pct'] > 0, 2.0, -2.0); df['VIX_Pts'] = np.where(df['VIX_Close'] > 20, 1.5, -0.5); df['SPY_Pts'] = np.where(df['SPY_Pct'] > 0, -1.0, 1.0); df['TIP_Pts'] = np.where(df['TIP_Pct'] > 0, 1.5, -1.5); df['XAE_Pts'] = np.where(df['XAE_Pct'] > 0, 1.5, -1.0); df['Miner_Pts'] = df['Miner_Vector'] * 2.5
        live_sent = extract_text_sentiment()
        df['News_Pts'] = np.random.normal(live_sent * 1.5, 0.1, len(df))
        df['Fund_Score'] = df['DXY_Pts'] + df['TLT_Pts'] + df['VIX_Pts'] + df['SPY_Pts'] + df['TIP_Pts'] + df['XAE_Pts'] + df['Miner_Pts'] + df['News_Pts']
        fund_score_live = float(df['Fund_Score'].iloc[-1])
        # --- PHASE 4: FIXED SAFE-GUARDED MACHINE LEARNING HORIZONS ---
        df['Target_ST'] = np.where(df['Gold_Close'].shift(-2) > df['Gold_Close'], 1, 0)   # Short Term
        df['Target_MT'] = np.where(df['Gold_Close'].shift(-14) > df['Gold_Close'], 1, 0)  # Medium Term
        df['Target_LT'] = np.where(df['Gold_Close'].shift(-45) > df['Gold_Close'], 1, 0)  # Long Term
        
        # Safe Execution Gate: Dropna is executed ONLY at the very end after rolling columns are filled
        df_clean = df.dropna().copy()
        
        # SYSTEM ARMOR CHECK: Guarantees the classifiers never run into an empty array if data feeds lag
        if df_clean.empty or len(df_clean) < 10:
            ai_st_pct, ai_mt_pct, ai_lt_pct = 50.0, 50.0, 50.0 # Return fallback balanced neutral states
        else:
            feature_cols = ['GC_Buy', 'GC_Sell', 'Fund_Score', 'DXY_Pts', 'TLT_Pts', 'Miner_Pts', 'VIX_Pts']
            X = df_clean[feature_cols]
            
            model_st = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X, df_clean['Target_ST'])
            model_mt = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X, df_clean['Target_MT'])
            model_lt = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X, df_clean['Target_LT'])
            
            # Extract scalars securely using index cell slices to eliminate runtime warnings
            ai_st_pct = float(model_st.predict_proba(X.iloc[[-1]])[0][1] * 100)
            ai_mt_pct = float(model_mt.predict_proba(X.iloc[[-1]])[0][1] * 100)
            ai_lt_pct = float(model_lt.predict_proba(X.iloc[[-1]])[0][1] * 100)

        # --- PHASE 5: RE-ENGINEERED SVG GAUGE GENERATOR COMPILER ---
        def generate_html_gauge(title_label, metric_score, type_mode):
            if type_mode == "TECH":
                if metric_score == 1: lbl, col, angle = "STRONG BUY", "#00FF66", 155
                elif metric_score == -1: lbl, col, angle = "STRONG SELL", "#FF0033", 25
                else: lbl, col, angle = "NEUTRAL", "#FF9900", 90
            elif type_mode == "FUND":
                if metric_score >= 5.5: lbl, col, angle = "STRONG BUY", "#00FF66", 155
                elif metric_score >= 1.5: lbl, col, angle = "BUY", "#88FF88", 120
                elif metric_score <= -5.5: lbl, col, angle = "STRONG SELL", "#FF0033", 25
                elif metric_score <= -1.5: lbl, col, angle = "SELL", "#FF8888", 60
                else: lbl, col, angle = "NEUTRAL", "#FF9900", 90
            elif type_mode == "AI":
                if metric_score >= 65.0: lbl, col, angle = "STRONG BUY", "#00FF66", 155
                elif metric_score >= 53.0: lbl, col, angle = "BUY", "#88FF88", 120
                elif metric_score <= 35.0: lbl, col, angle = "STRONG SELL", "#FF0033", 25
                elif metric_score <= 47.0: lbl, col, angle = "SELL", "#FF8888", 60
                else: lbl, col, angle = "NEUTRAL", "#FF9900", 90

            return f"""
            <div style="display: flex; justify-content: center; align-items: center; background: #161B22; border: 1px solid #21262D; padding: 25px; border-radius: 12px; font-family: Arial, sans-serif; box-sizing: border-box;">
                <div style="width: 100%; max-width: 380px; text-align: center;">
                    <div style="font-size: 13px; text-transform: uppercase; color: #8B949E; font-weight: bold; letter-spacing: 1px;">{title_label}</div>
                    <svg viewBox="0 0 200 130" width="100%" height="100%" style="max-width: 280px; margin-top: 15px; overflow: visible;">
                        <path d="M20,110 A80,80 0 0,1 180,110" fill="none" stroke="#22332A" stroke-width="14" stroke-linecap="round"/>
                        <path d="M20,110 A80,80 0 0,1 60,53" fill="none" stroke="#FF0033" stroke-width="5" opacity="0.4"/>
                        <path d="M60,53 A80,80 0 0,1 90,34" fill="none" stroke="#FF8888" stroke-width="5" opacity="0.4"/>
                        <path d="M90,34 A80,80 0 0,1 110,34" fill="none" stroke="#FF9900" stroke-width="5" opacity="0.4"/>
                        <path d="M110,34 A80,80 0 0,1 140,53" fill="none" stroke="#88FF88" stroke-width="5" opacity="0.4"/>
                        <path d="M140,53 A80,80 0 0,1 180,110" fill="none" stroke="#00FF66" stroke-width="5" opacity="0.4"/>
                        <text x="5" y="122" fill="#8B949E" font-size="7" font-weight="bold" text-anchor="start">S.SELL</text>
                        <text x="44" y="46" fill="#8B949E" font-size="7" font-weight="bold" text-anchor="middle">SELL</text>
                        <text x="100" y="20" fill="#8B949E" font-size="8" font-weight="bold" text-anchor="middle">NEUTRAL</text>
                        <text x="156" y="46" fill="#8B949E" font-size="7" font-weight="bold" text-anchor="middle">BUY</text>
                        <text x="195" y="122" fill="#8B949E" font-size="7" font-weight="bold" text-anchor="end">S.BUY</text>
                        <circle cx="100" cy="110" r="6" fill="#FFFFFF" stroke="#121A16" stroke-width="2"/>
                        <line x1="100" y1="110" x2="100" y2="36" stroke="{col}" stroke-width="3" stroke-linecap="round" transform="rotate({angle - 90} 100 110)" style="transition: transform 0.6s ease-in-out; filter: drop-shadow(0px 0px 4px {col}aa);"/>
                    </svg>
                    <div style="font-size: 24px; font-weight: bold; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 5px; color: {col}; text-shadow: 0 0 10px {col}55;">{lbl}</div>
                </div>
            </div>
            """

        # --- PHASE 6: RENDER INTERACTIVE TIME HORIZON TABS CONTAINER LAYER ---
        st.markdown("---")
        tab_st, tab_mt, tab_lt = st.tabs(["Short-Term\nIntraday - 1 Week", "Medium-Term\n2 Weeks - 1 Month", "Long-Term\n1 - 6 Months"])
        
        with tab_st:
            st.markdown('<div class="section-box"><div class="engine-header">Part 1 Engine</div><div class="engine-title">Macro Fundamental Sentiment</div>', unsafe_allow_html=True)
            st.components.v1.html(generate_html_gauge("Fundamental Matrix Arc", fund_score_live, "FUND"), height=255)
            st.markdown('</div><div class="section-box"><div class="engine-header">Part 2 Engine</div><div class="engine-title">GreenCrow Technical Breakouts</div>', unsafe_allow_html=True)
            st.components.v1.html(generate_html_gauge("Technical Breakout Arc", tech_vector, "TECH"), height=255)
            st.markdown('</div><div class="section-box"><div class="engine-header">Part 3 Engine</div><div class="engine-title">AI Predictive Forecast Oracle</div>', unsafe_allow_html=True)
            st.components.v1.html(generate_html_gauge("Short-Term Machine Conviction", ai_st_pct, "AI"), height=255)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab_mt:
            st.markdown('<div class="section-box"><div class="engine-header">Part 3 Engine</div><div class="engine-title">AI Medium-Term Horizon Forecast Oracle</div>', unsafe_allow_html=True)
            st.components.v1.html(generate_html_gauge("Medium-Term Machine Conviction", ai_mt_pct, "AI"), height=255)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab_lt:
            st.markdown('<div class="section-box"><div class="engine-header">Part 3 Engine</div><div class="engine-title">AI Long-Term Horizon Forecast Oracle</div>', unsafe_allow_html=True)
            st.components.v1.html(generate_html_gauge("Long-Term Machine Conviction", ai_lt_pct, "AI"), height=255)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- PHASE 7: RENDER THE HIGH-CONTRAST DATA MATRIX SCORECARD ---
        st.markdown("---")
        st.subheader("📋 Macro Portfolio Scorecard Matrix")
        
        def apply_color_shading(val):
            try:
                numeric_val = float(val)
                color = '#00FF66' if numeric_val > 0 else '#FF0033' if numeric_val < 0 else '#C9D1D9'
                return f'color: {color}; font-weight: bold; font-family: "Consolas", monospace;'
            except: return 'color: #C9D1D9;'
                
        styled_scorecard = df_scorecard.style.map(apply_color_shading, subset=['Change (Points)', 'Change (%)']).format({
            'Change (Points)': '{:+.2f}', 'Change (%)': '{:+.2f}%'
        })
        st.dataframe(styled_scorecard, use_container_width=True, hide_index=True)
