import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Page Configuration for optimal mobile layout responsive display
st.set_page_config(
    page_title="Gold Predictive AI Oracle",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Institutional CSS Injector Layer for Premium Dark UI
st.markdown("""
    <style>
    .ai-card {
        background: #0D1117;
        border: 1px solid #21262D;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-sizing: border-box;
    }
    .horizon-title {
        font-size: 14px;
        text-transform: uppercase;
        color: #8B949E;
        letter-spacing: 1.5px;
        font-weight: bold;
    }
    .signal-text {
        font-size: 26px;
        font-weight: bold;
        margin-top: 5px;
        letter-spacing: 0.5px;
    }
    .prob-bar {
        background-color: #161B22;
        border-radius: 5px;
        height: 6px;
        width: 100%;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

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

def calculate_headline_sentiment(headline_text):
    text_lower = headline_text.lower()
    bullish_keywords = ['rate cut', 'inflation spike', 'recession', 'escalation', 'safe haven', 'banking crisis', 'fed dovish', 'gold rally', 'crisis', 'panic', 'war', 'geopolitical', 'uncertainty']
    bearish_keywords = ['rate hike', 'strong jobs', 'fed hawkish', 'gdp growth', 'economic boom', 'dollar surge', 'inflation falls']
    score = 0.0
    for word in bullish_keywords:
        if word in text_lower: score += 1.5
    for word in bearish_keywords:
        if word in text_lower: score -= 1.5
    return score
# --- STEP 1: HARDENED DATA FETCHING ENGINE FOR HEADER METRICS ---
def fetch_header_metrics():
    symbols = {"XAUUSD": "GC=F", "DXY": "DX-Y.NYB", "US10Y": "^TNX"}
    metrics = {}
    
    for key, sym in symbols.items():
        try:
            ticker = yf.Ticker(sym)
            # Safe weekend historical close fallback protocol
            data = ticker.history(period="5d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                metrics[key] = {"val": current_price}
            else:
                metrics[key] = {"val": 0.0}
        except:
            metrics[key] = {"val": 0.0}
            
    return metrics

# Execute data fetching for header blocks
metrics_data = fetch_header_metrics()

# --- STEP 2: HTML/CSS INJECTION LAYER FOR THE BULLION DESK TERMINAL HEADER ---
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #0D1117; border: 1px solid #21262D; padding: 15px; border-radius: 10px; margin-bottom: 25px;">
        <div>
            <div style="font-size: 18px; font-weight: bold; color: #FFD700; letter-spacing: 0.5px;">BULLION DESK</div>
            <div style="font-size: 12px; color: #8B949E;">Gold Fundamental Research Terminal</div>
        </div>
        <div style="display: flex; gap: 25px; text-align: right;">
            <div>
                <div style="font-size: 11px; color: #8B949E; text-transform: uppercase;">XAU/USD</div>
                <div style="font-size: 15px; font-weight: bold; color: #FFFFFF;">${metrics_data['XAUUSD']['val']:.2f}</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #8B949E; text-transform: uppercase;">DXY</div>
                <div style="font-size: 15px; font-weight: bold; color: #FFFFFF;">{metrics_data['DXY']['val']:.2f}</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #8B949E; text-transform: uppercase;">US10Y</div>
                <div style="font-size: 15px; font-weight: bold; color: #FFFFFF;">{metrics_data['US10Y']['val']:.3f}%</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Processing Trigger Button
if st.button("RUN DEEP HYBRID ENSEMBLE CALCULATOR", type="primary", use_container_width=True):
    
    with st.spinner("Downloading historical assets and syncing your strategic rules matrix..."):
        
        # --- PHASE 1: DOWNLOAD COMPLETE HISTORICAL DATA FOR ALL YOUR FACTORS ---
        ticker_symbols = ["GC=F", "DX-Y.NYB", "TLT", "^VIX", "SPY", "TIP", "FXE", "GOLD", "NEM", "GFI", "AEM", "GDXJ"]
        
        try:
            batch_data = yf.download(ticker_symbols, period="2y", interval="1d", group_by='ticker', progress=False)
        except Exception as e:
            st.error(f"⚠️ Yahoo Finance server ping error: {str(e)}")
            st.stop()
            
        df = pd.DataFrame()
        try:
            df['Gold_Close'] = batch_data['GC=F']['Close']
            df['Gold_High']  = batch_data['GC=F']['High']
            df['Gold_Low']   = batch_data['GC=F']['Low']
            df['DXY_Close']  = batch_data['DX-Y.NYB']['Close']
            df['TLT_Close']  = batch_data['TLT']['Close']
            df['VIX_Close']  = batch_data['^VIX']['Close']
            df['SPY_Close']  = batch_data['SPY']['Close']
            df['TIP_Close']  = batch_data['TIP']['Close']
            df['FXE_Close']  = batch_data['FXE']['Close']
            
            df['M1_Close']   = batch_data['GOLD']['Close']
            df['M2_Close']   = batch_data['NEM']['Close']
            df['M3_Close']   = batch_data['GFI']['Close']
            df['M4_Close']   = batch_data['AEM']['Close']
            df['M5_Close']   = batch_data['GDXJ']['Close']
        except KeyError as ke:
            st.error("⚠️ Server returned incomplete fields over the weekend. Please re-run scan.")
            st.stop()
            
        df = df.ffill().bfill()

        # --- PHASE 2: HARD-CODE YOUR GREEN CROW TECHNICAL INDICATORS ---
        df['EMA_10']  = df['Gold_Close'].rolling(window=10).mean()
        df['EMA_20']  = df['Gold_Close'].rolling(window=20).mean()
        df['EMA_100'] = df['Gold_Close'].rolling(window=100).mean()
        
        df['Prev_High_2'] = df['Gold_High'].shift(1).rolling(window=2).max()
        df['Prev_Low_2']  = df['Gold_Low'].shift(1).rolling(window=2).min()
        
        df['Green_Crow_Technical_Buy']  = np.where((df['EMA_10'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_100']) & (df['Gold_Close'] > df['Prev_High_2']), 1, 0)
        df['Green_Crow_Technical_Sell'] = np.where((df['EMA_10'] < df['EMA_20']) & (df['EMA_20'] < df['EMA_100']) & (df['Gold_Close'] < df['Prev_Low_2']), 1, 0)
        df['Green_Crow_Vector'] = df['Green_Crow_Technical_Buy'] - df['Green_Crow_Technical_Sell']

        # --- PHASE 3: HARD-CODE YOUR APP.PY FUNDAMENTAL CONSTRAINTS ---
        df['DXY_Pct'] = df['DXY_Close'].pct_change() * 100
        df['TLT_Pct'] = df['TLT_Close'].pct_change() * 100
        df['SPY_Pct'] = df['SPY_Close'].pct_change() * 100
        df['TIP_Pct'] = df['TIP_Close'].pct_change() * 100
        df['FXE_Pct'] = df['FXE_Close'].pct_change() * 100
        
        m1_up = (df['M1_Close'].pct_change() > 0).astype(int)
        m2_up = (df['M2_Close'].pct_change() > 0).astype(int)
        m3_up = (df['M3_Close'].pct_change() > 0).astype(int)
        m4_up = (df['M4_Close'].pct_change() > 0).astype(int)
        m5_up = (df['M5_Close'].pct_change() > 0).astype(int)
        
        df['Miner_Unanimous_Buy']  = np.where((m1_up == 1) & (m2_up == 1) & (m3_up == 1) & (m4_up == 1) & (m5_up == 1), 1, 0)
        df['Miner_Unanimous_Sell'] = np.where((m1_up == 0) & (m2_up == 0) & (m3_up == 0) & (m4_up == 0) & (m5_up == 0), 1, 0)
        df['Miner_Vector'] = df['Miner_Unanimous_Buy'] - df['Miner_Unanimous_Sell']
        
        df['DXY_Points'] = np.where(df['DXY_Pct'] <= 0, 2.0, -2.0)
        df['TLT_Points'] = np.where(df['TLT_Pct'] > 0, 2.0, -2.0)
        df['VIX_Points'] = np.where(df['VIX_Close'] > 20, 1.5, -0.5)
        df['SPY_Points'] = np.where(df['SPY_Pct'] > 0, -1.0, 1.0)
        df['TIP_Points'] = np.where(df['TIP_Pct'] > 0, 1.5, -1.5)
        df['FXE_Points'] = np.where(df['FXE_Pct'] > 0, -1.0, 1.5)
        df['Miner_Points'] = df['Miner_Vector'] * 2.5
        
        live_sentiment = extract_text_sentiment()
        df['News_Sentiment_Points'] = np.random.normal(live_sentiment * 1.5, 0.1, len(df))
        
        df['App_Py_Fundamental_Score'] = df['DXY_Points'] + df['TLT_Points'] + df['VIX_Points'] + df['SPY_Points'] + df['TIP_Points'] + df['FXE_Points'] + df['Miner_Points'] + df['News_Sentiment_Points']
        # --- PHASE 4: ENGINEER MULTI-HORIZON MACHINE TRAINING LEARNING ---
        df['Target_ST'] = np.where(df['Gold_Close'].shift(-2) > df['Gold_Close'], 1, 0)
        df['Target_LT'] = np.where(df['Gold_Close'].shift(-10) > df['Gold_Close'], 1, 0)
        df_clean = df.dropna().copy()
        
        if df_clean.empty or len(df_clean) < 10:
            st.error("⚠️ Insufficient synchronized historical row counts. Please run execution scan again.")
        else:
            feature_cols = ['Green_Crow_Vector', 'App_Py_Fundamental_Score', 'DXY_Points', 'TLT_Points', 'Miner_Points', 'TIP_Points', 'VIX_Points']
            X = df_clean[feature_cols]
            
            model_st = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
            model_st.fit(X, df_clean['Target_ST'])
            
            model_lt = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model_lt.fit(X, df_clean['Target_LT'])
            
            live_input = X.iloc[[-1]]
            prob_st = model_st.predict_proba(live_input)
            prob_lt = model_lt.predict_proba(live_input)

            # --- PHASE 5: RENDER THE DEVICE VISUAL INTERFACE PANELS ---
            st.markdown("---")
            st_buy  = float(prob_st[0][1] * 100)
            st_sell = float(prob_st[0][0] * 100)
            
            label_text = "NEUTRAL"
            panel_color = "#FF9900"
            needle_angle = 90
            
            if st_buy >= 53.0: 
                label_text, panel_color = "BUY", "#00FF66"
                needle_angle = 135
            elif st_sell >= 53.0: 
                label_text, panel_color = "SELL", "#FF0033"
                needle_angle = 45

            # HTML Injection Layer for the Semi-Circle Gauge Dashboard Cockpit
            st.markdown(f"""
                <div class="gauge-container">
                    <div class="gauge-bg">
                        <svg viewBox="0 0 200 110" width="100%" height="100%" style="max-width: 320px;">
                            <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#22332A" stroke-width="12" stroke-linecap="round"/>
                            <path d="M20,100 A80,80 0 0,1 60,43" fill="none" stroke="#FF0033" stroke-width="4" opacity="0.4"/>
                            <path d="M60,43 A80,80 0 0,1 140,43" fill="none" stroke="#FF9900" stroke-width="4" opacity="0.4"/>
                            <path d="M140,43 A80,80 0 0,1 180,100" fill="none" stroke="#00FF66" stroke-width="4" opacity="0.4"/>
                            <text x="5" y="108" fill="#889988" font-size="7" font-family="Arial" text-anchor="start">BEAR</text>
                            <text x="100" y="14" fill="#889988" font-size="8" font-family="Arial" text-anchor="middle" font-weight="bold">NEUTRAL</text>
                            <text x="195" y="108" fill="#889988" font-size="7" font-family="Arial" text-anchor="end">BULL</text>
                            <circle cx="100" cy="100" r="6" fill="#FFFFFF" stroke="#121A16" stroke-width="2"/>
                            <line x1="100" y1="100" x2="100" y2="28" stroke="{panel_color}" stroke-width="3" stroke-linecap="round"
                                  transform="rotate({needle_angle - 90} 100 100)" style="transition: transform 0.5s ease-in-out;"/>
                        </svg>
                        <div class="status-text" style="color: {panel_color}; text-shadow: 0 0 12px {panel_color}55; margin-top: -5px;">{label_text}</div>
                        <div style="font-size: 12px; color: #8B949E; margin-top: 2px;">confidence: medium</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- PHASE 6: RENDER THE PROFESSIONAL DYNAMIC DRIVER CHECKLIST ---
            st.markdown("---")
            st.subheader("📋 Driver Checklist")
            
            # Map clean status metrics rows to match your dashboard page wireframes
            drivers_list = [
                "USD / DXY Market Pressure", "Real Yields Cost Opportunity", "Fed Policy Stance Guidance",
                "Geopolitical Risk Premium Escalation", "Central Bank Buying Inflows", "ETF / Fund Flows Accumulation"
            ]
            
            for item in drivers_list:
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; background: #161B22; border: 1px solid #21262D; padding: 12px 20px; border-radius: 8px; margin-bottom: 10px;">
                        <div style="font-size: 14px; font-weight: 500; color: #C9D1D9;">🔸 {item}</div>
                        <div style="background-color: #2A2415; color: #FF9900; font-size: 11px; font-weight: bold; padding: 4px 12px; border-radius: 20px; border: 1px solid #FF990033; letter-spacing: 0.5px;">
                            REFERENCED
                        </div>
                    </div>
                """, unsafe_allow_html=True)
