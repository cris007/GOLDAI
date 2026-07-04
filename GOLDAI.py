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

# Custom Institutional CSS Injector Layer
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

st.title("🔱 Gold Hybrid AI Oracle Cockpit")
st.markdown("##### Constrained Machine Learning System (Green Crow Tech + App.py Macro Filters)")

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
# Processing Trigger Button
if st.button("RUN DEEP HYBRID ENSEMBLE CALCULATOR", type="primary", use_container_width=True):
    
    with st.spinner("Executing high-speed batch downloads and syncing rules..."):
        
        # --- FIXED BULLETPROOF BATCH DOWNLOAD ENGINES ---
        # Fetching all 11 global macro assets in one single package data request string
        ticker_symbols = ["GC=F", "DX-Y.NYB", "TLT", "^VIX", "SPY", "TIP", "FXE", "GOLD", "NEM", "GFI", "AEM", "GDXJ"]
        
        try:
            batch_data = yf.download(ticker_symbols, period="2y", interval="1d", group_by='ticker', progress=False)
        except Exception as e:
            st.error(f"⚠️ Yahoo Finance server ping error: {str(e)}")
            st.stop()
            
        # Isolate closing historical rows safely inside a consolidated core dataframe table grid
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
            
            # Gold Miner Closures
            df['M1_Close']   = batch_data['GOLD']['Close']
            df['M2_Close']   = batch_data['NEM']['Close']
            df['M3_Close']   = batch_data['GFI']['Close']
            df['M4_Close']   = batch_data['AEM']['Close']
            df['M5_Close']   = batch_data['GDXJ']['Close']
        except KeyError as ke:
            st.error("⚠️ Server returned incomplete fields over the holiday weekend. Please re-run scan.")
            st.stop()
            
        # Complete clean gap-filling to secure historical matrix integrity
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
        # --- PHASE 4: ENGINEER MULTI-HORIZON TARGETS MATRIX ---
        df['Target_ST'] = np.where(df['Gold_Close'].shift(-2) > df['Gold_Close'], 1, 0)
        df['Target_LT'] = np.where(df['Gold_Close'].shift(-10) > df['Gold_Close'], 1, 0)
        
        df_clean = df.dropna().copy()
        
        if df_clean.empty or len(df_clean) < 10:
            st.error("⚠️ Insufficient synchronized historical row counts. Please run execution scan again.")
        else:
            feature_cols = ['Green_Crow_Vector', 'App_Py_Fundamental_Score', 'DXY_Points', 'TLT_Points', 'Miner_Points', 'TIP_Points', 'VIX_Points']
            X = df_clean[feature_cols]
            
            # Train the Classifiers
            model_st = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
            model_st.fit(X, df_clean['Target_ST'])
            
            model_lt = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model_lt.fit(X, df_clean['Target_LT'])
            
            live_input = X.iloc[[-1]]
            prob_st = model_st.predict_proba(live_input)
            prob_lt = model_lt.predict_proba(live_input)

            # --- PHASE 5: RENDER THE DEVICE VISUAL INTERFACE PANELS ---
            st.markdown("---")
            
            # Extract array slices using strict scalar mappings to eliminate runtime warnings
            st_buy  = float(prob_st[0][1] * 100)
            st_sell = float(prob_st[0][0] * 100)
            
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            st.markdown('<span class="horizon-title">⚡ SHORT-TERM SCALPER AI (24-48 Hours)</span>', unsafe_allow_html=True)
            if st_buy >= 53.0:
                st.markdown(f'<div class="signal-text" style="color: #00FF66;">BULLISH SCALPING BUY ({st_buy:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-bar"><div style="background-color:#00FF66; width:{st_buy}%; height:100%; border-radius:5px;"></div></div>', unsafe_allow_html=True)
            elif st_sell >= 53.0:
                st.markdown(f'<div class="signal-text" style="color: #FF0033;">BEARISH SCALPING SELL ({st_sell:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-bar"><div style="background-color:#FF0033; width:{st_sell}%; height:100%; border-radius:5px;"></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-text" style="color: #FF9900;">SIDEWAYS CONSOLIDATION WAIT ({max(st_buy, st_sell):.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-bar"><div style="background-color:#FF9900; width:{max(st_buy, st_sell)}%; height:100%; border-radius:5px;"></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            lt_buy  = float(prob_lt[0][1] * 100)
            lt_sell = float(prob_lt[0][0] * 100)
            
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            st.markdown('<span class="horizon-title">🏛️ LONG-TERM MACRO AI (1-2 Weeks)</span>', unsafe_allow_html=True)
            if lt_buy >= 53.0:
                st.markdown(f'<div class="signal-text" style="color: #00FF66;">MACRO STRUCTURAL BUY ({lt_buy:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-bar"><div style="background-color:#00FF66; width:{lt_buy}%; height:100%; border-radius:5px;"></div></div>', unsafe_allow_html=True)
            elif lt_sell >= 53.0:
                st.markdown(f'<div class="signal-text" style="color: #FF0033;">MACRO STRUCTURAL SELL ({lt_sell:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-bar"><div style="background-color:#FF0033; width:{lt_sell}%; height:100%; border-radius:5px;"></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-text" style="color: #FF9900;">MACRO ACCUMULATION WAIT ({max(lt_buy, lt_sell):.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-bar"><div style="background-color:#FF9900; width:{max(lt_buy, lt_sell)}%; height:100%; border-radius:5px;"></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- PHASE 6: RENDER REGULATORY DECISION MATRIX STREAMS ---
            st.subheader("📋 Rule Convergence Importance Scores")
            importances = model_lt.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Strategic Engine Parameter Input": [
                    "Green Crow Technical Setup Vector (10/20/100 EMA + Breakouts)",
                    "App.py Cumulative Fundamental Score (Total Multi-Sector Points)",
                    "US Dollar Index Velocity Direction (DXY Points)",
                    "US10Y Yield Inverse Flow Direction (TLT Points)",
                    "Institutional Mining Unanimous Allocation Vector (Giants Voting)",
                    "Real Interest Rates Opportunity Cost Vector (TIP Points)",
                    "CBOE Market Fear Risk-Aversion Gauge (VIX Points)"
                ],
                "AI Network Brain Dependence Allocation": importances
            }).sort_values(by="AI Network Brain Dependence Allocation", ascending=False)
            
            feature_importance_df["AI Network Brain Dependence Allocation"] = feature_importance_df["AI Network Brain Dependence Allocation"].map(lambda x: f"{x*100:.2f}%")
            st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)
