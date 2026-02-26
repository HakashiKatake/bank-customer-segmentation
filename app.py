import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="BANKSEG / Location Intelligence",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DESIGN SYSTEM ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  font-family: 'IBM Plex Sans', monospace;
  background: #f5f4f0;
  color: #1a1a1a;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: #1a1a1a;
  border-right: 1px solid #2e2e2e;
}
section[data-testid="stSidebar"] * { color: #c8c8c0 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #c8c8c0 !important; }
section[data-testid="stSidebar"] hr { border-color: #2e2e2e; }
section[data-testid="stSidebar"] a { color: #8a8a82 !important; text-decoration: underline; }
section[data-testid="stSidebar"] [data-testid="stInfo"] {
  background: #252525;
  border: 1px solid #333;
  border-radius: 0;
  color: #8a8a82 !important;
}
section[data-testid="stSidebar"] [data-testid="stInfo"] * { color: #8a8a82 !important; }

/* ── MAIN AREA ── */
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
.stApp { background: #f5f4f0; }

/* ── METRIC CARDS ── */
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #d0cfc8;
  border-radius: 0;
  padding: 18px 20px;
}
div[data-testid="stMetric"] label { 
  color: #5a5a52 !important; 
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { 
  color: #1a1a1a !important; 
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.7rem !important;
  font-weight: 500;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { 
  color: #3a7a3a !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.75rem !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: #1a1a1a;
  color: #f5f4f0;
  border: 1px solid #1a1a1a;
  border-radius: 0;
  padding: 10px 24px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  width: 100%;
  cursor: pointer;
}

/* ── FORM INPUTS ── */
.stSelectbox label, .stSlider label, .stNumberInput label {
  color: #5a5a52 !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.stSelectbox > div > div {
  background: #ffffff;
  border: 1px solid #d0cfc8;
  border-radius: 0;
  color: #1a1a1a;
}
.stNumberInput input {
  background: #ffffff;
  border: 1px solid #d0cfc8;
  border-radius: 0;
  color: #1a1a1a;
  font-family: 'IBM Plex Mono', monospace;
}

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid #d0cfc8; }

/* ── DIVIDER ── */
hr { border: none; border-top: 1px solid #d0cfc8; margin: 1.5rem 0; }

/* ── ALERT BOXES ── */
.stSuccess, .stInfo, .stWarning, .stError {
  border-radius: 0 !important;
  border-left-width: 3px !important;
}
.stSuccess { 
  background: #f0f7f0 !important; 
  border-left: 3px solid #3a7a3a !important;
}
.stInfo { 
  background: #f0f4f7 !important; 
  border-left: 3px solid #3a5a7a !important;
}
.stWarning { 
  background: #f7f5f0 !important; 
  border-left: 3px solid #7a6a3a !important;
}
.stSuccess *, .stInfo *, .stWarning * { color: #1a1a1a !important; }

/* ── PLOTLY CHARTS ── */
.js-plotly-plot { border: 1px solid #d0cfc8; }

h1, h2, h3, h4 { 
  font-family: 'IBM Plex Mono', monospace !important;
  color: #1a1a1a !important;
  font-weight: 500;
  letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
section[data-testid="stSidebar"] .stRadio label {
  padding: 6px 10px;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio label[data-selected="true"] {
  background: #2e2e2e;
  border: 1px solid #3a3a3a;
}
</style>
""", unsafe_allow_html=True)

NEUTRAL = dict(
    font_family="IBM Plex Mono",
    font_color="#1a1a1a",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    title_font_family="IBM Plex Mono",
    title_font_size=13,
    title_font_color="#1a1a1a",
    legend_font_family="IBM Plex Mono",
    legend_font_size=11,
    xaxis=dict(gridcolor="#e8e7e0", linecolor="#d0cfc8", tickfont_family="IBM Plex Mono", tickfont_size=10),
    yaxis=dict(gridcolor="#e8e7e0", linecolor="#d0cfc8", tickfont_family="IBM Plex Mono", tickfont_size=10),
)
NEUTRAL_SEQ = ["#1a1a1a", "#3a3a3a", "#5a5a5a", "#7a7a7a", "#9a9a9a"]
CATEGORICAL  = ["#1a1a1a", "#5a5a52", "#8a8a82", "#b4b4ac", "#d0cfc8"]


@st.cache_data
def load_data():
    df = pd.read_csv('bankdataset.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['AvgTxnValue'] = (df['Value'] / df['Transaction_count']).round(2)
    return df


@st.cache_resource
def train_models(df):
    # 1. Location-based Segmentation
    loc_stats = df.groupby('Location').agg(
        TotalValue=('Value', 'sum'),
        TotalTxns=('Transaction_count', 'sum'),
        AvgTxnValue=('AvgTxnValue', 'mean'),
        DistinctDomains=('Domain', 'nunique'),
        DaysActive=('Date', 'nunique')
    ).reset_index()
    
    feats = ['TotalValue', 'TotalTxns', 'AvgTxnValue', 'DistinctDomains', 'DaysActive']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(loc_stats[feats])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    loc_stats['Cluster'] = kmeans.fit_predict(X_sc)
    
    names = {0:'PREMIUM MARKETS', 1:'VOLUME HUBS', 2:'BALANCED ZONES', 3:'EMERGING AREAS'}
    loc_stats['Segment'] = loc_stats['Cluster'].map(names)
    
    # 2. Linear Regression (Forecasting by Domain & Location over Time)
    reg_df = df.groupby(['Location', 'Domain', 'Year', 'Quarter']).agg(
        TotalValue=('Value', 'sum'),
        TotalTxns=('Transaction_count', 'sum')
    ).reset_index()
    
    reg_df['TimeIndex'] = (reg_df['Year'] - reg_df['Year'].min())*4 + reg_df['Quarter']
    
    le_loc = LabelEncoder()
    le_dom = LabelEncoder()
    
    reg_df['LocEnc'] = le_loc.fit_transform(reg_df['Location'])
    reg_df['DomEnc'] = le_dom.fit_transform(reg_df['Domain'])
    
    rf = ['TimeIndex', 'LocEnc', 'DomEnc', 'TotalTxns']
    lr_sc = StandardScaler()
    Xr = lr_sc.fit_transform(reg_df[rf])
    lr = LinearRegression().fit(Xr, reg_df['TotalValue'])
    
    return df, loc_stats, kmeans, scaler, lr, lr_sc, le_loc, le_dom, names, feats


def chart(fig):
    fig.update_layout(**NEUTRAL, margin=dict(l=20, r=20, t=40, b=20), height=340)
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig


def section(label):
    st.markdown(f"""
    <div style="border-top:2px solid #1a1a1a; padding-top:10px; margin-top:8px; margin-bottom:18px;">
      <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.12em;
                   text-transform:uppercase; color:#5a5a52;">
        {label}
      </span>
    </div>""", unsafe_allow_html=True)


def kv(label, value):
    st.markdown(f"""
    <div style="border:1px solid #d0cfc8; padding:14px 16px; background:#fff; margin-bottom:4px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; text-transform:uppercase;
                  letter-spacing:0.1em; color:#5a5a52; margin-bottom:4px;">{label}</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:1.45rem; color:#1a1a1a; font-weight:500;">{value}</div>
    </div>""", unsafe_allow_html=True)


def main():
    try:
        df_raw = load_data()
    except FileNotFoundError:
        st.error("bankdataset.csv not found. Please ensure the dataset exists in the directory.")
        return

    df, loc_stats, kmeans, scaler, lr, lr_sc, le_loc, le_dom, names, feats = train_models(df_raw)

    with st.sidebar:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.95rem;
                    letter-spacing:0.06em; font-weight:500; color:#f0efe8;
                    padding:16px 0 4px;">
          BANKSEG <span style='color:#6a6a60;'>/ AI</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    color:#5a5a52; letter-spacing:0.1em; text-transform:uppercase;
                    padding-bottom:20px;">
          Location Intelligence
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        page = st.radio("Navigate", [
            "Overview",
            "Market Segments",
            "Domains",
            "Forecast",
            "Predictor"
        ], label_visibility="collapsed")

        st.divider()
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    text-transform:uppercase; letter-spacing:0.1em; color:#5a5a52;
                    padding-bottom:8px;">
          Dataset
        </div>""", unsafe_allow_html=True)
        st.info(f"{len(df):,} records\n{df['Location'].nunique()} locations\n{df['Domain'].nunique()} domains")


    st.markdown("""
    <div style="border-bottom:2px solid #1a1a1a; padding-bottom:20px; margin-bottom:28px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:1.9rem;
                  font-weight:500; color:#1a1a1a; letter-spacing:-0.01em;">
        BANK LOCATION & DOMAIN INTELLIGENCE
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem;
                  color:#5a5a52; margin-top:6px;">
        Market Segmentation via K-Means · Domain forecasting via Linear Regression
      </div>
    </div>
    """, unsafe_allow_html=True)


    if page == "Overview":
        section("Key metrics")
        c1, c2, c3, c4 = st.columns(4)
        total_val = df['Value'].sum()
        total_txns = df['Transaction_count'].sum()
        with c1: st.metric("Total Processing Vol", f"₹{total_val/1e9:.2f}B", "")
        with c2: st.metric("Total Transactions",  f"{total_txns/1e6:.1f}M", "")
        with c3: st.metric("Unique Locations",    f"{df['Location'].nunique()}", "")
        with c4: st.metric("Avg Value per Txn",   f"₹{df['AvgTxnValue'].mean():.2f}", "")

        st.divider()
        section("Transaction volume over time")
        tr = df.groupby(['Year', 'Month', 'Date'])['Value'].sum().reset_index()
        fig = px.line(tr, x='Date', y='Value', color_discrete_sequence=['#1a1a1a'])
        fig.update_traces(line_width=1.5)
        st.plotly_chart(chart(fig), use_container_width=True)

        st.divider()
        section("Top 10 Locations by Volume")
        top_locs = loc_stats.sort_values('TotalValue', ascending=False).head(10)
        fig2 = px.bar(top_locs, x='Location', y='TotalValue', color_discrete_sequence=['#1a1a1a'])
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(chart(fig2), use_container_width=True)


    elif page == "Market Segments":
        section("K-Means Market Clusters · PCA projection")
        
        X_sc = scaler.transform(loc_stats[feats])
        pca  = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_sc)
        pca_df = pd.DataFrame(coords, columns=['PC1','PC2'])
        pca_df['Segment'] = loc_stats['Segment']
        pca_df['Location'] = loc_stats['Location']

        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Segment', text='Location',
                         opacity=0.7, color_discrete_sequence=NEUTRAL_SEQ)
        fig.update_traces(marker_size=8, textposition='top center', textfont_size=8, textfont_family="IBM Plex Mono")
        st.plotly_chart(chart(fig), use_container_width=True)

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            section("Select market segment")
            seg = st.selectbox("Segment", list(names.values()), label_visibility="collapsed")

        seg_df = loc_stats[loc_stats['Segment'] == seg]

        with col2:
            section(f"Profile · {seg.lower()}")
            m = seg_df[['TotalValue','TotalTxns','AvgTxnValue','DistinctDomains']].mean().round(1)
            m_df = pd.DataFrame({'Metric': m.index, 'Mean': m.values})
            fig2 = px.bar(m_df, x='Metric', y='Mean', color_discrete_sequence=['#1a1a1a'])
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(chart(fig2), use_container_width=True)

        st.divider()
        section("Locations in this segment")
        disp_df = seg_df[['Location', 'TotalValue', 'TotalTxns', 'AvgTxnValue']].sort_values('TotalValue', ascending=False)
        disp_df['TotalValue'] = disp_df['TotalValue'].apply(lambda x: f"₹{x:,.0f}")
        disp_df['TotalTxns']  = disp_df['TotalTxns'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)


    elif page == "Domains":
        section("Volume by business domain")
        dom_df = df.groupby('Domain').agg(
            TotalValue=('Value','sum'),
            TotalTxns=('Transaction_count','sum')
        ).reset_index().sort_values('TotalValue', ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(dom_df, x='Domain', y='TotalValue', color_discrete_sequence=['#1a1a1a'])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(title_text="Total Transaction Volume")
            st.plotly_chart(chart(fig), use_container_width=True)
        with c2:
            fig2 = px.bar(dom_df, x='Domain', y='TotalTxns', color_discrete_sequence=['#5a5a52'])
            fig2.update_traces(marker_line_width=0)
            fig2.update_layout(title_text="Total Transaction Count")
            st.plotly_chart(chart(fig2), use_container_width=True)

        st.divider()
        section("Market Share Over Time")
        dom_time = df.groupby(['Year', 'Quarter', 'Domain'])['Value'].sum().reset_index()
        dom_time['Period'] = dom_time['Year'].astype(str) + " Q" + dom_time['Quarter'].astype(str)
        fig3 = px.bar(dom_time, x='Period', y='Value', color='Domain', 
                      barmode='stack', color_discrete_sequence=CATEGORICAL + ["#3a3a3a", "#7a7a7a"])
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(chart(fig3), use_container_width=True)


    elif page == "Forecast":
        section("Volume forecast by Domain · 2024-2025")
        
        reg_df = df.groupby(['Domain', 'Year', 'Quarter']).agg(
            TotalValue=('Value', 'sum'),
            TotalTxns=('Transaction_count', 'mean') # avg per location locally? no, just across board
        ).reset_index()

        min_yr = df['Year'].min()
        last_yr = df['Year'].max()
        last_q = df[df['Year'] == last_yr]['Quarter'].max()
        
        # Simple domain-level trending for visualization
        time_idx = (reg_df['Year'] - min_yr)*4 + reg_df['Quarter']
        
        rows = []
        domains = df['Domain'].unique()
        
        # We will generate rolling forecasts simply using a Domain-specific linear trend for demonstration
        for dom in domains:
            d_data = reg_df[reg_df['Domain'] == dom]
            x = ((d_data['Year'] - min_yr)*4 + d_data['Quarter']).values.reshape(-1,1)
            y = d_data['TotalValue'].values
            if len(x) > 1:
                mdl = LinearRegression().fit(x, y)
                for i in range(1, 5): # Predict next 4 quarters
                    future_idx = (last_yr - min_yr)*4 + last_q + i
                    yr = last_yr if (last_q + i) <= 4 else (last_yr + 1)
                    qt = (last_q + i) if (last_q + i) <= 4 else (last_q + i - 4)
                    pred = mdl.predict([[future_idx]])[0]
                    rows.append({'Domain': dom, 'Year': yr, 'Quarter': qt, 'Forecast': max(0, pred), 'Period': f"{yr} Q{qt}"})
            
        if rows:
            fdf = pd.DataFrame(rows)
            fig = px.bar(fdf, x='Period', y='Forecast', color='Domain',
                         barmode='group', color_discrete_sequence=CATEGORICAL + ["#3a3a3a", "#7a7a7a"],
                         labels={'Forecast':'Forecast Volume (₹)'})
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(chart(fig), use_container_width=True)

            st.divider()
            section("Total forecast summary")
            summary = fdf.groupby('Domain')['Forecast'].sum().sort_values(ascending=False).reset_index()
            summary['Forecast'] = summary['Forecast'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.info("Linear Regression trend projection based on historical domain performance.")
        else:
            st.warning("Not enough historical data to generate forecasts.")


    elif page == "Predictor":
        section("Location & Domain Forecaster")
        st.markdown(
            "<p style='font-family:IBM Plex Sans,sans-serif;font-size:0.85rem;"
            "color:#5a5a52;margin-bottom:20px;'>"
            "Estimate future transaction volume for a specific location and business domain.</p>",
            unsafe_allow_html=True)

        locations = sorted(df['Location'].unique())
        domains = sorted(df['Domain'].unique())

        c1, c2, c3 = st.columns(3)
        with c1:
            loc = st.selectbox("Market Location", locations)
        with c2:
            dom = st.selectbox("Business Domain", domains)
        with c3:
            est_txns = st.number_input("Estimated Txn Volume (Scale)", 100, 10000, 1500)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        run = st.button("CALCULATE PROJECTION")

        if run:
            # Segment info for location
            loc_seg = loc_stats[loc_stats['Location'] == loc]['Segment'].values[0]
            
            # Predict for next year using the trained model
            l_enc = le_loc.transform([loc])[0]
            d_enc = le_dom.transform([dom])[0]
            
            current_max_time = (df['Year'].max() - df['Year'].min())*4 + 4 
            
            row_sc = lr_sc.transform([[current_max_time + 4, l_enc, d_enc, est_txns]]) # Predict 4 quarters out
            forecast_val = max(0, lr.predict(row_sc)[0])

            st.divider()
            section("Projection results")
            r1, r2, r3 = st.columns(3)
            with r1: kv("Market Segment", loc_seg)
            with r2: kv("Location", loc)
            with r3: kv("Forecasted Base", f"₹{forecast_val:,.0f}")
            
            # Add a bit of visual flair
            st.divider()
            st.info(f"The structural market segment for **{loc}** is classified as **{loc_seg}**. The forecasted baseline volume per quarter for **{dom}** operations is roughly **₹{forecast_val:,.0f}**, assuming transaction counts hold at {est_txns}.")

if __name__ == "__main__":
    main()
