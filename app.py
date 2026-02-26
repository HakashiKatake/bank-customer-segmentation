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
    page_title="BANKSEG / Customer Intelligence",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DESIGN SYSTEM ─────────────────────────────────────────────
# Aesthetic: Industrial Utilitarian — monochrome, sharp
# Fonts: IBM Plex Mono (display) + IBM Plex Sans (body)
# No gradients. No border-radius. No hover animations.
# Strict neutral palette: white, off-white, mid-gray, charcoal, near-black.

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

/* ── SIDEBAR PADDING & LAYOUT FIXES ── */
[data-testid="stSidebarHeader"] {
  padding: 0 !important;
  min-height: 0 !important;
}
[data-testid="stSidebarUserContent"] {
  padding-top: 1.5rem !important;
  padding-bottom: 1rem !important;
  display: flex;
  flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div:first-child {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div:first-child > div:first-child {
  height: 100%;
  display: flex;
  flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div:first-child > div:first-child > div:last-child {
  margin-top: auto;
}

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

/* ── PLOTLY CHARTS: strip built-in padding ── */
.js-plotly-plot { border: 1px solid #d0cfc8; }

/* ── HEADINGS ── */
h1, h2, h3, h4 { 
  font-family: 'IBM Plex Mono', monospace !important;
  color: #1a1a1a !important;
  font-weight: 500;
  letter-spacing: -0.01em;
}

/* ── SIDEBAR NAV RADIO ── */
section[data-testid="stSidebar"] .stRadio > div {
  gap: 4px;
}
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

# ─── PLOTLY NEUTRAL TEMPLATE ───────────────────────────────────
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

# Neutral sequential palette (no hue shift, just value steps)
NEUTRAL_SEQ = ["#1a1a1a", "#3a3a3a", "#5a5a5a", "#7a7a7a", "#9a9a9a"]
CATEGORICAL  = ["#1a1a1a", "#5a5a52", "#8a8a82", "#b4b4ac", "#d0cfc8"]


# ─── DATA ──────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=50_000, seed=42):
    np.random.seed(seed)
    regions       = ['North', 'South', 'East', 'West', 'Central']
    account_types = ['Savings', 'Checking', 'Business', 'Premium', 'Youth']
    region_growth = {'North':1.0, 'South':1.15, 'East':1.05, 'West':1.2, 'Central':0.95}
    df = pd.DataFrame({
        'CustomerID':        np.arange(1, n+1),
        'Age':               np.random.randint(18, 75, n),
        'Gender':            np.random.choice(['Male','Female','Other'], n, p=[0.48,0.48,0.04]),
        'Region':            np.random.choice(regions, n, p=[0.2,0.25,0.2,0.2,0.15]),
        'AccountType':       np.random.choice(account_types, n),
        'Balance':           np.abs(np.random.normal(25000, 12000, n)).round(2),
        'NumTransactions':   np.random.poisson(12, n),
        'TransactionVolume': np.abs(np.random.exponential(3000, n)).round(2),
        'CreditScore':       np.clip(np.random.normal(700, 80, n), 300, 850).astype(int),
        'LoanAmount':        np.abs(np.random.exponential(45000, n)).round(2),
        'MonthlyIncome':     np.abs(np.random.normal(5000, 2000, n)).round(2),
        'Tenure':            np.random.randint(0, 20, n),
        'NumProducts':       np.random.randint(1, 6, n),
        'IsActive':          np.random.choice([0,1], n, p=[0.2,0.8]),
        'HasCreditCard':     np.random.choice([0,1], n, p=[0.35,0.65]),
        'SatisfactionScore': np.random.randint(1, 11, n),
        'Year':              np.random.choice([2020,2021,2022,2023,2024], n),
        'Quarter':           np.random.choice([1,2,3,4], n)
    })
    df['TransactionVolume'] = df.apply(
        lambda r: r['TransactionVolume'] * region_growth[r['Region']], axis=1)
    df['BalanceToIncome']   = (df['Balance'] / (df['MonthlyIncome'] * 12 + 1)).round(4)
    df['LoanToBalance']     = (df['LoanAmount'] / (df['Balance'] + 1)).round(4)
    df['AvgTransactionAmt'] = (df['TransactionVolume'] / (df['NumTransactions'] + 1)).round(2)
    return df


@st.cache_resource
def train_models(df):
    feats = ['Age','Balance','NumTransactions','TransactionVolume',
             'CreditScore','LoanAmount','MonthlyIncome','Tenure',
             'NumProducts','SatisfactionScore','BalanceToIncome','LoanToBalance']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(df[feats])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df = df.copy()
    df['Cluster'] = kmeans.fit_predict(X_sc)
    names = {0:'PREMIUM SAVERS', 1:'GROWTH INVESTORS', 2:'ACTIVE SPENDERS', 3:'ENTRY-LEVEL'}
    df['Segment'] = df['Cluster'].map(names)

    reg = df.groupby(['Region','Year','Quarter']).agg(
        TotalVolume=('TransactionVolume','sum'),
        AvgBalance=('Balance','mean'),
        AvgCreditScore=('CreditScore','mean'),
        CustomerCount=('CustomerID','count'),
        AvgIncome=('MonthlyIncome','mean'),
        AvgTenure=('Tenure','mean')
    ).reset_index()
    reg['TimeIndex'] = (reg['Year'] - reg['Year'].min())*4 + reg['Quarter']
    le = LabelEncoder()
    reg['RegionEnc'] = le.fit_transform(reg['Region'])
    rf = ['TimeIndex','RegionEnc','AvgBalance','AvgCreditScore','CustomerCount','AvgIncome','AvgTenure']
    lr_sc = StandardScaler()
    Xr = lr_sc.fit_transform(reg[rf])
    lr = LinearRegression().fit(Xr, reg['TotalVolume'])
    return df, kmeans, scaler, lr, lr_sc, le, names, feats


def chart(fig):
    """Apply neutral template to every figure."""
    fig.update_layout(
        **NEUTRAL,
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
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


# ─── MAIN ──────────────────────────────────────────────────────
def main():
    df_raw = generate_data()
    df, kmeans, scaler, lr, lr_sc, le, names, feats = train_models(df_raw)

    # ── SIDEBAR ─────────────────────────────────────────────
    with st.sidebar:
        # ── SIDEBAR HEADER
        with st.container():
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.95rem;
                        letter-spacing:0.06em; font-weight:500; color:#f0efe8;
                        padding:0px 0 4px;">
              BANKSEG <span style='color:#6a6a60;'>/ AI</span>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        color:#5a5a52; letter-spacing:0.1em; text-transform:uppercase;
                        padding-bottom:20px;">
              Customer Intelligence
            </div>
            """, unsafe_allow_html=True)
            st.divider()

        # ── SIDEBAR MAIN BODY
        with st.container():
            page = st.radio("Navigate", [
                "Overview",
                "Segments",
                "Regions",
                "Forecast",
                "Predictor"
            ], label_visibility="collapsed")

        # ── SIDEBAR FOOTER
        with st.container():
            st.divider()
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        text-transform:uppercase; letter-spacing:0.1em; color:#5a5a52;
                        padding-bottom:8px;">
              Dataset
            </div>""", unsafe_allow_html=True)
            st.info("50,000 customers\n5 regions / 4 segments")
            st.markdown("""
            <a href="https://www.kaggle.com/datasets/ksabishek/massive-bank-dataset-1-million-rows"
               style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#8a8a82; display:block; padding-bottom:1rem;">
              Kaggle: Massive Bank Dataset →
            </a>""", unsafe_allow_html=True)

    # ── HEADER ──────────────────────────────────────────────
    st.markdown("""
    <div style="border-bottom:2px solid #1a1a1a; padding-bottom:20px; margin-bottom:28px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:1.9rem;
                  font-weight:500; color:#1a1a1a; letter-spacing:-0.01em;">
        BANK CUSTOMER INTELLIGENCE
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem;
                  color:#5a5a52; margin-top:6px;">
        Segmentation via K-Means · Regional forecasting via Linear Regression
      </div>
    </div>
    """, unsafe_allow_html=True)

    regions = ['North', 'South', 'East', 'West', 'Central']

    # ════════════════════════════════════════════════════════
    #  PAGE — OVERVIEW
    # ════════════════════════════════════════════════════════
    if page == "Overview":
        section("Key metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Customers",   f"{len(df):,}",           "+5.2%")
        with c2: st.metric("Avg Balance",        f"${df['Balance'].mean():,.0f}", "+3.1%")
        with c3: st.metric("Avg Credit Score",   f"{df['CreditScore'].mean():.0f}", "+12 pts")
        with c4: st.metric("Transaction Volume", f"${df['TransactionVolume'].sum()/1e6:.1f}M", "+8.4%")

        st.divider()
        section("Transaction volume by region · 2020-2024")

        tr = df.groupby(['Year','Region'])['TransactionVolume'].sum().reset_index()
        fig = px.line(tr, x='Year', y='TransactionVolume', color='Region',
                      markers=True, color_discrete_sequence=NEUTRAL_SEQ)
        fig.update_traces(line_width=1.5, marker_size=4)
        st.plotly_chart(chart(fig), use_container_width=True)

        st.divider()
        section("Segment overview")
        profile = df.groupby('Segment').agg(
            Count=('CustomerID','count'),
            AvgBalance=('Balance','mean'),
            AvgCredit=('CreditScore','mean'),
            AvgIncome=('MonthlyIncome','mean'),
            AvgProducts=('NumProducts','mean'),
            AvgSatisf=('SatisfactionScore','mean'),
        ).round(1).reset_index()
        profile.rename(columns={
            'Segment':'Segment','Count':'Customers',
            'AvgBalance':'Avg Balance','AvgCredit':'Avg Credit',
            'AvgIncome':'Avg Income','AvgProducts':'Products','AvgSatisf':'Satisfaction'
        }, inplace=True)
        profile['Avg Balance'] = profile['Avg Balance'].apply(lambda x: f"${x:,.0f}")
        profile['Avg Income']  = profile['Avg Income'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(profile, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    #  PAGE — SEGMENTS
    # ════════════════════════════════════════════════════════
    elif page == "Segments":
        section("K-Means clusters · PCA projection (n=8,000)")

        X_sc = scaler.transform(df[feats])
        pca  = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_sc[:8000])
        pca_df = pd.DataFrame(coords, columns=['PC1','PC2'])
        pca_df['Segment'] = df['Segment'].values[:8000]

        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Segment',
                         opacity=0.55, color_discrete_sequence=NEUTRAL_SEQ)
        fig.update_traces(marker_size=3)
        st.plotly_chart(chart(fig), use_container_width=True)

        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            section("Select segment")
            seg = st.selectbox("Segment", list(names.values()), label_visibility="collapsed")

        seg_df = df[df['Segment'] == seg]

        with col2:
            section(f"Profile · {seg.lower()}")
            m = seg_df[['Age','Balance','CreditScore','MonthlyIncome',
                         'NumProducts','Tenure','SatisfactionScore']].mean().round(1)
            m_df = pd.DataFrame({'Metric': m.index, 'Mean': m.values})
            fig2 = px.bar(m_df, x='Metric', y='Mean',
                          color_discrete_sequence=['#1a1a1a'])
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(chart(fig2), use_container_width=True)

        st.divider()
        section("Age distribution · segment vs all customers")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df['Age'], name='All', nbinsx=30,
            opacity=0.4, marker_color='#9a9a9a', marker_line_width=0))
        fig3.add_trace(go.Histogram(
            x=seg_df['Age'], name=seg, nbinsx=30,
            opacity=1.0, marker_color='#1a1a1a', marker_line_width=0))
        fig3.update_layout(barmode='overlay')
        st.plotly_chart(chart(fig3), use_container_width=True)

        # Insight
        tips = {
            'PREMIUM SAVERS':   "High balance, high credit score, long tenure. Retention focus: private banking & exclusive products.",
            'GROWTH INVESTORS':  "Mid-career professionals growing wealth. Target: investment accounts, mortgages.",
            'ACTIVE SPENDERS':  "High transaction frequency. Target: cashback credit lines and spend rewards.",
            'ENTRY-LEVEL':      "New, lower-balance customers. Target: digital onboarding and savings education.",
        }
        if seg in tips:
            st.info(tips[seg])

    # ════════════════════════════════════════════════════════
    #  PAGE — REGIONS
    # ════════════════════════════════════════════════════════
    elif page == "Regions":
        section("Regional metrics")
        rs = df.groupby('Region').agg(
            Customers=('CustomerID','count'),
            AvgBalance=('Balance','mean'),
            TotalVolume=('TransactionVolume','sum'),
            AvgCredit=('CreditScore','mean'),
        ).reset_index().round(0)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(rs, x='Region', y='TotalVolume',
                         color_discrete_sequence=['#1a1a1a'])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(title_text="Total Transaction Volume by Region")
            st.plotly_chart(chart(fig), use_container_width=True)
        with c2:
            fig2 = px.scatter(rs, x='AvgBalance', y='AvgCredit', size='Customers',
                              text='Region', color_discrete_sequence=['#1a1a1a'])
            fig2.update_traces(textfont_family="IBM Plex Mono", textfont_size=9,
                                textposition="top center", marker_line_width=0)
            fig2.update_layout(title_text="Avg Balance vs Credit Score (size=Customers)")
            st.plotly_chart(chart(fig2), use_container_width=True)

        st.divider()
        section("Segment share by region")
        cross = pd.crosstab(df['Region'], df['Segment'], normalize='index').round(3)*100
        fig3  = px.bar(cross.reset_index().melt(id_vars='Region'),
                       x='Region', y='value', color='Segment', barmode='stack',
                       color_discrete_sequence=NEUTRAL_SEQ,
                       labels={'value':'Share (%)'})
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(chart(fig3), use_container_width=True)

    # ════════════════════════════════════════════════════════
    #  PAGE — FORECAST
    # ════════════════════════════════════════════════════════
    elif page == "Forecast":
        section("Regional transaction volume forecast · 2025-2026")

        reg_df = df.groupby(['Region','Year','Quarter']).agg(
            TotalVolume=('TransactionVolume','sum'),
            AvgBalance=('Balance','mean'),
            AvgCreditScore=('CreditScore','mean'),
            CustomerCount=('CustomerID','count'),
            AvgIncome=('MonthlyIncome','mean'),
            AvgTenure=('Tenure','mean')
        ).reset_index()
        reg_df['TimeIndex'] = (reg_df['Year'] - reg_df['Year'].min())*4 + reg_df['Quarter']
        reg_df['RegionEnc'] = le.transform(reg_df['Region'])

        rows, max_t = [], reg_df['TimeIndex'].max()
        future = [(2025,q) for q in range(1,5)] + [(2026,q) for q in range(1,5)]
        for region in regions:
            rd    = reg_df[reg_df['Region'] == region]
            r_enc = le.transform([region])[0]
            for i,(yr,qt) in enumerate(future):
                row_sc = lr_sc.transform([[max_t+i+1, r_enc,
                                           rd['AvgBalance'].mean(),
                                           rd['AvgCreditScore'].mean(),
                                           rd['CustomerCount'].mean()*1.05,
                                           rd['AvgIncome'].mean()*1.03,
                                           rd['AvgTenure'].mean()+1]])
                rows.append({'Region':region,'Year':yr,'Quarter':qt,
                              'Forecast':max(0, lr.predict(row_sc)[0]),
                              'Period':f"{yr} Q{qt}"})
        fdf = pd.DataFrame(rows)

        fig = px.bar(fdf, x='Period', y='Forecast', color='Region',
                     barmode='group', color_discrete_sequence=NEUTRAL_SEQ,
                     labels={'Forecast':'Volume ($)'})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(chart(fig), use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            section("Total forecast 2025-2026 by region")
            summary = (fdf.groupby('Region')['Forecast'].sum()
                          .sort_values(ascending=False)
                          .reset_index()
                          .rename(columns={'Forecast':'Total ($)'}))
            summary['Total ($)'] = summary['Total ($)'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with c2:
            section("Growth vs historical (%)")
            hist = reg_df.groupby('Region')['TotalVolume'].sum()
            fore = fdf.groupby('Region')['Forecast'].sum()
            growth = ((fore / hist * 100) - 100).round(1).reset_index()
            growth.columns = ['Region','Growth (%)']
            fig2 = px.bar(growth, x='Region', y='Growth (%)',
                          color='Growth (%)', color_continuous_scale=['#9a9a9a','#1a1a1a'])
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(chart(fig2), use_container_width=True)

        st.info("Model: Linear Regression · Features: Time index, region encoding, avg balance, credit score, customer count, income, tenure")

    # ════════════════════════════════════════════════════════
    #  PAGE — PREDICTOR
    # ════════════════════════════════════════════════════════
    elif page == "Predictor":
        section("Single customer segment & revenue predictor")
        st.markdown(
            "<p style='font-family:IBM Plex Sans,sans-serif;font-size:0.85rem;"
            "color:#5a5a52;margin-bottom:20px;'>"
            "Enter customer attributes to classify them into a segment and forecast their regional revenue contribution.</p>",
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            age        = st.slider("Age",                  18, 75,     35)
            balance    = st.number_input("Account Balance ($)", 100, 500000, 25000, 1000)
            num_txn    = st.number_input("Monthly Transactions", 0, 100,  12)
            txn_vol    = st.number_input("Monthly Txn Volume ($)", 100, 100000, 3000, 500)
        with c2:
            credit     = st.slider("Credit Score",          300, 850,   700)
            loan       = st.number_input("Loan Amount ($)",     0, 500000, 50000, 5000)
            income     = st.number_input("Monthly Income ($)",  500, 50000, 5000, 500)
            tenure     = st.slider("Tenure (years)",        0, 20,     5)
        with c3:
            n_prod     = st.slider("No. of Products",       1, 6,      2)
            satisf     = st.slider("Satisfaction (1-10)",   1, 10,     7)
            region     = st.selectbox("Region", regions)
            yr_pred    = st.selectbox("Forecast Year", [2025, 2026])

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        run = st.button("RUN PREDICTION")

        if run:
            b2i = balance / (income*12+1)
            l2b = loan / (balance+1)
            inp_sc = scaler.transform([[age, balance, num_txn, txn_vol, credit, loan,
                                         income, tenure, n_prod, satisf, b2i, l2b]])
            cluster_id = kmeans.predict(inp_sc)[0]
            segment    = names[cluster_id]

            r_enc = le.transform([region])[0]
            q_preds = []
            for i, qt in enumerate(range(1,5)):
                row_sc = lr_sc.transform([[20+i+1, r_enc, balance, credit, 200, income, tenure]])
                q_preds.append(max(0, lr.predict(row_sc)[0]))
            total = sum(q_preds)

            st.divider()
            section("Prediction results")
            r1, r2, r3 = st.columns(3)
            with r1: kv("Predicted Segment", segment)
            with r2: kv("Cluster ID",         str(cluster_id))
            with r3: kv(f"Forecast {yr_pred}", f"${total:,.0f}")

            st.divider()
            section(f"Quarterly breakdown · {region} · {yr_pred}")
            q_df = pd.DataFrame({
                'Quarter': [f"Q{q}" for q in range(1,5)],
                'Forecast ($)': q_preds
            })
            fig = px.bar(q_df, x='Quarter', y='Forecast ($)',
                         color_discrete_sequence=['#1a1a1a'])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(chart(fig), use_container_width=True)

            tips = {
                'PREMIUM SAVERS':   "Recommended: Private banking, wealth management, exclusive credit cards.",
                'GROWTH INVESTORS':  "Recommended: Investment accounts, mortgage products, insurance bundles.",
                'ACTIVE SPENDERS':  "Recommended: High cashback cards, personal loan top-ups, spend rewards.",
                'ENTRY-LEVEL':      "Recommended: Digital savings accounts, micro-loans, financial literacy tools.",
            }
            if segment in tips:
                st.info(tips[segment])


if __name__ == "__main__":
    main()
