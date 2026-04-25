import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import os

st.set_page_config(
    page_title="Voice AI EDA Dashboard",
    layout="wide",
    page_icon="🎙"
)

@st.cache_data
def load_data():
    return pd.read_csv("commonvoice_cleaned_features.csv")

df = load_data()

# ── SIDEBAR ──────────────────────────────────
st.sidebar.title("Filters")
st.sidebar.markdown("All charts update when you change a filter.")

all_genders = df['gender'].unique().tolist()
sel_gender = st.sidebar.multiselect(
    "Gender", options=all_genders, default=all_genders
)

all_ages = df['age'].unique().tolist()
sel_age = st.sidebar.multiselect(
    "Age group", options=all_ages, default=all_ages
)

min_d = float(df['duration'].min())
max_d = float(df['duration'].max())
dur = st.sidebar.slider(
    "Duration (seconds)",
    min_value=min_d, max_value=max_d, value=(min_d, max_d)
)

df_f = df[
    df['gender'].isin(sel_gender) &
    df['age'].isin(sel_age) &
    df['duration'].between(dur[0], dur[1])
]
st.sidebar.metric("Clips visible", len(df_f))

# ── TITLE ────────────────────────────────────
st.title("Voice AI EDA Dashboard")
st.markdown("Turkish CommonVoice 17 — 500 audio clips analysed")
st.divider()

# ── KPI CARDS ────────────────────────────────
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total clips", len(df_f))
k2.metric("Avg duration", f"{df_f['duration'].mean():.2f}s")
k3.metric("Avg word count", f"{df_f['word_count'].mean():.1f}")
top_g = (df_f['gender'].value_counts().idxmax()
         if len(df_f) > 0 else "N/A")
k4.metric("Top gender",
          top_g.replace("_masculine","").replace("_feminine",""))
st.divider()

# ── ROW 1: Duration + Gender ──────────────────
st.subheader("Duration and demographics")
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown("**Duration distribution**")
    fig = px.histogram(df_f, x='duration', nbins=30,
                       color_discrete_sequence=['#378ADD'])
    fig.add_vline(x=df_f['duration'].mean(), line_dash="dash",
                  line_color="red",
                  annotation_text=f"Avg {df_f['duration'].mean():.2f}s")
    fig.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    st.markdown("**Clips by gender**")
    gc = df_f['gender'].value_counts().reset_index()
    gc.columns = ['gender', 'count']
    fig2 = px.bar(gc, x='gender', y='count', color='gender',
                  color_discrete_sequence=['#378ADD','#D4537E','#888780'])
    fig2.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── ROW 2: Scatter + Age ──────────────────────
st.subheader("Duration vs words and age groups")
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown("**Duration vs word count — hover dots to see transcript**")
    fig3 = px.scatter(df_f, x='duration', y='word_count',
                      color='gender', trendline='ols',
                      hover_data=['transcription_clean'],
                      color_discrete_sequence=['#378ADD','#D4537E','#888780'])
    fig3.update_layout(height=360)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    st.markdown("**Clips by age group**")
    ac = df_f['age'].value_counts().reset_index()
    ac.columns = ['age', 'count']
    fig4 = px.bar(ac, x='age', y='count', color='count',
                  color_continuous_scale='Teal')
    fig4.update_layout(height=360, showlegend=False,
                       coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── ROW 3: Heatmap + Box plot ─────────────────
st.subheader("Feature analysis")
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.markdown("**Correlation heatmap**")
    if len(df_f) > 5:
        cc = ['duration','word_count','up_votes',
              'pitch_mean','energy_mean','mfcc_1','mfcc_2']
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        sns.heatmap(df_f[cc].corr().round(2), annot=True,
                    fmt='.2f', cmap='coolwarm', center=0,
                    ax=ax5, linewidths=0.4)
        st.pyplot(fig5)
        plt.close()

with r3c2:
    st.markdown("**Pitch by age group**")
    dk = df_f[df_f['age'] != 'unknown']
    if len(dk) > 0:
        fig6 = px.box(dk, x='age', y='pitch_mean', color='age',
                      points='outliers')
        fig6.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── WORD CLOUD ────────────────────────────────
st.subheader("Word cloud")
if len(df_f) > 10:
    txt = ' '.join(df_f['transcription_clean'].dropna())
    wc = WordCloud(width=1400, height=450,
                   background_color='white',
                   max_words=120,
                   colormap='Blues').generate(txt)
    figw, axw = plt.subplots(figsize=(14, 4))
    axw.imshow(wc, interpolation='bilinear')
    axw.axis('off')
    st.pyplot(figw)
    plt.close()

st.divider()

# ── CHART GALLERY ─────────────────────────────
st.subheader("Audio analysis charts")
g1, g2 = st.columns(2)
for path, cap, col in [
    ("charts/chart3_waveform_spectrogram.png", "Waveform and spectrogram", g1),
    ("charts/chart4_mfcc_heatmap.png",         "MFCC heatmap",            g2),
    ("charts/chart7_pitch_by_age.png",          "Pitch by age",            g1),
    ("charts/chart8_wordcloud.png",             "Phase 3 word cloud",      g2),
]:
    if os.path.exists(path):
        col.image(path, caption=cap, use_container_width=True)

st.divider()

# ── DATA TABLE ────────────────────────────────
st.subheader("Raw data table")
st.markdown(f"Showing **{len(df_f)}** clips. Click any column to sort.")
st.dataframe(
    df_f[['transcription_clean','duration','word_count',
          'gender','age','up_votes','pitch_mean','energy_mean']]
    .reset_index(drop=True),
    use_container_width=True,
    height=300
)
csv_bytes = df_f.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", csv_bytes,
                   "filtered_data.csv", "text/csv")
st.divider()
st.caption("Voice AI EDA Dashboard · Built with Streamlit")
