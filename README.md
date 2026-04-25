# Voice AI EDA Dashboard

An end-to-end exploratory data analysis project built on real Turkish voice recordings from the Mozilla CommonVoice 17 dataset. The goal was to go beyond a simple Jupyter notebook — extract meaningful audio features using AI tools, analyse patterns across 500 clips, and present everything through a live interactive web dashboard that anyone can open in a browser.

---

## Live Dashboard

[Open the dashboard here](https://voice-ai-eda-dashboard-fzn9bkspwevhjw8ybmpr4z.streamlit.app)


---

## Why I built this

Most data analytics portfolios stop at CSV analysis and bar charts. I wanted to work with something harder — raw audio files — and show how AI libraries like librosa and Hugging Face fit into a real data workflow. This project covers the complete pipeline from loading a speech dataset to deploying a working web application.

---

## What AI was used and how

### Hugging Face — loading the dataset

Instead of downloading files manually, I used the Hugging Face `datasets` library to stream real voice recordings directly into Python. This is how production AI teams load training data for models like OpenAI Whisper and Google Speech-to-Text.

```python
from datasets import load_dataset
dataset = load_dataset("ysdede/commonvoice_17_tr_fixed", split="train")
```

### librosa — audio feature extraction

librosa is the standard Python library for audio analysis. It is used in speech recognition research and by companies building voice AI systems. I used it to extract four features from every audio clip:

| Feature | What it captures |
|---|---|
| MFCC (13 values) | The tone and texture of the voice — works like a fingerprint |
| Pitch | How high or low the speaker's voice is, measured in Hz |
| RMS Energy | How loud the clip is on average |
| Zero Crossing Rate | How noisy or smooth the audio signal is |

These are the same features used when training speaker identification and emotion detection models. Extracting and analysing them before model training is standard data analyst work at AI companies.

```python
mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
pitch  = librosa.yin(audio, fmin=50, fmax=400)
rms    = librosa.feature.rms(y=audio)
zcr    = librosa.feature.zero_crossing_rate(audio)
```

### NLP — transcription text analysis

I cleaned 500 Turkish transcriptions using regex, then ran a word frequency analysis to find which words appear most across all clips. The result feeds into an interactive word cloud in the dashboard.

```python
df['transcription_clean'] = df['transcription'].str.lower()
df['transcription_clean'] = df['transcription_clean'].str.replace(
    r'[^a-zçğışöü\s]', '', regex=True
)
```

### Plotly and Seaborn — analytical visualisation

The correlation heatmap shows which audio features are statistically linked — this is a core step in deciding which features to include when training a machine learning model. The scatter plot uses an OLS regression trendline to show the relationship between clip duration and word count.

### Streamlit — deployment

Streamlit converts a Python script into a live web application. I used `@st.cache_data` to cache the dataset in memory so filters update charts without reloading the file each time. The app is deployed on Streamlit Cloud with a permanent public URL.

---

## Project overview

| Detail | Value |
|---|---|
| Dataset | Mozilla CommonVoice 17 Turkish |
| Source | Hugging Face — ysdede/commonvoice_17_tr_fixed |
| Total clips in dataset | 91,135 |
| Clips analysed | 500 |
| Columns after processing | 23 (7 original + 16 extracted) |
| AI tools used | librosa, Hugging Face datasets, NLP regex, OLS regression |
| Deployment | Streamlit Cloud |

---

## Project structure

```
VoiceAI_Dashboard/
│
├── app.py                   — Streamlit dashboard
├── requirements.txt         — Python dependencies
│
├── data/
│   └── commonvoice_cleaned_features.csv
│
└── charts/
    ├── chart1_duration_distribution.png
    ├── chart2_demographics.png
    ├── chart3_waveform_spectrogram.png
    ├── chart4_mfcc_heatmap.png
    ├── chart5_correlation_heatmap.png
    ├── chart7_pitch_by_age.png
    └── chart8_wordcloud.png
```

---

## How the data flows

```
Raw audio clips on Hugging Face
        |
        v
Load 500 Turkish voice recordings using datasets library
        |
        v
Clean missing values, fix data types, clean transcription text
        |
        v
Extract MFCC, pitch, energy, ZCR from each audio file using librosa
        |
        v
Run NLP word frequency analysis on all transcriptions
        |
        v
Build 8 EDA charts covering distributions, correlations, and outliers
        |
        v
Deploy as a live interactive Streamlit dashboard
```

---

## Dashboard features

**Sidebar filters** — filter all charts simultaneously by gender, age group, and clip duration.

**KPI cards** — total clips visible, average duration, average word count, top gender.

**8 charts across the dashboard:**

| Chart | What it shows |
|---|---|
| Duration histogram | How clip lengths are distributed across the dataset |
| Gender bar chart | Which gender contributed the most recordings |
| Age group bar chart | Age group representation in the dataset |
| Scatter plot | Relationship between clip duration and word count, with regression line |
| Correlation heatmap | Which audio features are statistically related |
| Pitch box plot | Pitch variation across age groups, with outlier detection |
| Waveform and spectrogram | Raw audio signal visualised using librosa |
| Word cloud | Most frequent words across all 500 Turkish transcriptions |

**Raw data table** — sortable, filterable, with a download button for the filtered CSV.

---

## How to run locally

```bash
git clone https://github.com/arunnn9088/voice-ai-eda-dashboard.git
cd voice-ai-eda-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Tech stack

**Data and AI**
- pandas — data manipulation and cleaning
- numpy — numerical operations on audio arrays
- librosa — audio feature extraction
- scikit-learn — label encoding and preprocessing
- Hugging Face datasets — loading the voice dataset
- re (regex) — NLP text cleaning for Turkish

**Visualisation**
- plotly express — interactive charts with hover and zoom
- seaborn — heatmaps and statistical plots
- matplotlib — waveform, spectrogram, and MFCC rendering
- wordcloud — word frequency visualisation

**Application**
- streamlit — web dashboard framework
- Streamlit Cloud — free deployment
- Google Colab — cloud environment for data processing
- VS Code — local development

---

## What this type of analysis is used for

Before any speech AI model is trained, a data analyst has to understand the dataset — how long are the clips, are certain speaker groups underrepresented, what do the audio features look like, are there outliers that could affect training quality. This is the work that happens at companies like Mozilla (CommonVoice), call centre AI platforms, healthcare voice tools, and any product that trains speech recognition models. This dashboard replicates that workflow end to end.

---

## What comes next

The next phase adds a voice query layer to the dashboard:

- User speaks a question through the microphone
- OpenAI Whisper converts speech to text
- An LLM reads the current dashboard data and generates an answer
- The answer is spoken back using text-to-speech

This turns the dashboard from a static analytics tool into a voice-powered data assistant.

---

## About

**Arunachalaeswar**
Data Analyst — Python, Audio AI, Streamlit

aruneswar1912@gmail.com
[LinkedIn](www.linkedin.com/in/arunachalaeswar)

---

*Built as a portfolio project. All data is from the Mozilla CommonVoice open dataset.*
