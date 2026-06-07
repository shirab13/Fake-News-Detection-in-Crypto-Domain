import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sqlalchemy import create_engine

# ─── Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector - Crypto",
    page_icon="🔍",
    layout="wide"
)

DB_URL        = os.getenv("DB_URL",        "postgresql://admin:secret123@db:5432/fakenews")
SCRAPER_URL   = os.getenv("SCRAPER_URL",   "http://scraper:8000")
MODEL_URL     = os.getenv("MODEL_URL",     "http://model:8002")
STATS_URL     = os.getenv("STATS_URL",     "http://stats:8003")
SENTIMENT_URL = os.getenv("SENTIMENT_URL", "http://sentiment:8004")
TOPIC_URL     = os.getenv("TOPIC_URL",     "http://topic:8005")
GEMINI_URL    = os.getenv("GEMINI_URL",    "http://gemini:8006")


# ─── DB helper ───────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    try:
        engine = create_engine(DB_URL)
        df = pd.read_sql("SELECT * FROM posts ORDER BY scraped_at DESC LIMIT 5000", engine)
        return df
    except Exception as e:
        st.warning(f"DB not available: {e}. Showing demo data.")
        return _demo_data()


def _demo_data():
    import numpy as np
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "title":                 [f"Sample crypto title #{i}" for i in range(n)],
        "subreddit":             np.random.choice(["CryptoCurrency","Bitcoin","Buttcoin"], n),
        "label_pred":            np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "label_prob":            np.random.uniform(0, 1, n),
        "sentiment_polarity":    np.random.uniform(-1, 1, n),
        "sentiment_subjectivity":np.random.uniform(0, 1, n),
        "caps_ratio":            np.random.uniform(0, 0.5, n),
        "word_count":            np.random.randint(5, 20, n),
        "exclamation_count":     np.random.randint(0, 4, n),
        "topic":                 np.random.choice(list(["Price & Market","Scam & Fraud","Regulation & Government","Technology & Development"]), n),
        "scraped_at":            pd.date_range("2024-01-01", periods=n, freq="h"),
    })


def call_service(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.json()
    except:
        return {"error": "Service unavailable"}


# ─── UI ──────────────────────────────────────────────────
st.title("🔍 Fake News Detector - Crypto Domain")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔎 Analyze Post", "⚙️ Scraper"])


# ══════════════════════════════════════════════════════════
# TAB 1: Dashboard
# ══════════════════════════════════════════════════════════
with tab1:
    df = load_data()

    if df.empty:
        st.info("No data yet. Go to the Scraper tab to collect data.")
        st.stop()

    # ── KPIs ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total      = len(df)
    fake_count = (df["label_pred"] == 0).sum() if "label_pred" in df.columns else 0
    real_count = (df["label_pred"] == 1).sum() if "label_pred" in df.columns else 0
    fake_pct   = fake_count / total * 100 if total > 0 else 0

    col1.metric("סה\"כ פוסטים",  f"{total:,}")
    col2.metric("FAKE",           f"{fake_count:,}", f"{fake_pct:.1f}%")
    col3.metric("REAL",           f"{real_count:,}", f"{100-fake_pct:.1f}%")
    col4.metric("Subreddits",     df["subreddit"].nunique() if "subreddit" in df.columns else "-")

    st.markdown("---")

    # ── Row 1: Label distribution + Subreddit breakdown ─
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("התפלגות FAKE vs REAL")
        if "label_pred" in df.columns:
            counts = df["label_pred"].map({0: "FAKE", 1: "REAL"}).value_counts()
            fig = px.pie(values=counts.values, names=counts.index,
                         color_discrete_map={"FAKE": "#e74c3c", "REAL": "#2ecc71"},
                         hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("FAKE vs REAL לפי Subreddit")
        if "subreddit" in df.columns and "label_pred" in df.columns:
            sub_df = df.groupby(["subreddit", "label_pred"]).size().reset_index(name="count")
            sub_df["label"] = sub_df["label_pred"].map({0: "FAKE", 1: "REAL"})
            fig = px.bar(sub_df, x="subreddit", y="count", color="label",
                         color_discrete_map={"FAKE": "#e74c3c", "REAL": "#2ecc71"},
                         barmode="group")
            fig.update_layout(margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Sentiment ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Polarity: FAKE vs REAL")
        if all(c in df.columns for c in ["sentiment_polarity", "label_pred"]):
            fig = px.histogram(
                df, x="sentiment_polarity",
                color=df["label_pred"].map({0: "FAKE", 1: "REAL"}),
                barmode="overlay", opacity=0.7,
                color_discrete_map={"FAKE": "#e74c3c", "REAL": "#2ecc71"},
                labels={"color": "Label"}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
            fig.update_layout(margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Subjectivity: FAKE vs REAL")
        if all(c in df.columns for c in ["sentiment_subjectivity", "label_pred"]):
            fig = px.box(
                df, x=df["label_pred"].map({0: "FAKE", 1: "REAL"}),
                y="sentiment_subjectivity",
                color=df["label_pred"].map({0: "FAKE", 1: "REAL"}),
                color_discrete_map={"FAKE": "#e74c3c", "REAL": "#2ecc71"},
                labels={"x": "Label", "y": "Subjectivity"}
            )
            fig.update_layout(margin=dict(t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Topic + CAPS ──────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Topic Distribution")
        if "topic" in df.columns and df["topic"].notna().any():
            topic_counts = df["topic"].value_counts().head(8)
            topic_df = pd.DataFrame({'topic': topic_counts.index, 'count': topic_counts.values})
            fig = px.bar(topic_df, x='count', y='topic',
                         orientation='h', color='count',
                         color_continuous_scale="Blues")
            fig.update_layout(margin=dict(t=0, b=0), showlegend=False,
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("אין נתוני Topic עדיין")

    with col2:
        st.subheader("CAPS Ratio: FAKE vs REAL")
        if all(c in df.columns for c in ["caps_ratio", "label_pred"]):
            fig = px.violin(
                df, x=df["label_pred"].map({0: "FAKE", 1: "REAL"}),
                y="caps_ratio",
                color=df["label_pred"].map({0: "FAKE", 1: "REAL"}),
                color_discrete_map={"FAKE": "#e74c3c", "REAL": "#2ecc71"},
                box=True,
                labels={"x": "Label", "y": "CAPS Ratio"}
            )
            fig.update_layout(margin=dict(t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Recent posts table ───────────────────────────────
    st.subheader("פוסטים אחרונים")
    display_cols = [c for c in ["title", "subreddit", "label_pred", "sentiment_polarity", "topic", "scraped_at"]
                    if c in df.columns]
    display_df = df[display_cols].head(20).copy()
    if "label_pred" in display_df.columns:
        display_df["label_pred"] = display_df["label_pred"].map({0: "🔴 FAKE", 1: "🟢 REAL"})
    st.dataframe(display_df, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2: Analyze a single post
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔎 ניתוח פוסט בודד")

    input_type = st.radio("סוג קלט:", ["טקסט חופשי", "Reddit URL"], horizontal=True)

    if input_type == "טקסט חופשי":
        user_text = st.text_area("הכניסי כותרת או טקסט:", height=80,
                                  placeholder="Bitcoin to reach $1M by end of year...")
    else:
        user_url = st.text_input("Reddit URL:", placeholder="https://www.reddit.com/r/.../")
        user_text = ""

    if st.button("נתח", type="primary"):
        if input_type == "טקסט חופשי":
            text_to_analyze = user_text
        else:
            text_to_analyze = user_url

        if not text_to_analyze:
            st.warning("נא להכניס טקסט או URL")
        else:
            if input_type == "Reddit URL":
                with st.spinner("שולף פוסט מ-Reddit..."):
                    try:
                        r = requests.post(f"{SCRAPER_URL}/scrape/url",
                                          params={"url": text_to_analyze}, timeout=15)
                        result = r.json()
                        if "error" in result:
                            st.error(f"שגיאה בשליפה: {result['error']}")
                            st.stop()
                        text_to_analyze = result.get("title", text_to_analyze)
                        st.info(f"כותרת הפוסט: **{text_to_analyze}**")
                    except Exception as e:
                        st.error(f"לא ניתן לשלוף את הפוסט: {e}")
                        st.stop()

            with st.spinner("מנתח..."):
                col1, col2, col3 = st.columns(3)

                # Model prediction
                model_res = call_service(f"{MODEL_URL}/predict/all", {"text": text_to_analyze})
                # Sentiment
                sent_res  = call_service(f"{SENTIMENT_URL}/analyze",  {"text": text_to_analyze})
                # Stats
                stats_res = call_service(f"{STATS_URL}/analyze",      {"text": text_to_analyze})
                # Topic
                topic_res = call_service(f"{TOPIC_URL}/analyze",      {"text": text_to_analyze})
                # Gemini
                gemini_res= call_service(f"{GEMINI_URL}/analyze",     {"text": text_to_analyze})

            # ── תצוגת תוצאות ────────────────────────────
            verdict = model_res.get("majority_vote", {}).get("label_text", "?")
            color   = "🔴" if verdict == "FAKE" else "🟢"
            st.markdown(f"## {color} פסיקה: **{verdict}**")

            col1, col2, col3, col4 = st.columns(4)

            lstm_label = model_res.get("lstm", {}).get("label_text", "?")
            svm_label  = model_res.get("svm",  {}).get("label_text", "?")
            lr_label   = model_res.get("lr",   {}).get("label_text", "?")
            gemini_v   = gemini_res.get("verdict", "?")

            col1.metric("LSTM",   lstm_label)
            col2.metric("SVM",    svm_label)
            col3.metric("LR",     lr_label)
            col4.metric("Gemini", gemini_v)

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment")
                tb = sent_res.get("textblob", {})
                st.write(f"**Polarity:** {tb.get('polarity', '?')} ({tb.get('sentiment', '?')})")
                st.write(f"**Subjectivity:** {tb.get('subjectivity', '?')} ({sent_res.get('subjectivity_label', '?')})")
                vader = sent_res.get("vader", {})
                st.write(f"**VADER Compound:** {vader.get('compound', '?')}")

                st.subheader("Topic")
                st.write(f"**Primary:** {topic_res.get('primary_topic', '?')}")
                st.write(f"**Strength:** {topic_res.get('match_strength', '?')}")

            with col2:
                st.subheader("מדדי טקסט")
                st.write(f"**מילים:** {stats_res.get('word_count', '?')}")
                st.write(f"**CAPS Ratio:** {stats_res.get('caps_ratio', '?')}")
                st.write(f"**סימני קריאה:** {stats_res.get('exclamation_count', '?')}")
                signals = stats_res.get("clickbait_signals", {})
                active  = [k for k, v in signals.items() if v]
                if active:
                    st.warning(f"⚠️ Clickbait signals: {', '.join(active)}")

                st.subheader("Gemini Analysis")
                st.write(f"**Verdict:** {gemini_res.get('verdict', '?')}")
                st.write(f"**Confidence:** {gemini_res.get('confidence', '?')}")
                st.write(f"**Explanation:** {gemini_res.get('explanation', '?')}")
                flags = gemini_res.get("red_flags", [])
                if flags:
                    st.write(f"**Red flags:** {', '.join(flags)}")


# ══════════════════════════════════════════════════════════
# TAB 3: Scraper Control
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚙️ שליטה ב-Scraper")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**הפעלה ידנית**")
        if st.button("סרוק Reddit עכשיו", type="primary"):
            with st.spinner("סורק Reddit..."):
                r = call_service(f"{SCRAPER_URL}/scrape/now", {})
                st.success(r.get("message", "Started"))

        if st.button("סרוק חדשות קריפטו (RSS) עכשיו", type="primary"):
            with st.spinner("שולף כתבות מ-CoinDesk, CoinTelegraph..."):
                r = call_service(f"{SCRAPER_URL}/scrape/rss", {})
                sources = r.get("sources", [])
                st.success(f"✅ {r.get('message', 'Started')} | מקורות: {', '.join(sources)}")

        st.markdown("**סטטיסטיקות Scraper**")
        if st.button("רענן סטטיסטיקות"):
            try:
                r = requests.get(f"{SCRAPER_URL}/stats", timeout=5)
                stats = r.json()
                st.json(stats)
            except:
                st.error("Scraper not available")

    with col2:
        st.markdown("**סטטוס סרוויסים**")
        services = {
            "Scraper":      SCRAPER_URL,
            "Preprocessor": "http://preprocessor:8001",
            "Model":        MODEL_URL,
            "Stats":        STATS_URL,
            "Sentiment":    SENTIMENT_URL,
            "Topic":        TOPIC_URL,
            "Gemini":       GEMINI_URL,
        }
        for name, url in services.items():
            try:
                r = requests.get(url, timeout=3)
                st.success(f"✅ {name}")
            except:
                st.error(f"❌ {name}")
