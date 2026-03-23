import streamlit as st
import requests
import pandas as pd

API_URL = "https://sentiment-analysis-api-oegw.onrender.com"

st.set_page_config(page_title="Tweet Sentiment Analyser", page_icon="✈️", layout="wide")
st.title("✈️ Airline Tweet Sentiment Analyser")
st.markdown("Real-time sentiment classification + department routing powered by TF-IDF + LR — 76.9% accuracy on 14,640 tweets.")

try:
    requests.get(f"{API_URL}/health", timeout=5)
    st.success("API is live and healthy")
except:
    st.warning("API is waking up — first prediction may take 30 seconds.")

tab1, tab2, tab3 = st.tabs(["✍️ Single Tweet", "📋 Batch Analysis", "📊 How It Works"])

# ── TAB 1: Single Tweet ────────────────────────────────────────
with tab1:
    st.markdown("### Analyse a Tweet")
    tweet = st.text_area(
        "Enter a tweet:",
        placeholder="e.g. @UnitedAirlines lost my bag AGAIN. Crew was rude and flight delayed 3 hours!",
        height=100
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        analyse = st.button("🔍 Analyse", type="primary")
    with col2:
        if st.button("🎲 Random example"):
            examples = [
                "@UnitedAirlines lost my bag AGAIN. Crew was rude and flight delayed 3 hours!",
                "Just landed in NYC. Flight was on time and crew was super friendly 😊",
                "@Delta flight delayed by 30 mins. Not ideal but these things happen I guess.",
                "Never flying @SouthwestAir again. Rude staff and cramped seats. Awful!",
                "Upgraded to first class for free! @AmericanAir you made my day!",
                "The food was absolutely horrible but the seat was comfortable enough.",
            ]
            import random
            st.session_state['example_tweet'] = random.choice(examples)

    if 'example_tweet' in st.session_state:
        tweet = st.session_state['example_tweet']
        st.info(f"Example: {tweet}")

    if analyse and tweet.strip():
        with st.spinner("Analysing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": tweet},
                    timeout=30
                )
                result = response.json()
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        st.markdown("---")
        sentiment  = result["sentiment"]
        confidence = result["confidence"]
        probs      = result["probabilities"]
        absa       = result["absa_results"]
        depts      = result["departments_flagged"]
        summary    = result["summary"]

        # Sentiment banner
        if sentiment == "negative":
            st.error(f"😠 **NEGATIVE** — {confidence*100:.1f}% confidence")
        elif sentiment == "positive":
            st.success(f"😊 **POSITIVE** — {confidence*100:.1f}% confidence")
        else:
            st.warning(f"😐 **NEUTRAL** — {confidence*100:.1f}% confidence")

        # Probability bars
        st.markdown("### Sentiment Probabilities")
        c1, c2, c3 = st.columns(3)
        c1.metric("Negative", f"{probs['negative']*100:.1f}%")
        c2.metric("Neutral",  f"{probs['neutral']*100:.1f}%")
        c3.metric("Positive", f"{probs['positive']*100:.1f}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.progress(probs['negative'], text="Negative")
        with col2:
            st.progress(probs['neutral'], text="Neutral")
        with col3:
            st.progress(probs['positive'], text="Positive")

        # ABSA results
        if absa:
            st.markdown(f"### 🎯 Aspect-Based Analysis — {len(absa)} aspect(s) detected")

            df_absa = pd.DataFrame(absa)

            def color_sentiment(val):
                if val == "negative":
                    return "background-color: #ffcccc; color: #8b0000; font-weight: bold"
                elif val == "positive":
                    return "background-color: #ccffcc; color: #1a5c1a; font-weight: bold"
                else:
                    return "background-color: #fff3cc; color: #7d5a00; font-weight: bold"

            st.dataframe(
                df_absa.style.applymap(color_sentiment, subset=["sentiment"]),
                use_container_width=True, hide_index=True
            )

        # Department routing
        if depts:
            st.markdown("### 🚨 Departments to Action")
            cols = st.columns(len(depts))
            for i, dept in enumerate(depts):
                with cols[i]:
                    st.error(f"🔴 {dept}")
        else:
            st.success("✅ No departments flagged")

# ── TAB 2: Batch Analysis ──────────────────────────────────────
with tab2:
    st.markdown("### Batch Tweet Analysis")
    st.markdown("Enter multiple tweets — one per line. Get sentiment + department routing summary.")

    batch_input = st.text_area(
        "Enter tweets (one per line):",
        height=200,
        placeholder="@UnitedAirlines lost my bag again!\nFlight was amazing, crew so friendly!\nDelayed by 2 hours, no explanation given."
    )

    if st.button("🔍 Analyse Batch", type="primary"):
        tweets = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
        if not tweets:
            st.warning("Please enter at least one tweet.")
        else:
            with st.spinner(f"Analysing {len(tweets)} tweets..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"tweets": tweets},
                        timeout=60
                    )
                    result = response.json()
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    st.stop()

            predictions   = result["predictions"]
            sent_summary  = result["sentiment_summary"]
            dept_summary  = result["departments_to_action"]

            # Summary metrics
            st.markdown("### Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tweets", result["count"])
            c2.metric("Negative", sent_summary["negative"])
            c3.metric("Neutral",  sent_summary["neutral"])
            c4.metric("Positive", sent_summary["positive"])

            # Department action summary
            if dept_summary:
                st.markdown("### 🚨 Departments to Action")
                dept_df = pd.DataFrame(
                    list(dept_summary.items()),
                    columns=["Department", "Complaints"]
                ).sort_values("Complaints", ascending=False)
                st.dataframe(dept_df, use_container_width=True, hide_index=True)

            # Full results
            st.markdown("### Full Results")
            rows = []
            for p in predictions:
                rows.append({
                    "Tweet":      p["text"][:60] + "..." if len(p["text"]) > 60 else p["text"],
                    "Sentiment":  p["sentiment"].upper(),
                    "Confidence": f"{p['confidence']*100:.1f}%",
                    "Aspects":    len(p["absa_results"]),
                    "Depts Flagged": ", ".join(p["departments_flagged"]) if p["departments_flagged"] else "None"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("Download Results CSV", csv, "sentiment_results.csv", "text/csv")

# ── TAB 3: How It Works ────────────────────────────────────────
with tab3:
    st.markdown("### How This Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Model
        - TF-IDF vectorizer (15,000 features, trigrams)
        - Logistic Regression (C=1.0, saga solver)
        - Trained on 14,640 real airline tweets
        - 3-class: Negative / Neutral / Positive
        - Accuracy: **76.9%** vs 33% random baseline = **133% lift**

        #### Why TF-IDF beats BERT here
        - Domain-specific vocabulary (airline jargon)
        - TF-IDF captures in-domain terms better than base BERT
        - 10x faster inference, no GPU needed
        - Production-ready without heavy infrastructure
        """)
    with col2:
        st.markdown("""
        #### ABSA — Aspect-Based Sentiment Analysis
        Automatically detects which aspect of the experience is being discussed:

        | Aspect | Department |
        |--------|-----------|
        | Food/Meal | Catering |
        | Staff/Crew | HR / Customer Experience |
        | Bag/Luggage | Baggage Operations |
        | Delay/Cancel | Operations / Scheduling |
        | Seat/Comfort | Product / Fleet |
        | Price/Fee | Revenue Management |

        #### Use Case
        An airline's social media team can send 1,000 tweets to the batch endpoint
        and instantly know which departments need to respond — without reading every tweet.
        """)

    st.markdown("---")
    st.markdown("### Model Performance")
    perf_data = {
        "Class":     ["Negative", "Neutral", "Positive", "Overall"],
        "Precision": ["79%", "63%", "82%", "76.9%"],
        "Recall":    ["91%", "52%", "71%", "76.9%"],
        "F1":        ["85%", "57%", "76%", "76.9%"]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
