import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Montra Guest Experience Dashboard", layout="wide")

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None

st.markdown("""
<style>
.stApp { background:#f5f7fa; }
.block-container { padding-top:2rem; }
.dashboard-title { font-size:32px; font-weight:700; color:#172033; }
.dashboard-subtitle { font-size:15px; color:#687386; margin-bottom:28px; }
.section-title { font-size:21px; font-weight:700; color:#172033; margin:20px 0 15px; }
.metric-card { background:white; border:1px solid #e4e8ee; border-radius:12px; padding:18px; min-height:105px; }
.metric-label { color:#687386; font-size:14px; font-weight:600; }
.metric-value { color:#172033; font-size:28px; font-weight:700; margin-top:8px; }
div.stButton > button, div.stDownloadButton > button { border-radius:8px; min-height:42px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Montra Dashboard")
st.sidebar.caption("Guest Experience & Sentiment Analysis")
st.sidebar.divider()

st.session_state.api_url = st.sidebar.text_input(
    "FastAPI URL",
    value=st.session_state.api_url
).rstrip("/")

if st.sidebar.button("Check API Status", use_container_width=True):
    try:
        response = requests.get(f"{st.session_state.api_url}/", timeout=10)
        if response.ok:
            st.sidebar.success("API is online")
        else:
            st.sidebar.error(f"API returned {response.status_code}")
    except requests.RequestException as error:
        st.sidebar.error(f"Connection failed: {error}")

st.markdown('<div class="dashboard-title">Montra Guest Experience Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Hotel review sentiment analysis and guest experience performance monitoring</div>', unsafe_allow_html=True)

single_tab, batch_tab = st.tabs(["Single Review", "Batch Analysis"])

with single_tab:
    st.markdown('<div class="section-title">Guest Review Prediction</div>', unsafe_allow_html=True)
    st.info("Only the guest feedback is sent to the sentiment model. Hotel name and rating are used for dashboard information.")

    with st.form("single_review_form"):
        col1, col2 = st.columns(2)
        with col1:
            hotel_name = st.text_input("Hotel Name", placeholder="Enter hotel name")
        with col2:
            rating = st.selectbox("Guest Rating", [1, 2, 3, 4, 5], index=4)
        review_text = st.text_area("Guest Feedback", height=180, placeholder="Enter guest feedback here...")
        submit_review = st.form_submit_button("Analyse Review", type="primary", use_container_width=True)

    if submit_review:
        if not hotel_name.strip():
            st.warning("Please enter the hotel name.")
        elif not review_text.strip():
            st.warning("Please enter the guest feedback.")
        else:
            try:
                with st.spinner("Analysing guest feedback..."):
                    response = requests.post(
                        f"{st.session_state.api_url}/predict",
                        json={"text": review_text.strip()},
                        timeout=60
                    )
                if response.ok:
                    result = response.json()
                    st.session_state.single_result = {
                        "hotel": hotel_name.strip(),
                        "rating": rating,
                        "review": review_text.strip(),
                        "label": str(result.get("label", "Unknown")).title(),
                        "confidence": float(result.get("confidence", 0))
                    }
                else:
                    st.error(f"Prediction failed: {response.text}")
            except requests.RequestException as error:
                st.error(f"Could not connect to FastAPI: {error}")

    if st.session_state.single_result:
        result = st.session_state.single_result
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Hotel", result["hotel"])
        with c2:
            st.metric("Guest Rating", f'{result["rating"]}/5')
        with c3:
            st.metric("Sentiment", result["label"])
        confidence = max(0.0, min(1.0, result["confidence"]))
        st.progress(confidence, text=f"Prediction confidence: {confidence:.1%}")
        st.text_area("Guest Feedback", value=result["review"], height=120, disabled=True)

with batch_tab:
    st.markdown('<div class="section-title">Batch Sentiment Analysis</div>', unsafe_allow_html=True)
    st.write("CSV must contain hotel_name and review_text. A rating column is recommended.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            required = {"hotel_name", "review_text"}
            missing = required - set(raw_df.columns)

            if missing:
                st.error("Missing required column(s): " + ", ".join(sorted(missing)))
            else:
                st.markdown('<div class="section-title">Uploaded Data</div>', unsafe_allow_html=True)
                st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

                if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Running batch prediction..."):
                            response = requests.post(
                                f"{st.session_state.api_url}/predict/batch",
                                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                                timeout=300
                            )

                        if response.ok:
                            data = response.json()
                            if isinstance(data, dict) and "error" in data:
                                st.error(data["error"])
                            elif isinstance(data, list):
                                result_df = pd.DataFrame(data)
                                if "sentiment_label" in result_df.columns:
                                    result_df["sentiment_label"] = (
                                        result_df["sentiment_label"].astype(str)
                                        .str.replace("Error: ", "", regex=False)
                                        .str.title()
                                    )
                                if "sentiment_confidence" in result_df.columns:
                                    result_df["sentiment_confidence"] = pd.to_numeric(
                                        result_df["sentiment_confidence"], errors="coerce"
                                    ).fillna(0)
                                if "rating" in result_df.columns:
                                    result_df["rating"] = pd.to_numeric(
                                        result_df["rating"], errors="coerce"
                                    )
                                st.session_state.batch_df = result_df
                                st.success(f"Batch prediction completed for {len(result_df):,} reviews.")
                            else:
                                st.error("Unexpected API response.")
                        else:
                            st.error(f"Batch prediction failed: {response.text}")
                    except requests.RequestException as error:
                        st.error(f"Could not connect to FastAPI: {error}")
                    except Exception as error:
                        st.error(f"Batch processing error: {error}")
        except Exception as error:
            st.error(f"Could not read CSV: {error}")

    if st.session_state.batch_df is not None:
        df = st.session_state.batch_df.copy()

        total = len(df)
        if "sentiment_label" in df.columns:
            positive = int(df["sentiment_label"].str.lower().eq("positive").sum())
            negative = int(df["sentiment_label"].str.lower().eq("negative").sum())
            neutral = int(df["sentiment_label"].str.lower().eq("neutral").sum())
        else:
            positive = negative = neutral = 0

        average_rating = None
        if "rating" in df.columns and df["rating"].notna().any():
            average_rating = float(df["rating"].mean())

        st.markdown('<div class="section-title">Overall Performance</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Total Reviews", f"{total:,}")
        with c2: st.metric("Positive Reviews", f"{positive:,}")
        with c3: st.metric("Negative Reviews", f"{negative:,}")
        with c4: st.metric("Neutral Reviews", f"{neutral:,}")
        with c5: st.metric("Average Rating", f"{average_rating:.2f}/5" if average_rating is not None else "N/A")

        if "hotel_name" in df.columns:
            st.markdown('<div class="section-title">Hotel Performance</div>', unsafe_allow_html=True)

            summary = df.groupby("hotel_name", dropna=False).size().reset_index(name="total_reviews")

            if "rating" in df.columns:
                ratings = df.groupby("hotel_name", dropna=False)["rating"].mean().reset_index(name="average_rating")
                summary = summary.merge(ratings, on="hotel_name", how="left")
            else:
                summary["average_rating"] = None

            if "sentiment_label" in df.columns:
                counts = pd.crosstab(df["hotel_name"], df["sentiment_label"]).reset_index()
                counts.columns = [
                    str(x).lower() if x != "hotel_name" else x
                    for x in counts.columns
                ]
                summary = summary.merge(counts, on="hotel_name", how="left")

            for label in ["positive", "negative", "neutral"]:
                if label not in summary.columns:
                    summary[label] = 0
                summary[label] = summary[label].fillna(0).astype(int)

            summary["positive_rate"] = (
                summary["positive"] / summary["total_reviews"].replace(0, 1) * 100
            )
            summary["negative_rate"] = (
                summary["negative"] / summary["total_reviews"].replace(0, 1) * 100
            )
            summary["average_rating"] = pd.to_numeric(
                summary["average_rating"], errors="coerce"
            ).round(2)

            st.dataframe(
                summary.sort_values("total_reviews", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "hotel_name": "Hotel",
                    "total_reviews": "Total Reviews",
                    "average_rating": st.column_config.NumberColumn("Average Rating", format="%.2f"),
                    "positive": "Positive",
                    "negative": "Negative",
                    "neutral": "Neutral",
                    "positive_rate": st.column_config.NumberColumn("Positive Rate (%)", format="%.1f"),
                    "negative_rate": st.column_config.NumberColumn("Negative Rate (%)", format="%.1f")
                }
            )

            st.markdown('<div class="section-title">Reviews by Hotel and Sentiment</div>', unsafe_allow_html=True)
            sentiment_data = summary.set_index("hotel_name")[["positive", "negative", "neutral"]]
            st.bar_chart(sentiment_data, use_container_width=True)

            if "rating" in df.columns:
                st.markdown('<div class="section-title">Average Rating by Hotel</div>', unsafe_allow_html=True)
                rating_data = summary[["hotel_name", "average_rating"]].dropna().set_index("hotel_name")
                st.bar_chart(rating_data, use_container_width=True)

        st.markdown('<div class="section-title">Overall Sentiment Distribution</div>', unsafe_allow_html=True)
        distribution = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Reviews": [positive, negative, neutral]
        }).set_index("Sentiment")
        st.bar_chart(distribution, use_container_width=True)

        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Prediction Results",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="montra_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
