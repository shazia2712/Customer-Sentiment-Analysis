import pandas as pd
import streamlit as st
from textblob import TextBlob


# Page setup
st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)


# Sidebar
with st.sidebar:

    st.title("😊 Sentiment Analysis")

    st.write(
        "Analyze customer reviews "
        "using Natural Language Processing."
    )


# Sentiment function
def analyze_sentiment(text):

    blob = TextBlob(str(text))

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.05:
        sentiment = "Positive"

    elif polarity < -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment, polarity, subjectivity


# Main title
st.title("😊 Customer Sentiment Analysis")

st.write(
    "Analyze customer reviews using "
    "Natural Language Processing (NLP)."
)


# Tabs
tab1, tab2 = st.tabs([
    "✍️ Single Review",
    "📂 Test Reviews"
])


# Single Review
with tab1:

    st.header("Analyze a Customer Review")

    review = st.text_area(
        "Enter customer review:",
        placeholder=(
            "Example: The food was amazing "
            "and the staff were very friendly."
        ),
        height=150
    )


    if st.button("🔍 Analyze Sentiment"):

        if review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            sentiment, polarity, subjectivity = (
                analyze_sentiment(review)
            )


            if sentiment == "Positive":

                st.success(
                    f"😊 Sentiment: {sentiment}"
                )

            elif sentiment == "Negative":

                st.error(
                    f"😞 Sentiment: {sentiment}"
                )

            else:

                st.info(
                    f"😐 Sentiment: {sentiment}"
                )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Polarity",
                    f"{polarity:.2f}"
                )


            with col2:

                st.metric(
                    "Subjectivity",
                    f"{subjectivity:.2f}"
                )


# Test Reviews
with tab2:

    st.header("📂 Test Reviews")

    uploaded_file = st.file_uploader(
        "Upload TestReviews.csv",
        type=["csv"]
    )


    if uploaded_file:

        # Read CSV
        df = pd.read_csv(uploaded_file)


        # Check review column
        if "review" not in df.columns:

            st.error(
                "❌ Your CSV must contain a 'review' column."
            )

            st.stop()


        # Show original data
        st.subheader("📋 Original Data")

        st.dataframe(
            df.head(),
            use_container_width=True
        )


        # Analyze reviews
        if st.button("🚀 Analyze All Reviews"):

            results = df["review"].fillna("").apply(
                analyze_sentiment
            )


            # Add sentiment results
            df["Sentiment"] = results.apply(
                lambda x: x[0]
            )

            df["Polarity"] = results.apply(
                lambda x: x[1]
            )

            df["Subjectivity"] = results.apply(
                lambda x: x[2]
            )


            # Results
            st.subheader("📊 Sentiment Results")

            st.dataframe(
                df,
                use_container_width=True
            )


            # Count sentiments
            positive = (
                df["Sentiment"] == "Positive"
            ).sum()

            neutral = (
                df["Sentiment"] == "Neutral"
            ).sum()

            negative = (
                df["Sentiment"] == "Negative"
            ).sum()


            # Metrics
            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Total Reviews",
                    len(df)
                )


            with col2:

                st.metric(
                    "😊 Positive",
                    positive
                )


            with col3:

                st.metric(
                    "😐 Neutral",
                    neutral
                )


            with col4:

                st.metric(
                    "😞 Negative",
                    negative
                )


            # Sentiment distribution
            st.subheader(
                "📈 Sentiment Distribution"
            )

            sentiment_counts = (
                df["Sentiment"]
                .value_counts()
            )

            st.bar_chart(
                sentiment_counts
            )


            # Average polarity
            st.subheader(
                "📊 Average Polarity"
            )

            average_polarity = (
                df["Polarity"].mean()
            )

            st.metric(
                "Average Polarity",
                f"{average_polarity:.2f}"
            )


            # Download results
            csv = df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
