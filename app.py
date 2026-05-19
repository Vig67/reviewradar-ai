import streamlit as st
import pandas as pd
from transformers import pipeline


# PAGE CONFIGURATION


st.set_page_config(
    page_title="ReviewRadar",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM STYLING


st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #333333;
}

h1, h2, h3 {
    color: white;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# HEADER SECTION


st.title("🤖 ReviewRadar")

st.markdown("""
### AI-Powered Sentiment Analytics Dashboard

ReviewRadar is an intelligent AI platform that analyzes customer reviews,
feedback, and text datasets using advanced Natural Language Processing models.

#### Features
✅ Real-time review analysis  
✅ Bulk CSV sentiment processing  
✅ Interactive analytics dashboard  
✅ Downloadable AI reports  
✅ Automatic text-column detection  
""")

st.markdown("---")

# LOAD AI MODEL


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

with st.spinner("🚀 Initializing AI Engine..."):
    classifier = load_model()


# SIDEBAR


with st.sidebar:

    st.header("📌 ReviewRadar")

    st.info("""
    This dashboard uses HuggingFace Transformers
    to classify customer sentiment instantly.
    """)

    st.markdown("---")

    st.write("### Supported Features")

    st.write("""
    - AI Sentiment Detection
    - CSV File Upload
    - Confidence Scores
    - Analytics Dashboard
    - Export Results
    """)


# CREATE TABS


tab1, tab2 = st.tabs([
    "🔮 Live Sentiment Sandbox",
    "📊 Bulk CSV Analytics"
])


# TAB 1 — SINGLE REVIEW ANALYSIS


with tab1:

    st.subheader("🔍 Real-Time Sentiment Analysis")

    st.write("""
    Enter any customer review, opinion, or feedback below.
    The AI model will predict whether the sentiment is positive or negative.
    """)

    user_input = st.text_area(
        "Enter Review Text:",
        placeholder="Example: The interface is beautiful but the customer support response time is disappointing..."
    )

    if st.button("Analyze Sentiment"):

        if user_input.strip() == "":

            st.warning("⚠ Please enter review text.")

        else:

            with st.spinner("Analyzing sentiment..."):

                result = classifier(user_input)[0]

                sentiment = result['label']
                confidence = round(result['score'] * 100, 2)

                st.markdown("---")

                if sentiment == "POSITIVE":

                    st.success("### 😊 Positive Sentiment Detected")

                else:

                    st.error("### 😠 Negative Sentiment Detected")

                st.info(f"Confidence Score: {confidence}%")


# TAB 2 — BULK CSV ANALYSIS


with tab2:

    st.subheader("📊 Bulk CSV Sentiment Analytics")

    st.write("""
    Upload ANY CSV file containing customer reviews or text data.

    ReviewRadar will automatically detect text columns
    and allow you to choose which column to analyze.
    """)

    uploaded_file = st.file_uploader(
        "📂 Upload CSV File",
        type=["csv"]
    )

    # PROCESS FILE
 

    if uploaded_file is not None:

        try:

            # READ CSV
            df = pd.read_csv(uploaded_file)

            st.success("✅ CSV Uploaded Successfully!")

            st.markdown("---")

            # DISPLAY PREVIEW
            st.write("### 📄 Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

           
            # DETECT TEXT COLUMNS
           

            text_columns = df.select_dtypes(
                include=['object']
            ).columns.tolist()

            if len(text_columns) == 0:

                st.error("❌ No text columns were found in this CSV file.")

            else:

                # COLUMN SELECTOR
                selected_column = st.selectbox(
                    "📝 Select the column containing reviews/text:",
                    text_columns
                )

                # RUN ANALYSIS
                if st.button("🚀 Run Bulk Analysis"):

                    with st.spinner("AI engine processing reviews..."):

                        sentiments = []
                        confidence_scores = []

                        for text in df[selected_column].fillna(""):

                            try:

                                result = classifier(str(text))[0]

                                sentiments.append(
                                    result['label'].capitalize()
                                )

                                confidence_scores.append(
                                    round(result['score'], 4)
                                )

                            except Exception:

                                sentiments.append("Error")
                                confidence_scores.append(0.0)

                        # ADD RESULTS TO DATAFRAME
                        df['Sentiment'] = sentiments
                        df['Confidence_Score'] = confidence_scores

                    # SUCCESS MESSAGE
                    st.success("✅ Analysis Completed Successfully!")

                    st.markdown("---")

                   
                    # METRICS
                  

                    total_reviews = len(df)

                    positive_reviews = len(
                        df[df['Sentiment'] == 'Positive']
                    )

                    negative_reviews = len(
                        df[df['Sentiment'] == 'Negative']
                    )

                    positive_ratio = round(
                        (positive_reviews / total_reviews) * 100,
                        1
                    ) if total_reviews > 0 else 0

                    # DISPLAY METRICS
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Reviews",
                        total_reviews
                    )

                    col2.metric(
                        "Positive Reviews",
                        positive_reviews
                    )

                    col3.metric(
                        "Positive Ratio",
                        f"{positive_ratio}%"
                    )

                    st.markdown("---")

                    
                    # CHARTS + TABLE
                 

                    chart_col1, chart_col2 = st.columns([1, 2])

                    with chart_col1:

                        st.write("### 📈 Sentiment Distribution")

                        sentiment_counts = (
                            df['Sentiment']
                            .value_counts()
                        )

                        st.bar_chart(sentiment_counts)

                    with chart_col2:

                        st.write("### 📋 Full Analysis Results")

                        st.dataframe(
                            df,
                            use_container_width=True
                        )

                    st.markdown("---")

                
                    # DOWNLOAD RESULTS
                  

                    csv_output = (
                        df.to_csv(index=False)
                        .encode('utf-8')
                    )

                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_output,
                        file_name="reviewradar_analysis_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(f"❌ Error reading CSV file: {e}")


st.markdown("---")

st.caption("🚀 ReviewRadar • Built with Streamlit + HuggingFace Transformers")
