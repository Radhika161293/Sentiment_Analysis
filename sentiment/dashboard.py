
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(layout="wide")
st.title("🧠 ChatGPT Review Sentiment Dashboard")

# Load data
df = pd.read_csv("chatgpt_reviews_cleaned.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Add sentiment column if missing
if "sentiment" not in df.columns:
    def map_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"
    df["sentiment"] = df["rating"].apply(map_sentiment)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Ensure cleaned_review exists and is string
df["cleaned_review"] = df["cleaned_review"].astype(str)
df["predicted_sentiment"] = model.predict(vectorizer.transform(df["cleaned_review"]))

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filter Reviews")
    sentiment_filter = st.multiselect("Predicted Sentiment", ["Positive", "Neutral", "Negative"],
                                      default=["Positive", "Neutral", "Negative"])
    min_rating, max_rating = st.slider("Rating Range", 1, 5, (1, 5))
    platforms = df["platform"].dropna().unique().tolist()
    platform = st.multiselect("Platform", platforms, default=platforms)
    verified = st.radio("Verified Purchase", ["All", "Yes", "No"], index=0)

# Apply filters
filtered_df = df[
    (df["predicted_sentiment"].isin(sentiment_filter)) &
    (df["rating"].between(min_rating, max_rating)) &
    (df["platform"].isin(platform))
]
if verified != "All":
    filtered_df = filtered_df[filtered_df["verified_purchase"] == verified]

# Rating distribution
st.subheader("📊 Review Rating Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="rating", palette="coolwarm", ax=ax1)
st.pyplot(fig1)

# Helpful reviews
st.subheader("👍 Helpful Reviews (Votes > 10)")
st.metric("Helpful Reviews", (df["helpful_votes"] > 10).sum())

# Word Clouds
st.subheader("☁️ Word Clouds by Sentiment")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Positive Reviews**")
    pos_text = " ".join(df[df["predicted_sentiment"] == "Positive"]["cleaned_review"])
    if pos_text:
        st.image(WordCloud(width=400, height=300, background_color="white").generate(pos_text).to_array())

with col2:
    st.markdown("**Negative Reviews**")
    neg_text = " ".join(df[df["predicted_sentiment"] == "Negative"]["cleaned_review"])
    if neg_text:
        st.image(WordCloud(width=400, height=300, background_color="black", colormap="Reds").generate(neg_text).to_array())

# Average rating over time
st.subheader("📆 Average Rating Over Time")
df_time = df.groupby("date")["rating"].mean().reset_index()
fig2, ax2 = plt.subplots()
sns.lineplot(data=df_time, x="date", y="rating", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Model Performance
st.subheader("📋 Model Performance")
y_true = df["sentiment"].astype(str)
y_pred = df["predicted_sentiment"].astype(str)
acc = accuracy_score(y_true, y_pred)
st.metric("Accuracy", f"{acc*100:.2f}%")
st.json(classification_report(y_true, y_pred, output_dict=True))

# Live review prediction
st.subheader("💬 Live Review Sentiment Prediction")
user_review = st.text_area("Enter a review:")
if user_review:
    input_vec = vectorizer.transform([user_review])
    predicted = model.predict(input_vec)[0]
    st.success(f"Predicted Sentiment: **{predicted}**")
