import pandas as pd
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define the preprocessing function
def preprocess_review(text):
    if pd.isnull(text):
        return ""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)


# Load your dataset
df = pd.read_csv("reviews.csv")  # Make sure this file exists in the same folder

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_review)
def map_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(map_sentiment)

# Save the output
df.to_csv("chatgpt_reviews_cleaned.csv", index=False)

print("✅ Preprocessing complete. Cleaned data saved to 'chatgpt_reviews_cleaned.csv'")
