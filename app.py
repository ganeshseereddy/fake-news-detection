import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------------------------------

# Download NLTK Stopwords (for Streamlit Cloud)

# --------------------------------------------------

try:
stopwords.words("english")
except LookupError:
nltk.download("stopwords")

# --------------------------------------------------

# Page Configuration

# --------------------------------------------------

st.set_page_config(
page_title="AI Fake News Detection",
page_icon="📰",
layout="centered"
)

# --------------------------------------------------

# Load Model & Vectorizer

# --------------------------------------------------

model = pickle.load(
open("models/fake_news_model.pkl", "rb")
)

tfidf = pickle.load(
open("models/tfidf_vectorizer.pkl", "rb")
)

# --------------------------------------------------

# NLP Setup

# --------------------------------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# --------------------------------------------------

# Text Preprocessing Function

# --------------------------------------------------

def preprocess(text):

```
text = text.lower()

text = re.sub(r'[^a-zA-Z]', ' ', text)

words = text.split()

words = [
    stemmer.stem(word)
    for word in words
    if word not in stop_words
]

return " ".join(words)
```

# --------------------------------------------------

# Sidebar

# --------------------------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""

### Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Streamlit

### NLP Techniques

* Text Cleaning
* Stopword Removal
* Stemming
* TF-IDF Vectorization

### Machine Learning Model

* Random Forest Classifier

### Accuracy

~99%
""")

# --------------------------------------------------

# Main UI

# --------------------------------------------------

st.title("📰 AI-Powered Fake News Detection")

st.markdown("""

### Detect Fake News Using Machine Learning & NLP

This application analyzes news articles using:

✅ Natural Language Processing (NLP)

✅ TF-IDF Vectorization

✅ Random Forest Classifier

Enter a complete news article below and click **Predict News Type**.
""")

st.markdown("---")

# --------------------------------------------------

# User Input

# --------------------------------------------------

news = st.text_area(
"📝 Enter News Article",
height=250,
placeholder="Paste the complete news article here..."
)

# --------------------------------------------------

# Prediction

# --------------------------------------------------

if st.button("🔍 Predict News Type"):

```
if news.strip() == "":
    st.warning("⚠ Please enter a news article.")

elif len(news.split()) < 20:
    st.warning(
        "⚠ Please enter a complete news article (minimum 20 words)."
    )

else:

    cleaned_news = preprocess(news)

    transformed_news = tfidf.transform(
        [cleaned_news]
    )

    prediction = model.predict(
        transformed_news
    )

    prediction_proba = model.predict_proba(
        transformed_news
    )

    confidence = max(
        prediction_proba[0]
    ) * 100

    st.markdown("---")

    if prediction[0] == 1:
        st.success(
            "✅ Predicted as Real News"
        )
    else:
        st.error(
            "❌ Predicted as Fake News"
        )

    st.info(
        f"📊 Model Confidence: {confidence:.2f}%"
    )
```

# --------------------------------------------------

# Footer

# --------------------------------------------------

st.markdown("---")

st.caption(
"Built using NLP, TF-IDF, Random Forest, and Streamlit"
)
