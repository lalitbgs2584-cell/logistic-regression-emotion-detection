import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="centered")

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Preprocessing helpers ─────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.translate(str.maketrans("", "", string.digits))
    text = text.encode("ascii", "ignore").decode("ascii")
    tokens = word_tokenize(text)
    return " ".join(w for w in tokens if w not in stop_words)

# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on emotion data…")
def train_model():
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

    # Build label map preserving original order
    unique_emotions = df["emotion"].unique().tolist()
    label_map = {e: i for i, e in enumerate(unique_emotions)}
    number_label = {v: k for k, v in label_map.items()}

    df["text"] = df["text"].apply(preprocess)
    df["label"] = df["emotion"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_bow, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_bow))
    return model, vectorizer, number_label, round(acc * 100, 2)

model, vectorizer, number_label, train_acc = train_model()

EMOTION_EMOJI = {
    "sadness": "😢", "anger": "😠", "love": "❤️",
    "surprise": "😲", "fear": "😨", "joy": "😄",
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 Emotion Classifier</h1>
        <p>Detect emotions in text using Bag-of-Words + Logistic Regression</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        <div class="sidebar-info">
        <b>Model:</b> Logistic Regression<br>
        <b>Features:</b> Bag of Words (CountVectorizer)<br>
        <b>Preprocessing:</b> lowercase → remove punctuation/numbers/emojis → stopword removal<br><br>
        <b>Emotions detected:</b><br>
        😢 Sadness &nbsp; 😠 Anger<br>
        ❤️ Love &nbsp;&nbsp;&nbsp;&nbsp; 😲 Surprise<br>
        😨 Fear &nbsp;&nbsp;&nbsp;&nbsp; 😄 Joy
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Model Accuracy", f"{train_acc}%")

# Input
user_input = st.text_area(
    "Enter your text below:",
    placeholder="e.g. I feel so happy and excited today!",
    height=130,
)

if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])
        pred_label = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        emotion = number_label[pred_label]
        emoji = EMOTION_EMOJI.get(emotion, "🤔")
        confidence = round(proba[pred_label] * 100, 1)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="emotion-emoji">{emoji}</div>
                <div class="emotion-label" style="color:#4F8BF9">{emotion.upper()}</div>
                <div class="confidence">Confidence: {confidence}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Probability breakdown")
        prob_df = pd.DataFrame({
            "Emotion": [number_label[i] for i in range(len(proba))],
            "Probability": proba,
        }).sort_values("Probability", ascending=False)

        for _, row in prob_df.iterrows():
            em = row["Emotion"]
            p = row["Probability"]
            st.markdown(f'<div class="prob-label">{EMOTION_EMOJI.get(em,"🤔")} {em.capitalize()}</div>', unsafe_allow_html=True)
            st.progress(float(p))
