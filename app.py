import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import os
import pandas as pd
import os


# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load model and vectorizer
model_path = "spam_model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model files not found. Please train the model first using train.py.")
    st.stop()

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def save_feedback(message, predicted_label, correct_label):
    feedback_file = "user_feedback.csv"
    feedback_data = {
        "message": [message],
        "predicted_label": [predicted_label],
        "correct_label": [correct_label]
    }
    df = pd.DataFrame(feedback_data)

    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, index=False)
# # Theme toggle
# if "dark_mode" not in st.session_state:
#     st.session_state.dark_mode = False

# toggle = st.checkbox("🌗 Dark Mode", value=st.session_state.dark_mode)

# st.session_state.dark_mode = toggle

# Toggle for Dark Mode
dark_mode = st.checkbox("🌗 Dark Mode", value=True)

# Theme Application
def apply_theme(dark):
    if dark:
        st.markdown("""
            <style>
            html, body, .main {
                background-color: #0e1117;
                color: #FFFFFF;
            }
            textarea, input, .stTextInput > div > div > input {
                background-color: #1e1e1e !important;
                color: white !important;
                border: 1px solid #444;
            }
            .stSlider > div {
                background-color: #1e1e1e;
                color: white;
            }
            .stButton > button {
                background-color: #444;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            html, body, .main {
                background-color: #ffffff;
                color: #000000;
            }
            textarea, input, .stTextInput > div > div > input {
                background-color: #f5f5f5 !important;
                color: black !important;
                border: 1px solid #ccc;
            }
            .stSlider > div {
                background-color: #ffffff;
                color: black;
            }
            .stButton > button {
                background-color: #e0e0e0;
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

# Apply the selected theme
apply_theme(dark_mode)


# Streamlit App UI
st.title("📩 Spam Message Classifier")
st.write("Enter an SMS message to check if it's spam. Adjust the confidence threshold below.")

# User input
user_input = st.text_area("Enter your message:", height=100)

# Threshold slider
threshold = st.slider("Set Spam Confidence Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)


# Predict button
if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")

    else:
        processed = preprocess_text(user_input)
        vect_input = vectorizer.transform([processed])
        probas = model.predict_proba(vect_input)[0]
        spam_prob = probas[1]  # spam class probability

        if spam_prob >= threshold:
            prediction_label = "Spam"
            st.error(f"🚨 Spam detected! (Confidence: {spam_prob:.2f})")
        else:
            prediction_label = "Not Spam"
            st.success(f"✅ Message is not spam. (Confidence: {spam_prob:.2f})")

        st.caption(f"Threshold: {threshold} | P(Spam): {spam_prob:.2f})")

        # --- Feedback Mechanism ---
        import pandas as pd
        import os

        def save_feedback(message, predicted_label, correct_label):
            feedback_file = "user_feedback.csv"
            feedback_data = {
                "message": [message],
                "predicted_label": [predicted_label],
                "correct_label": [correct_label]
            }
            df = pd.DataFrame(feedback_data)

            if os.path.exists(feedback_file):
                df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                df.to_csv(feedback_file, index=False)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ This is correct"):
                save_feedback(user_input, prediction_label, prediction_label)
                st.success("Thanks for your confirmation!")

        with col2:
            if st.button("❌ This is wrong"):
                correct_label = "Not Spam" if prediction_label == "Spam" else "Spam"
                save_feedback(user_input, prediction_label, correct_label)
                st.info("Got it! We’ll use this to improve the model.")
