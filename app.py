import os
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "next_word_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"

# -----------------------------
# Check required files
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'next_word_model.h5' not found.")
    st.stop()

if not os.path.exists(TOKENIZER_PATH):
    st.error("❌ Tokenizer file 'tokenizer.pickle' not found.")
    st.stop()

# -----------------------------
# Load model and tokenizer
# -----------------------------
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return "⚠️ Unknown words"

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(prediction, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "❓"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔮 Next Word Prediction (LSTM)")

input_text = st.text_input("Enter a text sequence:")

if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(
            model, tokenizer, input_text, max_sequence_len
        )
        st.success(f"👉 Predicted Next Word: **{next_word}**")

model.save("next_word_model.h5")

import pickle
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

MODEL_PATH = "models/next_word_model.h5"
