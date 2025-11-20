import streamlit as st
import pandas as pd
import pickle

# ------------------------------------
# Load model and preprocessors
# ------------------------------------
with open("sales_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocess.pkl", "rb") as f:
    pre = pickle.load(f)

scaler = pre["scaler"]
encoder = pre["encoder"]
numeric_cols = pre["numeric_cols"]
categorical_cols = pre["categorical_cols"]
encoded_cols = pre["encoded_cols"]
input_cols = pre["input_cols"]

# ------------------------------------
# UI
# ------------------------------------
st.title("📈 Rossmann Sales Prediction App")

user_input = {}

placeholders = {
    "Store": "1",
    "DayOfWeek": "5",
    "Date": "2015-07-10",
    "Customers": "555",
    "Open": "1",
    "Promo": "1",
    "StateHoliday": "0",
    "SchoolHoliday": "0",
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": "1000",
    "CompetitionOpenSinceMonth": "9",
    "CompetitionOpenSinceYear": "2008",
    "Promo2": "1",
    "Promo2SinceWeek": "13",
    "Promo2SinceYear": "2011",
    "PromoInterval": "Jan,Apr,Jul,Oct"
}

for col in input_cols:
    user_input[col] = st.text_input(col, placeholders.get(col, "0"))

# Convert to DataFrame
df = pd.DataFrame([user_input])

# ------------------------------------
# Apply preprocessing
# ------------------------------------
# Convert numeric columns to float
df[numeric_cols] = df[numeric_cols].astype(float)

# Encode categorical
df[categorical_cols] = df[categorical_cols].astype(str)
encoded = encoder.transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

# Scale numeric
scaled = scaler.transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

# Final X matrix
X_final = pd.concat([scaled_df, encoded_df], axis=1)

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Predict"):
    pred = model.predict(X_final)[0]
    st.success(f"Predicted Sales: {pred:.2f}")
