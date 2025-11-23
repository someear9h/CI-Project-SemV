import streamlit as st
import pandas as pd
import pickle

# ------------------------------------
# Load model + preprocessors
# ------------------------------------
with open("sales_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

model = art["model"]
scaler = art["scaler"]
encoder = art["encoder"]
numeric_cols = art["numeric_cols"]
categorical_cols = art["categorical_cols"]
encoded_cols = art["encoded_cols"]
final_cols = art["final_cols"]

# THESE MUST MATCH YOUR NOTEBOOK
input_cols = [
    'Store','DayOfWeek','Promo','StateHoliday','SchoolHoliday',
    'StoreType','Assortment','CompetitionDistance','Promo2',
    'Year','Month','Day','WeekOfYear','CompetitionOpen',
    'Promo2Open','IsPromo2Month'
]

# ------------------------------------
# UI
# ------------------------------------
st.title("📈 Rossmann Sales Prediction App")

user_input = {}

placeholders = {
    "Store": "1",
    "DayOfWeek": "5",
    "Promo": "1",
    "StateHoliday": "0",
    "SchoolHoliday": "0",
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": "500",
    "Promo2": "1",
    "Year": "2015",
    "Month": "7",
    "Day": "10",
    "WeekOfYear": "28",
    "CompetitionOpen": "30",      # means competitor open since ~30 months
    "Promo2Open": "20",           # promo running for ~20 months
    "IsPromo2Month": "1"
}

for col in input_cols:
    user_input[col] = st.text_input(col, placeholders.get(col, "0"))

df = pd.DataFrame([user_input])

# ------------------------------------
# Preprocessing
# ------------------------------------
# Convert numeric values
df[numeric_cols] = df[numeric_cols].astype(float)

# Convert categorial to string
df[categorical_cols] = df[categorical_cols].astype(str)

# Encode categoricals
encoded = encoder.transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

# Scale numeric columns
scaled = scaler.transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

# Combine
X_final = pd.concat([scaled_df, encoded_df], axis=1)

# Ensure exact training column order
X_final = X_final[final_cols]

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Predict"):
    pred = model.predict(X_final)[0]
    st.success(f"Predicted Sales: {pred:.2f}")
