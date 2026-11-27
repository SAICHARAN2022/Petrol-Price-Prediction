import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.keras")
SEQ = 30


@st.cache_resource
def load():
    df = pd.read_csv(os.path.join(HERE, "train_data.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").rename(columns={"Petrol (USD)": "price"}).dropna(subset=["price"])
    dates = df["Date"].reset_index(drop=True)
    prices = df["price"].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        x = np.array([scaled[i - SEQ:i] for i in range(SEQ, len(scaled))])
        y = np.array([scaled[i] for i in range(SEQ, len(scaled))])
        model = Sequential([Input((SEQ, 1)), LSTM(50, activation="tanh"), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(x, y, epochs=50, batch_size=32, validation_split=0.1, shuffle=False, verbose=0)
        model.save(MODEL_PATH)
    return dates, prices, scaler, model


def predict(target, dates, prices, scaler, model):
    def nxt(window):
        a = scaler.transform(np.array(window).reshape(-1, 1)).reshape(1, SEQ, 1)
        return float(scaler.inverse_transform(model.predict(a, verbose=0))[0][0])

    target = pd.Timestamp(target)
    if target <= dates.iloc[-1]:
        i = int((dates - target).abs().idxmin())
        if i < SEQ:
            raise ValueError(f"Date too early. Earliest supported: {dates.iloc[SEQ].date()}")
        return nxt(prices[i - SEQ:i])
    steps = int(np.ceil((target - dates.iloc[-1]).days / 7))
    window = list(prices[-SEQ:])
    for _ in range(steps):
        window = window[1:] + [nxt(window)]
    return window[-1]


st.title("Petrol price prediction")
dates, prices, scaler, model = load()
d = st.date_input("Pick a date", value=pd.Timestamp("2018-06-01"),
                  min_value=dates.iloc[SEQ].date(), max_value=pd.Timestamp("2030-12-31").date())
if st.button("Predict"):
    try:
        st.metric("Predicted price (USD)", f"${predict(d, dates, prices, scaler, model):.2f}")
    except Exception as e:
        st.error(str(e))
