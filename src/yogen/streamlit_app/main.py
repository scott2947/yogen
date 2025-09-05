import streamlit as st

import numpy as np

import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

if "device" not in st.session_state:
    st.session_state.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def get_model() -> keras.models.Sequential:
    return keras.models.load_model("./src/yogen/Modelling/Resources/model.keras")

def make_prediction(data):
    model: keras.models.Sequential = get_model()
    prediction = model.predict(data)[0][0]
    if prediction > 0.5:
        return "Price predicted to remain the same or rise"
    return "Price predicted to fall"

st.markdown("# yogen v0.1.0")
st.caption("DISCLAIMER: Model is highly unrefined")

data_input = st.text_area("Enter opening and closing prices for the previous 10 trading days, and today's opening price")
if data_input:
    data = data_input.split(",")
    data = [float(i) for i in data]
    prediction = make_prediction(np.array([data]))
    st.write(prediction)