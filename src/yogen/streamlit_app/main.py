import streamlit as st

import numpy as np
import torch

from yogen.Modelling.lstm import LSTMModel

if "device" not in st.session_state:
    st.session_state.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def get_model():
    checkpoint = torch.load("./src/yogen/Modelling/Resources/model2.pth")
    hparams = checkpoint["hyperparams"]

    model = LSTMModel(device=st.session_state.device, **hparams).to(st.session_state.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def make_prediction(data):
    model = get_model()
    data = np.array([data.reshape(-1, 2)])
    with torch.no_grad():
        data_t = torch.tensor(data, dtype=torch.float).to(st.session_state.device)
        states = model.init_state(data_t.size(0))
        outputs, states = model(data_t, states)
        prediction = outputs.tolist()[0][0]
    if prediction > 0.5:
        return "Price predicted to remain the same or rise"
    return "Price predicted to fall"

st.markdown("# yogen v0.2.0")
st.caption("DISCLAIMER: Model is unrefined")

data_input = st.text_area("Enter opening and closing prices for the previous 10 trading days, and today's opening price")
if data_input:
    data = data_input.split(",")
    data = [float(i) for i in data]
    data.append(0)
    prediction = make_prediction(np.array([data]))
    st.write(prediction)