import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import date
 
st.set_page_config(page_title="Home page", layout="wide")

st.title("CPI predictor using natural disasters data")
st.header("Functions")
st.markdown(
 """
 1. See the analyzation disasters data
 2. Add training data
 3. Predict the CPI based on data
 """
)
