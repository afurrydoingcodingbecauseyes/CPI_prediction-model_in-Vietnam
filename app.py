import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import date
 
st.set_page_config(page_title="Predicting CPI of Vietnam", layout="wide")


# 1. Load model
@st.cache_resource

def load_model_year():
    return joblib.load("pred_CPI_year.pkl")

def load_model_disasters():
    return joblib.load("predCPI_disasterAmount.pkl")
 
year_model = load_model_year()
 
disasters_model = load_model_disasters()

# 2. Dataframe
df = pd.read_csv("data/data.csv")
df = df[df["Start Year"] > 1967]
df  = df[["Start Year", "CPI", "Total Damage, Adjusted ('000 US$)"]]
df = df.dropna(subset = "Total Damage, Adjusted ('000 US$)")
df["Sum"] = df.groupby(["Start Year"])[["CPI"]].transform("count")
df["Total Damage, Adjusted ('000 US$)"] = df.groupby(["Start Year"])["Total Damage, Adjusted ('000 US$)"].sum()
df["cumsum"] = df["Sum"].cumsum()

# Disaster -> CPI
st.title("Predicting CPI by Disasters")


# Create Col
col1, col2 = st.columns([3, 1])

# Disaster Values Slider
with col1:
    number_of_disasters = st.slider(label="Number of disasters since 1989", min_value=0, max_value=2000, step=1)

# Predict
cpi = disasters_model.predict([[number_of_disasters]])[0]
with col2:
    st.metric("Predicted CPI", f"{cpi:.2f}")

# Graph

    # Graph drawing
fig = px.scatter(df, x="cumsum", y="CPI", title="Data and Regression Line")
x_range = [[i] for i in range(0, 750)]
y_pred = disasters_model.predict(x_range)
fig.add_scatter(
    x=list(range(0, 750)), 
    y=y_pred,
    mode="lines", name="Regression Line",
    line=dict(color="red")
)
fig.update_xaxes(title_text="Disasters Since 1989")

    # Prediction point
fig.add_scatter(
    x=[number_of_disasters],
    y=[cpi],
    mode="markers",
    name=f"Prediction: ({number_of_disasters}, {cpi:.2f})",
    marker=dict(color="red", size=10, symbol="circle"),
    text=f"{cpi:.2f}"
)

    # Vert Line
fig.add_vline(
    x=number_of_disasters, line_width=2, line_dash="dash", line_color="grey", name="Selected Year", annotation_text= f"{number_of_disasters} disasters"
)

    # Render
st.plotly_chart(fig, use_container_width=True)


# Date -> CPI
st.title("Predicting CPI by Date")

col3, col4 = st.columns([3,1])

# Disaster Values Slider
with col3:
    year = st.slider(label="Year", min_value=2022, max_value=2050, step=1)

# Predict
with col4:
    cpi = year_model.predict([[year]])[0]
    st.metric("Predicted CPI", f"{cpi:.2f}")

# Graph
    # 2.2 Graph drawing
fig = px.line(df, x="Start Year", y="CPI", title="Data and Regression Line")
x_range = [[i] for i in range(1989, 2050)]
y_pred = year_model.predict(x_range)
fig.update_xaxes(title_text="Year")

    # Regression Line
fig.add_scatter(
    x=list(range(1989, 2050)), 
    y=y_pred,
    mode="lines", name="Regression Line",
    line=dict(color="red", dash="dash"),
)
    # Prediction point
fig.add_scatter(
    x=[year],
    y=[cpi],
    mode="markers",
    name=f"Prediction: ({year}, {cpi:.2f})",
    marker=dict(color="red", size=10, symbol="circle"),
    text=f"{cpi:.2f}"
)

    # Vert Line
fig.add_vline(
    x=year, line_width=2, line_dash="dash", line_color="grey", name="Selected Year", annotation_text= f"Year: {year}"
)

    # Render
st.plotly_chart(fig, use_container_width=True)
